"""Qwen2.5-0.5B backbone written in PyTorch;
"""
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer


CACHE_DIR = str(Path(__file__).resolve().parents[1] / "data" / "huggingface")


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # RMSNorm(x) = weight * x / sqrt(mean(x^2) + eps).
        original_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x.to(original_dtype)


def rotary_embeddings(position_ids, head_dim, theta, dtype):
    # Frequency_i = theta^(-2i / head_dim); angle = position * frequency.
    with torch.autocast(device_type=position_ids.device.type, enabled=False):
        frequencies = theta ** (-torch.arange(0, head_dim, 2,
                                              device=position_ids.device).float() / head_dim)
        angles = position_ids.float().unsqueeze(-1) * frequencies
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos().to(dtype)[:, None], angles.sin().to(dtype)[:, None]


def apply_rope(x, cos, sin):
    # Rotate paired coordinates from the first and second halves of each head.
    first, second = x.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return x * cos.to(x.dtype) + rotated * sin.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = dim // self.num_heads
        self.dropout = config["attention_dropout"]
        self.lora_q = LoraAdaptor(r=8,alpha=16,dim_in=dim, dim_out=self.num_heads*self.head_dim)
        self.lora_v = LoraAdaptor(r=8,alpha=16,dim_in=dim, dim_out=self.num_kv_heads*self.head_dim)     
        self.lora_k = LoraAdaptor(r=8,alpha=16,dim_in=dim, dim_out=self.num_kv_heads*self.head_dim)      
        self.q_proj = nn.Linear(dim, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, allowed, cos, sin):
        batch, length, dim = x.shape
        q = (self.q_proj(x) + self.lora_q(x)).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = (self.k_proj(x) + self.lora_k(x)).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = (self.v_proj(x) + self.lora_v(x)).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        # Grouped-query attention: 14 query heads share 2 key/value heads.
        repeats = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

        # Attention(Q,K,V) = softmax(Q K^T / sqrt(head_dim) + mask) V.
        # Padding mask only (True = attend); attention remains bidirectional.
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, is_causal=False,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(batch, length, dim)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim, intermediate = config["hidden_size"], config["intermediate_size"]
        self.gate_proj = nn.Linear(dim, intermediate, bias=False)
        self.up_proj = nn.Linear(dim, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, dim, bias=False)
        self.lora_gate = LoraAdaptor(r=8,alpha=16,dim_in=dim,dim_out=intermediate)
        self.lora_up = LoraAdaptor(r=8,alpha=16,dim_in=dim,dim_out=intermediate)
        self.lora_down = LoraAdaptor(r=8,alpha=16,dim_in=intermediate,dim_out=dim)
        
    def forward(self, x):
        # SwiGLU(x) = W_down [SiLU(W_gate x) * (W_up x)].
        gate = self.gate_proj(x) + self.lora_gate(x)
        silu = gate * torch.sigmoid(gate)
        x = silu * (self.up_proj(x) + self.lora_up(x))
        return self.down_proj(x) + self.lora_down(x)


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim, eps = config["hidden_size"], config["rms_norm_eps"]
        self.input_layernorm = RMSNorm(dim, eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(dim, eps)
        self.mlp = SwiGLU(config)

    def forward(self, x, allowed, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), allowed, cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen(nn.Module):
    def __init__(self, config):
        super().__init__()
        if (config.get("use_sliding_window", False) or config.get("rope_scaling")
                or config["hidden_act"] != "silu"):
            raise ValueError("This implementation supports standard Qwen2.5-0.5B only.")
        self.config = config
        self.hidden_dim = config["hidden_size"]
        self.embed_tokens = nn.Embedding(config["vocab_size"], self.hidden_dim)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config["num_hidden_layers"])])
        self.norm = RMSNorm(self.hidden_dim, config["rms_norm_eps"])

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        """Return [batch, tokens, hidden_dim]. Mask padding when you add pooling."""
        length = input_ids.shape[1]
        if length > self.config["max_position_embeddings"]:
            raise ValueError("Input exceeds the model's context length.")
        x = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(length, device=x.device).unsqueeze(0)
        head_dim = self.hidden_dim // self.config["num_attention_heads"]
        cos, sin = rotary_embeddings(position_ids, head_dim, self.config["rope_theta"], x.dtype)
        # Bidirectional: every query can attend to every non-padding key.
        # [batch, 1, 1, length] broadcasts over heads and query positions.
        allowed = None if attention_mask is None else attention_mask[:, None, None, :].bool()
        for layer in self.layers:
            x = layer(x, allowed, cos, sin)
        return self.norm(x)

    @classmethod
    def from_pretrained(cls, model_name="Qwen/Qwen2.5-0.5B", dtype=torch.float32):
        folder = Path(snapshot_download(model_name, cache_dir=CACHE_DIR,
                                       allow_patterns=["*.json", "*.safetensors", "*.txt"]))
        config = json.loads((folder / "config.json").read_text())
        model = cls(config).to(dtype=dtype)
        weights = {}
        for shard in sorted(folder.glob("*.safetensors")):
            weights.update(load_file(str(shard)))
        # Checkpoints prefix backbone parameters with 'model.'. The LM head is unused.
        state_dict = {name.removeprefix("model."): value for name, value in weights.items()
                      if name != "lm_head.weight"}
        model.load_state_dict(state_dict, strict=False)
        model.tokenizer = AutoTokenizer.from_pretrained(str(folder), padding_side="right")
        if model.tokenizer.pad_token_id is None:
            model.tokenizer.pad_token = model.tokenizer.eos_token
        return model


class QwenClassifier(nn.Module):
    def __init__(
        self,
        model_name="Qwen/Qwen2.5-0.5B",
        dtype=torch.float32,
        config=None,
    ):
        super().__init__()

        self.backbone = Qwen(config) if config is not None else Qwen.from_pretrained(
            model_name=model_name,
            dtype=dtype,
        )
        self.lm_head = nn.Linear(self.backbone.hidden_dim, 1)

    def forward(self, input_ids, attention_mask=None,position_ids=None):
        B,S = input_ids.shape
        hidden = self.backbone(input_ids=input_ids,
                               attention_mask=attention_mask,
                               position_ids = position_ids) # B, S, E

        if attention_mask is None:
            hidden = hidden.mean(dim=1) # B,E
        else:
            hidden = hidden*(attention_mask).reshape(B,S,1)
            hidden = hidden.sum(dim=1) # B, E
            hidden = hidden / attention_mask.sum(dim=1,keepdim=True)

        logits = self.lm_head(hidden)
        return logits 
        


class LoraAdaptor(nn.Module):
    def __init__(self,alpha,r, dim_in, dim_out):
        super().__init__()
        self.alpha = alpha 
        self.r = r 
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.B = nn.Linear(r,dim_out,bias=False)
        self.A = nn.Linear(dim_in,r,bias=False)
        nn.init.zeros_(self.B.weight)
    
    def forward(self, x):
        return (self.alpha/self.r)*self.B(self.A(x))




