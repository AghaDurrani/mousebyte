"""Gemma 3 270M/1B in plain PyTorch: equations, LoRA and HF state-dict loading.

No Transformers model wrapper. Transformers is used only for the tokenizer.
causal=True reproduces the pretrained attention; False enables classification
with bidirectional global attention and symmetric local windows.
"""
import json
import math
import os
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

CACHE_DIR = str(Path(__file__).resolve().parents[1] / 'data' / 'huggingface')


class LoraAdaptor(nn.Module):
    def __init__(self, dim_in, dim_out, r=8, alpha=16):
        super().__init__()
        self.A = nn.Linear(dim_in, r, bias=False)
        self.B = nn.Linear(r, dim_out, bias=False)
        nn.init.zeros_(self.B.weight)
        self.scale = alpha / r

    def forward(self, x):
        # Delta W x = (alpha / rank) B A x; initially zero.
        return self.B(self.A(x)) * self.scale


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        # Gemma stores an offset: y = (1 + weight) x / RMS(x).
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (y * (1 + self.weight.float())).to(x.dtype)


def rotary_embeddings(positions, head_dim, theta, dtype):
    with torch.autocast(device_type=positions.device.type, enabled=False):
        frequency = theta ** (-torch.arange(0, head_dim, 2, device=positions.device).float() / head_dim)
        angle = positions.float()[..., None] * frequency
        angle = torch.cat((angle, angle), dim=-1)
    return angle.cos().to(dtype)[:, None], angle.sin().to(dtype)[:, None]


def apply_rope(x, cos, sin):
    first, second = x.chunk(2, dim=-1)
    return x * cos.to(x.dtype) + torch.cat((-second, first), -1) * sin.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, config, layer_index):
        super().__init__()
        d = config['hidden_size']
        self.head_dim = config['head_dim']
        self.heads = config['num_attention_heads']
        self.kv_heads = config['num_key_value_heads']
        self.scale = config['query_pre_attn_scalar'] ** -0.5
        self.dropout = config.get('attention_dropout', 0.0)
        types = config.get('layer_types')
        self.local = (types[layer_index] == 'sliding_attention' if types else
                      (layer_index + 1) % config.get('sliding_window_pattern', 6) != 0)
        self.window = config['sliding_window']
        self.theta = config.get('rope_local_base_freq', 10000) if self.local else config['rope_theta']
        for name, size in [('q', self.heads), ('k', self.kv_heads), ('v', self.kv_heads)]:
            setattr(self, name + '_proj', nn.Linear(d, size * self.head_dim, bias=config.get('attention_bias', False)))
            setattr(self, 'lora_' + name, LoraAdaptor(d, size * self.head_dim))
        self.o_proj = nn.Linear(self.heads * self.head_dim, d, bias=config.get('attention_bias', False))
        self.q_norm = RMSNorm(self.head_dim, config['rms_norm_eps'])
        self.k_norm = RMSNorm(self.head_dim, config['rms_norm_eps'])

    def forward(self, x, valid, positions, causal):
        b, length, _ = x.shape
        q = (self.q_proj(x) + self.lora_q(x)).view(b, length, self.heads, self.head_dim).transpose(1, 2)
        k = (self.k_proj(x) + self.lora_k(x)).view(b, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = (self.v_proj(x) + self.lora_v(x)).view(b, length, self.kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = rotary_embeddings(positions, self.head_dim, self.theta, q.dtype)
        q, k = apply_rope(self.q_norm(q), cos, sin), apply_rope(self.k_norm(k), cos, sin)
        repeats = self.heads // self.kv_heads
        k, v = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)

        def attend(qpart, kpart, vpart, mask):
            # softmax(Q K^T / sqrt(query_pre_attn_scalar) + mask) V.
            return F.scaled_dot_product_attention(qpart, kpart, vpart, attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0, scale=self.scale, is_causal=False)

        if self.local:
            # Compute local attention in query blocks; never allocate an L x L mask.
            # This is exact windowed attention, not splitting the document for pooling.
            outputs = []
            for start in range(0, length, 256):
                end = min(start + 256, length)
                left = max(0, start - self.window + 1)
                right = end if causal else min(length, end + self.window - 1)
                qi = torch.arange(start, end, device=x.device)[:, None]
                ki = torch.arange(left, right, device=x.device)[None, :]
                allowed = (qi - ki).abs() < self.window
                if causal:
                    allowed = allowed & (ki <= qi)
                mask = allowed[None, None] & valid[:, None, None, left:right]
                outputs.append(attend(q[:, :, start:end], k[:, :, left:right], v[:, :, left:right], mask))
            out = torch.cat(outputs, dim=2)
        else:
            mask = valid[:, None, None, :]
            if causal:
                mask = mask & torch.ones(length, length, device=x.device, dtype=torch.bool).tril()[None, None]
            out = attend(q, k, v, mask)
        return self.o_proj(out.transpose(1, 2).reshape(b, length, -1))


class GeGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, intermediate = config['hidden_size'], config['intermediate_size']
        self.gate_proj = nn.Linear(d, intermediate, bias=False)
        self.up_proj = nn.Linear(d, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, d, bias=False)
        self.lora_gate = LoraAdaptor(d, intermediate)
        self.lora_up = LoraAdaptor(d, intermediate)
        self.lora_down = LoraAdaptor(intermediate, d)

    def forward(self, x):
        # GeGLU(x) = W_down [GELU(W_gate x) * W_up x], with LoRA on each W.
        gate = F.gelu(self.gate_proj(x) + self.lora_gate(x), approximate='tanh')
        hidden = gate * (self.up_proj(x) + self.lora_up(x))
        return self.down_proj(hidden) + self.lora_down(hidden)


class DecoderLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.self_attn = Attention(config, index)
        self.mlp = GeGLU(config)
        for name in ('input_layernorm', 'post_attention_layernorm',
                     'pre_feedforward_layernorm', 'post_feedforward_layernorm'):
            setattr(self, name, RMSNorm(config['hidden_size'], config['rms_norm_eps']))

    def forward(self, x, valid, positions, causal):
        x = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x), valid, positions, causal))
        return x + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))


class Gemma(nn.Module):
    def __init__(self, config, causal=False):
        super().__init__()
        if config.get('rope_scaling') or config.get('attn_logit_softcapping'):
            raise ValueError('This implementation targets unscaled Gemma 3 1B RoPE and attention.')
        if config['hidden_activation'] != 'gelu_pytorch_tanh':
            raise ValueError('Expected Gemma 3 tanh-approximate GELU.')
        self.config, self.causal = config, causal
        self.gradient_checkpointing = False
        self.embed_tokens = nn.Embedding(config['vocab_size'], config['hidden_size'], padding_idx=config.get('pad_token_id', 0))
        self.layers = nn.ModuleList([DecoderLayer(config, i) for i in range(config['num_hidden_layers'])])
        self.norm = RMSNorm(config['hidden_size'], config['rms_norm_eps'])

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        if input_ids.shape[1] > self.config['max_position_embeddings']:
            raise ValueError('Input exceeds Gemma context length; choose crops explicitly.')
        valid = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        positions = (torch.arange(input_ids.shape[1], device=input_ids.device)[None]
                     if position_ids is None else position_ids)
        x = self.embed_tokens(input_ids)
        x = x * torch.tensor(math.sqrt(self.config['hidden_size']), dtype=x.dtype, device=x.device)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, valid, positions, self.causal, use_reentrant=False)
            else:
                x = layer(x, valid, positions, self.causal)
        return self.norm(x)

    @classmethod
    def from_pretrained(cls, model_name='google/gemma-3-1b-pt', causal=False):
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        folder = Path(snapshot_download(model_name, cache_dir=CACHE_DIR,
            token=os.getenv('HF_TOKEN_WRITE') or os.getenv('HF_TOKEN'),
            allow_patterns=['*.json', '*.safetensors', 'tokenizer.model']))
        config = json.loads((folder / 'config.json').read_text())
        model = cls(config, causal=causal)
        expected = {n for n, _ in model.named_parameters() if '.lora_' not in n}
        loaded = set()
        for path in sorted(folder.glob('*.safetensors')):
            weights = load_file(str(path))
            state = {}
            for name, value in weights.items():
                if name == 'lm_head.weight':  # Tied vocabulary head; classifier uses hidden states.
                    continue
                key = name.removeprefix('model.')
                if key not in expected or key in loaded:
                    raise ValueError(f'Unexpected/duplicate pretrained weight: {name}')
                state[key] = value
            model.load_state_dict(state, strict=False)
            loaded.update(state)
            del state, weights
        if loaded != expected:
            raise ValueError(f'Missing pretrained weights: {sorted(expected - loaded)}')
        model.tokenizer = AutoTokenizer.from_pretrained(folder, padding_side='right')
        print(f'Loaded all {len(loaded)} Gemma backbone tensors; LoRA starts at zero.', flush=True)
        return model


class GemmaClassifier(nn.Module):
    def __init__(self, model_name='google/gemma-3-270m', config=None):
        super().__init__()
        self.backbone = Gemma(config, causal=False) if config is not None else Gemma.from_pretrained(model_name, causal=False)
        self.lm_head = nn.Linear(self.backbone.config['hidden_size'], 1)

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = input_ids.ne(self.backbone.config.get('pad_token_id', 0))
        hidden = self.backbone(input_ids, attention_mask)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.lm_head(pooled)
