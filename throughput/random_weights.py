"""Models with random weights for throughput measurements: dense FP16/BF16, or SPHQuant W2 with random codes.

Only the architecture matters for decode speed, so no checkpoint is downloaded. The model is built on the meta
device, its decoder projections are optionally fused (q, k, v into qkv_proj and gate, up into gate_up_proj, for dense
and W2 alike), and every tensor is then created on the GPU: dense weights N(0, 0.02), W2 projections from random
sphquant-w2-v1 codes (uniform signs, directions and radius codes; decoded weights of about the same scale), norms 1.
"""
import json
import types

import torch
from torch import nn

INIT_STD = 0.02


def random_export(n, k, gen, device):
    """Random sphquant-w2-v1 fields of an [n, k] linear layer (signs, directions and 4-bit radius codes uniform)."""
    from sphquant.packing import pack

    kv = k // 8

    def ints(high, shape):
        return torch.randint(0, high, shape, generator=gen, device=device, dtype=torch.int64).to(torch.uint8)

    def uniform(lo, hi, shape):
        return (lo + (hi - lo) * torch.rand(shape, generator=gen, device=device)).to(torch.float16)

    codebook = torch.randn(64, 8, generator=gen, device=device).abs()
    # |w| = radius * |codebook entry|, entries about 1/sqrt(8): radii around 0.057 give weights of std about 0.02
    state = {"signs": ints(256, (n, kv)), "dir_idx": ints(64, (n, kv)), "codes": ints(16, (n, kv)),
             "step": uniform(0.003, 0.005, (n, k // 128)), "rmin": uniform(0.015, 0.025, (n, k // 128)),
             "codebook": (codebook / codebook.norm(dim=1, keepdim=True)).to(torch.float16)}
    return pack(state)


def _qwen3_attention_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None,
                             **kwargs):
    """Qwen3Attention.forward of transformers 5.19 with one fused q/k/v projection."""
    from transformers.models.qwen3 import modeling_qwen3 as mq

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    q, k, v = self.qkv_proj(hidden_states).split(self.qkv_sizes, dim=-1)
    query_states = self.q_norm(q.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(k.view(hidden_shape)).transpose(1, 2)
    value_states = v.view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = mq.apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
    attention_interface = mq.ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation,
                                                                   mq.eager_attention_forward)
    attn_output, attn_weights = attention_interface(
        self, query_states, key_states, value_states, attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout, scaling=self.scaling,
        sliding_window=self.sliding_window, **kwargs)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), attn_weights


def _gated_mlp_forward(self, x):
    """Qwen3MLP.forward with one fused gate/up projection; a W2 one also applies SiLU and the product in its kernel
    (unless SPH_GLU=0)."""
    from cuda_kernel import packed_linear as pl

    config = getattr(self, "config", None)
    if pl.GLU and isinstance(self.gate_up_proj, pl.PackedSPHLinear) and getattr(config, "hidden_act", None) == "silu":
        return self.down_proj(self.gate_up_proj.forward_glu(x))
    gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
    return self.down_proj(self.act_fn(gate) * up)


def projections(model):
    """{name: (out_features, in_features)} of every decoder projection, after fusion if any."""
    return {name: (m.out_features, m.in_features) for name, m in model.named_modules()
            if isinstance(m, nn.Linear) and ".layers." in name}


def fuse_projections(model):
    """Replace q/k/v and gate/up of every Qwen3 decoder layer by one fused nn.Linear each (still on meta)."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3MLP

    fused = 0
    for module in model.modules():
        if isinstance(module, Qwen3Attention):
            parts = [module.q_proj, module.k_proj, module.v_proj]
            module.qkv_sizes = [p.out_features for p in parts]
            module.qkv_proj = _fused_linear(parts)
            del module.q_proj, module.k_proj, module.v_proj
            module.forward = types.MethodType(_qwen3_attention_forward, module)
            fused += 1
        elif isinstance(module, Qwen3MLP):
            module.gate_up_proj = _fused_linear([module.gate_proj, module.up_proj])
            module.fused_parts = [module.gate_proj.out_features, module.up_proj.out_features]
            del module.gate_proj, module.up_proj
            module.forward = types.MethodType(_gated_mlp_forward, module)
            fused += 1
    if not fused:
        raise ValueError("--fuse: no Qwen3 attention or MLP modules in this model")
    return fused


def _fused_linear(parts):
    m = nn.Linear(parts[0].in_features, sum(p.out_features for p in parts), bias=parts[0].bias is not None,
                  device="meta")
    m.fused_parts = [p.out_features for p in parts]
    return m


def build(config_path, method, dtype, seed, fuse=True, device="cuda"):
    """Random-weight model from a config.json; METHOD dense or w2. Returns (model, number of W2 modules)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from cuda_kernel.packed_linear import PackedSPHLinear, fuse as fuse_packed

    with open(config_path) as f:
        cfg = json.load(f)
    config = AutoConfig.for_model(cfg.pop("model_type"), **cfg)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype, attn_implementation="sdpa")
    if fuse:
        fuse_projections(model)
    gen = torch.Generator(device=device).manual_seed(seed)
    packed = 0
    for name, m in list(model.named_modules()):
        if not (isinstance(m, nn.Linear) and ".layers." in name):
            continue
        if method == "w2":
            parts = getattr(m, "fused_parts", [m.out_features])
            new = [PackedSPHLinear.from_export(random_export(n, m.in_features, gen, device)) for n in parts]
            new = new[0] if len(new) == 1 else fuse_packed(new)
            parent, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent), child, new)
            packed += 1
        else:
            m.weight = nn.Parameter(torch.empty(m.weight.shape, dtype=dtype, device=device).normal_(
                0, INIT_STD, generator=gen))
    _materialize(model, config, dtype, gen, device)
    if getattr(config, "tie_word_embeddings", False):
        model.tie_weights()
    return model.eval(), packed


def _materialize(model, config, dtype, gen, device):
    """Create every tensor still on meta: embeddings and linear weights N(0, 0.02), norm weights 1, rotary
    embeddings rebuilt from the config."""
    for name, m in list(model.named_modules()):
        if "Rotary" in type(m).__name__ and any(b.is_meta for b in m.buffers()):
            with torch.device(device):
                new = type(m)(config=m.config)
            parent, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent) if parent else model, child, new)
            continue
        for pname, p in list(m.named_parameters(recurse=False)):
            if not p.is_meta:
                continue
            t = torch.empty(p.shape, dtype=dtype, device=device)
            if isinstance(m, (nn.Linear, nn.Embedding)):
                t.normal_(0, INIT_STD, generator=gen)
            else:
                t.fill_(1.0)
            setattr(m, pname, nn.Parameter(t, requires_grad=False))
        for bname, b in list(m.named_buffers(recurse=False)):
            if b.is_meta:
                raise ValueError(f"cannot create buffer {name}.{bname} of {type(m).__name__}")
