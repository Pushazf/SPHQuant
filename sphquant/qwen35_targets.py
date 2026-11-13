"""Qwen3.5 (HF Qwen3_5ForConditionalGeneration) quantization targets and direct decoder-layer calls."""
from transformers.masking_utils import create_causal_mask

MODEL_REV = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"  # Qwen/Qwen3.5-9B revision used by the pipeline
LAYERS_PREFIX = "model.language_model.layers"
MLP = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
TARGETS = {
    "linear_attention": ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj") + MLP,
    "full_attention": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj") + MLP,
}
REAL_COUNTS = (200, 6_912_212_992)  # (modules, weights) for Qwen3.5-9B


def text_config(config):
    return getattr(config, "text_config", config)


def text_layers(model):
    """Decoder layers of a Qwen3_5ForConditionalGeneration."""
    return model.model.language_model.layers


def layer_types(model):
    return list(text_config(model.config).layer_types)


def module_name(layer_idx, rel):
    """Full HF module path (checkpoint key without '.weight')."""
    return f"{LAYERS_PREFIX}.{layer_idx}.{rel}"


def layer_targets(layer, layer_type):
    """{relative name: nn.Linear} of the quantized modules of one decoder layer."""
    return {rel: layer.get_submodule(rel) for rel in TARGETS[layer_type]}


def expected_counts(config):
    """(number of target modules, number of target weights) implied by a Qwen3.5 config."""
    t = text_config(config)
    h = t.hidden_size
    mlp = 3 * h * t.intermediate_size
    key_dim = t.linear_key_head_dim * t.linear_num_key_heads
    value_dim = t.linear_value_head_dim * t.linear_num_value_heads
    linear = (2 * key_dim + value_dim) * h + value_dim * h + h * value_dim + mlp
    q = t.num_attention_heads * t.head_dim
    kv = t.num_key_value_heads * t.head_dim
    full = 2 * q * h + 2 * kv * h + h * q + mlp  # q_proj carries the output gate (2x rows)
    per_layer = {"linear_attention": linear, "full_attention": full}
    types = list(t.layer_types)
    return sum(len(TARGETS[x]) for x in types), sum(per_layer[x] for x in types)


def rotary(model, embeds, position_ids):
    """(cos, sin) in embeds.dtype for 3-axis position ids [3, B, S]; computed in FP32 from the FP32 inv_freq."""
    return model.model.language_model.rotary_emb(embeds, position_ids)


def layer_call(layer, layer_type, hidden, position_embeddings):
    """One decoder layer without cache or padding, with the mask the full model would build for its type."""
    mask = None
    if layer_type == "full_attention":
        mask = create_causal_mask(config=layer.self_attn.config, inputs_embeds=hidden, attention_mask=None,
                                  past_key_values=None)
    return layer(hidden, position_embeddings=position_embeddings, attention_mask=mask)
