# Model Remap TOML DSL Specification

Per-architecture file: `models/arch/<arch>.arch.remap.toml`

Maps HuggingFace safetensors tensor/param names to the logical names used by our
GGUF-based architecture definitions. This file is required for safetensors loading to work for a given architecture.

## Purpose

Safetensors models from HuggingFace use different naming conventions than GGUF.
The remap file is a one-time mapping that translates:

1. **GGUF metadata keys** ← **config.json keys** (parameter names)
2. **GGUF per-layer tensor names** ← **HF tensor names** (with `{N}` layer substitution)
3. **GGUF global tensor names** ← **HF tensor names** (no layer prefix)

By the time data reaches inference, it looks identical regardless of source format.
Architecture TOML files and block builders are unchanged.

## Section Reference

### `[architecture]`

| Field      | Type   | Required | Description |
|------------|--------|----------|-------------|
| `hf_class` | string | yes      | The `config.json` `"architectures"[0]` value that identifies this architecture (e.g. `"LlamaForCausalLM"`). Used at load time to auto-select the correct remap file. |

### `[params]`

Maps HuggingFace `config.json` parameter keys to their GGUF-equivalent metadata keys. Both the GGUF loader and the safetensors loader resolve params through the same `[params]` section in the `.arch.toml` file — the remap provides the bridge from HF config keys to those GGUF keys.

Format: `"hf_config_key" = "gguf.metadata.key"`

Example:
```toml
[params]
"num_hidden_layers"     = "llama.block_count"
"hidden_size"           = "llama.embedding_length"
"intermediate_size"     = "llama.feed_forward_length"
"num_attention_heads"   = "llama.attention.head_count"
"num_key_value_heads"   = "llama.attention.head_count_kv"
"rms_norm_eps"          = "llama.attention.layer_norm_rms_epsilon"
"rope_theta"            = "llama.rope.freq_base"
```

These mappings are consulted when loading `config.json` to populate the same
param namespace that the GGUF loader derives from the GGUF metadata block.

### `[layer_prefix]`

Defines the HF and GGUF per-layer key prefix templates. The `{N}` token (documented as `{N}` in the HF prefix and `@{layer_idx}` in the GGUF prefix) is substituted with the zero-based layer index during tensor resolution.

| Field    | Type   | Required | Description |
|----------|--------|----------|-------------|
| `hf`     | string | yes      | HF layer prefix with `{N}` substitution (e.g. `"model.layers.{N}."`) |
| `gguf`   | string | no       | GGUF layer prefix with `@{layer_idx}` substitution. Defaults to `"blk.@{layer_idx}."`. |

Example:
```toml
[layer_prefix]
hf = "model.layers.{N}."
# gguf defaults to "blk.@{layer_idx}."
```

The `{N}` / `@{layer_idx}` substitution is the only one needed. A per-layer tensor in our system
named `attn_q.weight` becomes `model.layers.0.self_attn.q_proj.weight` (HF) or `blk.0.attn_q.weight` (GGUF) by
combining the respective prefix with the `[tensors]` mapping.

### `[tensors]`

Maps our short per-layer tensor names (used in `.arch.toml` weight bindings) to
HF short tensor names (without the layer prefix). The loader concatenates
`[layer_prefix].hf` (with `{N}` substituted) + `[tensors].<short-name>` to produce
the full safetensors tensor key.

Format: `"our_short_name" = "hf_short_name"`

Example:
```toml
[tensors]
"attn_q.weight"      = "self_attn.q_proj.weight"
"attn_k.weight"      = "self_attn.k_proj.weight"
"attn_v.weight"      = "self_attn.v_proj.weight"
"attn_output.weight" = "self_attn.o_proj.weight"
"attn_norm.weight"   = "input_layernorm.weight"
"ffn_gate.weight"    = "mlp.gate_proj.weight"
"ffn_up.weight"      = "mlp.up_proj.weight"
"ffn_down.weight"    = "mlp.down_proj.weight"
"ffn_norm.weight"    = "post_attention_layernorm.weight"
```

Resolution example for layer 5:
- Our name: `attn_q.weight`
- HF prefix: `model.layers.{N}.` → `model.layers.5.`
- HF short: `self_attn.q_proj.weight`
- Resolved: `model.layers.5.self_attn.q_proj.weight`

### `[tensors.global]`

Maps our short global (non-layered) tensor names to HF full tensor names.
These do **not** use the layer prefix — the HF name is used as-is.

Format: `"our_short_name" = "hf_full_name"`

Example:
```toml
[tensors.global]
"token_embd.weight"  = "model.embed_tokens.weight"
"output_norm.weight" = "model.norm.weight"
"output.weight"      = "lm_head.weight"
```

## Complete Example: Llama

```toml
[architecture]
hf_class = "LlamaForCausalLM"

[params]
"num_hidden_layers"     = "llama.block_count"
"hidden_size"           = "llama.embedding_length"
"intermediate_size"     = "llama.feed_forward_length"
"num_attention_heads"   = "llama.attention.head_count"
"num_key_value_heads"   = "llama.attention.head_count_kv"
"rms_norm_eps"          = "llama.attention.layer_norm_rms_epsilon"
"rope_theta"            = "llama.rope.freq_base"

[layer_prefix]
hf = "model.layers.{N}."

[tensors]
"attn_q.weight"      = "self_attn.q_proj.weight"
"attn_k.weight"      = "self_attn.k_proj.weight"
"attn_v.weight"      = "self_attn.v_proj.weight"
"attn_output.weight" = "self_attn.o_proj.weight"
"attn_norm.weight"   = "input_layernorm.weight"
"ffn_gate.weight"    = "mlp.gate_proj.weight"
"ffn_up.weight"      = "mlp.up_proj.weight"
"ffn_down.weight"    = "mlp.down_proj.weight"
"ffn_norm.weight"    = "post_attention_layernorm.weight"

[tensors.global]
"token_embd.weight"  = "model.embed_tokens.weight"
"output_norm.weight" = "model.norm.weight"
"output.weight"      = "lm_head.weight"
```

## Naming Conventions

- Remap files use the `.arch.remap.toml` extension (e.g. `llama.arch.remap.toml`).
- The base name must match an existing `.arch.toml` architecture.
- Tensor naming conventions are architecture-level — all variants of the same
  architecture (e.g., all Llama 3.x models) share one remap file.
- One remap file covers one `hf_class` → one architecture mapping.
