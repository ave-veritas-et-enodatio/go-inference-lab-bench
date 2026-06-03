# Model STMap TOML DSL Specification

Per-architecture file: `models/arch/<arch>.arch.stmap.toml`

Maps HuggingFace safetensors tensor/param names to the logical names used by our
GGUF-based architecture definitions. This file is required for safetensors loading to work for a given architecture.

## Purpose

Safetensors models from HuggingFace use different naming conventions than GGUF.
The stmap file is a one-time mapping that translates:

1. **GGUF metadata keys** ← **config.json keys** (parameter names)
2. **GGUF per-layer tensor names** ← **HF tensor names** (with `{N}` layer substitution)
3. **GGUF global tensor names** ← **HF tensor names** (no layer prefix)

By the time data reaches inference, it looks identical regardless of source format.
Architecture TOML files and block builders are unchanged.

## Section Reference

### `[architecture]`

| Field      | Type   | Required | Description |
|------------|--------|----------|-------------|
| `hf_class` | string | yes      | The `config.json` `"architectures"[0]` value that identifies this architecture (e.g. `"LlamaForCausalLM"`). Used at load time to auto-select the correct stmap file. |

### `[params]`

Maps HuggingFace `config.json` parameter keys to their GGUF-equivalent metadata keys. Both the GGUF loader and the safetensors loader resolve params through the same `[params]` section in the `.arch.toml` file — the stmap provides the bridge from HF config keys to those GGUF keys.

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

### `[gguf_metadata]`

Literal GGUF metadata values injected at load time. Used for architecture-level
constants that the converter writes into the GGUF as fixed values — they don't
come from `config.json` and don't depend on the model instance.

Format: `"gguf.metadata.key" = <literal>` (string, integer, float, or array literal)

Example:
```toml
[gguf_metadata]
"qwen35.ssm.inner_size"          = 4096
"qwen35.attention.key_length"    = 256
"qwen35.rope.dimension_sections" = [11, 11, 10, 0]
```

Use when a GGUF metadata key has the same value across every instance of an
architecture (a fixed kernel dimension, an architecture-wide constant).
Per-instance values that need to be computed at load time belong in
`[[derived_metadata]]` instead.

### `[[derived_metadata]]`

GGUF metadata values *computed* at load time from one or more `config.json`
fields, rather than being a direct 1:1 mapping or a literal. Used when the
converter applies a non-trivial transformation from source config to GGUF
metadata.

Each entry registers an op handler (looked up in the `derivedMetadataOps`
Go-side registry) and runs it once during the safetensors load. The handler
reads from `config.json` and produces a value stored under `target` in the
same param namespace that `[params]` and `[gguf_metadata]` populate. The
loader's `[param]` debug dump shows derived entries alongside the rest, so
visual GGUF-vs-safetensors diffing remains a single-flat-list operation.

Format:
```toml
[[derived_metadata]]
target = "gguf.metadata.key"       # required: the GGUF key to populate
op     = "<op_name>"               # required: handler name from the registry
# ... op-specific params (see "Registered ops" below)
```

#### Registered ops

| `op` value | Purpose | Required op-params |
|---|---|---|
| `string_array_eq` | Translate a `config.json` string array into a `[]bool` by comparing each element to a constant. Used for `layer_types: ["sliding_attention", "full_attention", ...]` → `sliding_window_pattern: [true, false, ...]` patterns. | `source` (config.json dotted key path), `match` (the string compared element-wise) |
| `copy_param` | Copy one `config.json` value verbatim under a different GGUF key. Used when a single source value populates multiple GGUF keys — the `[params]` map can only express a 1:1 HF-key→GGUF-key relationship, so the second target needs `copy_param`. | `source` (config.json dotted key path) |

Adding a new op is a Go-side change: register a new handler in
`src/internal/inference/arch/model_reader_safetensors_derived.go`'s
`derivedMetadataOps` map, mirroring the block builder registry pattern. The
TOML side stays declarative.

Example (Gemma 4):
```toml
# text_config.layer_types is ["sliding_attention", "full_attention", ...].
# Convert to a bool array used by [layers.routing].pattern in gemma4.arch.toml.
[[derived_metadata]]
target = "gemma4.attention.sliding_window_pattern"
op     = "string_array_eq"
source = "text_config.layer_types"
match  = "sliding_attention"

# global_head_dim is mapped to gemma4.attention.key_length in [params]. The
# converter ALSO writes the same value to gemma4.rope.dimension_count — but
# [params] can only assign one target per source, so the second target needs
# copy_param.
[[derived_metadata]]
target = "gemma4.rope.dimension_count"
op     = "copy_param"
source = "text_config.global_head_dim"
```

### `[[derived_tensors]]`

GGUF tensors *synthesized* at load time — no source tensor exists in the
safetensors file. Used when the converter generates extra tensors procedurally
(e.g., `convert_hf_to_gguf.py`'s `generate_extra_tensors` method), typically
for compatibility shims between the source model's math and the inference
engine's primitives.

Each entry registers an op handler (looked up in the `derivedTensorOps`
Go-side registry) that returns an F32 array and its shape. The synthesized
tensor appears in the loader's tensor enumeration (`TensorCount`,
`TensorNames`, `TensorSpec`, `ReadTensor`) alongside safetensors-sourced
tensors — downstream code is format-agnostic and cannot tell the two apart.
Synthesized tensors are stored as F32 little-endian bytes.

Format:
```toml
[[derived_tensors]]
target = "gguf.tensor.name"        # required: the GGUF tensor name (typically global)
op     = "<op_name>"               # required: handler name from the registry
# ... op-specific params (see "Registered ops" below)
```

#### Registered ops

| `op` value | Purpose | Required op-params |
|---|---|---|
| `rope_freqs_proportional` | Synthesize the `rope_freqs.weight` tensor for models using "proportional" RoPE on top of ggml's full-rotary NeoX primitive. Produces an F32 array of length `head_dim / 2` with `1.0` repeated `n_rot` times followed by `1e30` repeated `n_unrot` times, where `n_rot = int(head_dim × partial_rotary_factor / 2)`. The `1e30` values divide the rotation frequency so dramatically that the corresponding pairs effectively don't rotate, implementing partial-rotary RoPE on top of a full-rotary primitive op. | `head_dim_source` (config.json dotted key for the full head dim), `partial_rotary_source` (config.json dotted key for the partial rotary factor) |

Adding a new op is a Go-side change in the same file as `derivedMetadataOps`,
under `derivedTensorOps`.

Example (Gemma 4):
```toml
# Gemma 4 full-attention layers use proportional RoPE (only ~25% of each head's
# dimensions are rotated). The reference converter computes a freq_factors
# tensor that masks the unrotated pairs by setting their per-pair theta divider
# to 1e30 (effective theta ≈ 0, no rotation). We reproduce that computation at
# load time from the rope params in config.json.
[[derived_tensors]]
target                = "rope_freqs.weight"
op                    = "rope_freqs_proportional"
head_dim_source       = "text_config.global_head_dim"
partial_rotary_source = "text_config.rope_parameters.full_attention.partial_rotary_factor"
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

- STMap files use the `.arch.stmap.toml` extension (e.g. `llama.arch.stmap.toml`).
- The base name must match an existing `.arch.toml` architecture.
- Tensor naming conventions are architecture-level — all variants of the same
  architecture (e.g., all Llama 3.x models) share one stmap file.
- One stmap file covers one `hf_class` → one architecture mapping.
