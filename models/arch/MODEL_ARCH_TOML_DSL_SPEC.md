# Model Architecture TOML DSL Specification

Language-agnostic implementation guide for the model architecture TOML DSL. A developer with this document and the Go source as reference can implement a compatible loader in any language.

---

## Context: Layer Loop

The framework drives a per-layer loop. Each layer applies, in order:

```
cur   = rms_norm(x) * common_weights["attn_norm"]
cur   = block.Build(cur, ...)         ← block builder
x     = x + cur                       ← residual
ffn_x = rms_norm(x) * common_weights["ffn_norm"]
ffn_x = ffn.Build(ffn_x, ...)         ← FFN builder
x     = x + ffn_x                     ← residual
```

**Builder I/O contract** (block and FFN both): input `[n_embd, n_tokens]` (pre-normed residual), output `[n_embd, n_tokens]` (contribution before residual add).

Weights are resolved per-layer by concatenating the layer prefix (`blk.@{layer_idx}.`) with each weight suffix from `[blocks.<name>.weights]` and `[layers.common_weights]`.

---

## Builder: `full_attention_gated`

Multi-head softmax attention with joint Q+gate projection, per-head QK-norm, multi-section RoPE, GQA, and a sigmoid output gate.

### Params Used

| Param | Description |
|---|---|
| `head_dim` | Key/query/value dimension per head |
| `n_heads` | Number of query heads |
| `n_kv_heads` | Number of key/value heads (≤ n_heads, GQA) |
| `rms_eps` | Epsilon for RMS normalization |
| `rope_n_rot` | Number of rotary dimensions |
| `rope_sections` | int[4] — multi-section RoPE section sizes |
| `rope_freq_base` | RoPE frequency base |
| `rope_mode` | Always "neox" (NeoX-style rotation ordering) |

### Required Weights

| Key | GGUF shape | Description |
|---|---|---|
| `attn_q` | `[n_embd, n_heads × head_dim × 2]` | Joint Q+gate projection; output interleaves Q and gate at stride 2×head_dim per head |
| `attn_k` | `[n_embd, n_kv_heads × head_dim]` | Key projection |
| `attn_v` | `[n_embd, n_kv_heads × head_dim]` | Value projection |
| `attn_output` | `[n_heads × head_dim, n_embd]` | Output projection |
| `attn_q_norm` | broadcast-compatible with Q | Per-head Q scale after RMS norm (optional — skip mul if absent) |
| `attn_k_norm` | broadcast-compatible with K | Per-head K scale after RMS norm (optional — skip mul if absent) |

Shape notation uses ggml mul_mat conventions: `mul_mat(W, x)` contracts `W.ne[0]` with `x.ne[0]`, producing `[W.ne[1], x.ne[1]]`.

### Config Keys

| Key | Type | Allowed values | Description |
|---|---|---|---|
| `q_has_gate` | bool | `true` | Q projection is 2× wide, second half is gate |
| `qk_norm` | string | `"rms"` | Normalize Q and K with RMS norm before RoPE |
| `rope` | string | `"multi"` | Multi-section rotary positional encoding |
| `output_gate` | string | `"sigmoid"` | Multiply merged heads by sigmoid(gate) before output projection |

### Cache Tensors

| Key | Dims | Dtype | Description |
|---|---|---|---|
| `k` | `[head_dim, max_seq_len, n_kv_heads]` | f32 | K cache — each head stored as a `[head_dim, max_seq_len]` slice |
| `v` | `[head_dim, max_seq_len, n_kv_heads]` | f32 | V cache — same layout |

### Forward Pass

**Step 1 — Joint Q+gate projection:**

```
qg = attn_q @ cur            # shape [n_heads × head_dim × 2, n_tokens]
```

Q and gate are interleaved per head: for head h, Q occupies bytes `[2h×head_dim×4, (2h+1)×head_dim×4)`, gate occupies `[(2h+1)×head_dim×4, (2h+2)×head_dim×4)`. Extract as **non-contiguous views** (nb[1]=2×head_dim×sizeof(f32), skipping the adjacent block):

```
q    = view(qg, ne=[head_dim, n_heads, n_tokens],
                nb1=2×head_dim×4, nb2=2×head_dim×n_heads×4, offset=0)
gate = view(qg, ne=[head_dim, n_heads, n_tokens],
                nb1=2×head_dim×4, nb2=2×head_dim×n_heads×4, offset=head_dim×4)
```

**Critical:** gate is non-contiguous and must be made contiguous (copy to dense buffer) before any element-wise operation. Flatten to `[n_heads × head_dim, n_tokens]`:

```
gate = cont2D(gate)            # [n_heads × head_dim, n_tokens] — dense copy required
```

Q can remain non-contiguous through RmsNorm (which only reads ne[0]=head_dim slices).

**Step 2 — QK projections and RMS-norm:**

```
k = reshape(attn_k @ cur, [head_dim, n_kv_heads, n_tokens])
v = reshape(attn_v @ cur, [head_dim, n_kv_heads, n_tokens])

q = rms_norm(q)                 # normalize over head_dim
if attn_q_norm present: q = q * attn_q_norm
k = rms_norm(k)
if attn_k_norm present: k = k * attn_k_norm
```

**Step 3 — Multi-section RoPE (NeoX mode):**

Apply multi-section RoPE to Q and K using `rope_n_rot` rotary dims, `rope_sections[4]` section sizes, `rope_freq_base`, and absolute position indices (starting at `seq_pos` in cached mode).

**Step 4 — Permute and scaled dot-product attention:**

```
q = permute(q, [0,2,1,3])    # [head_dim, n_tokens,    n_heads]
k = permute(k, [0,2,1,3])    # [head_dim, n_kv,        n_kv_heads]
v = permute(v, [0,2,1,3])    # [head_dim, n_kv,        n_kv_heads]

kq = k^T @ q                  # [n_kv, n_tokens, n_heads], GQA broadcasts n_kv_heads → n_heads
kq = softmax(kq * (1/√head_dim) + mask)   # mask: 0 for allowed, -inf for masked positions
```

The attention mask is `[n_kv, n_tokens]` (causal): position `(qi, kj)` is 0 if `kj ≤ seq_pos + qi`, else `-inf`.

**Step 5 — Weighted values:**

```
kqv = cont(v^T) @ kq          # [head_dim, n_heads, n_tokens]
cur = cont2D(permute(kqv, [0,2,1,3]))  # [n_heads × head_dim, n_tokens]
```

**Step 6 — Gate and output projection:**

```
cur = cur * sigmoid(gate)
cur = attn_output @ cur        # [n_embd, n_tokens]
```

### Cached Mode

**Prefill** (`seq_pos = 0`, `n_new` prompt tokens): Compute K, V → use inline (not cache) for attention to avoid write-before-read race → writeback to cache. Mask: `[n_new, n_new]` (causal).

**Decode** (`seq_pos > 0`, 1 token): Compute K, V → writeback at `seq_pos` → attend over full cache `[0:seq_pos+1]`. Mask: `[seq_pos+1, 1]`.

**Writeback**: per-head strided copy. Heads stored at stride `max_seq_len × head_dim × sizeof(f32)`. Each copy: `head_dim × n_new × sizeof(f32)` bytes at offset `seq_pos × head_dim × sizeof(f32)` within the head's slice.

---

## Builder: `gated_delta_net`

Gated delta-net SSM layer with 1D convolution preprocessing, L2-normalized QK, a fused gated delta-net recurrence, and gated RMS-norm output.

### Params Used

| Param | Description |
|---|---|
| `conv_channels` | `ssm_d_inner + 2 × ssm_n_group × ssm_d_state` |
| `ssm_dt_rank` | Number of SSM heads (time-step rank) |
| `ssm_d_state` | State dimension per group |
| `ssm_n_group` | Number of Q/K groups (may differ from ssm_dt_rank) |
| `head_v_dim` | `ssm_d_inner / ssm_dt_rank` — value dimension per head |
| `ssm_d_conv` | Conv kernel width (including causal padding) |
| `ssm_d_inner` | SSM inner dimension |
| `n_embd` | Embedding dimension |
| `rms_eps` | Epsilon for RMS normalization |

### Required Weights

| Key | GGUF shape | Description |
|---|---|---|
| `attn_qkv` | `[n_embd, conv_channels]` | Joint QKV projection (pre-conv) |
| `attn_gate` | `[n_embd, ssm_d_inner]` | Gating projection |
| `ssm_a` | `[ssm_dt_rank]` | State decay scalar per SSM head |
| `ssm_alpha` | `[n_embd, ssm_dt_rank]` | Time-step input projection |
| `ssm_beta` | `[n_embd, ssm_dt_rank]` | Beta gating projection |
| `ssm_conv1d` | `[ssm_d_conv, 1, conv_channels]` | Depthwise 1D conv kernel |
| `ssm_dt_bias` | `[ssm_dt_rank]` | Bias added to alpha before softplus |
| `ssm_norm` | broadcast-compatible with SSM output | Per-channel scale after RMS norm (optional) |
| `ssm_out` | `[ssm_d_inner, n_embd]` | Output projection |

### Config Keys

| Key | Type | Allowed values | Description |
|---|---|---|---|
| `conv_activation` | string | `"silu"` | Activation applied after conv1d |
| `qk_norm` | string | `"l2"` | Normalize Q and K with L2 norm |
| `gate_norm` | string | `"rms"` | Normalize SSM output with RMS norm before gating |
| `gate_activation` | string | `"silu"` | Activation applied to gate z before element-wise multiply |

### Cache Tensors

| Key | Dims | Dtype | Description |
|---|---|---|---|
| `conv_state` | `[ssm_d_conv - 1, conv_channels]` | f32 | Last (d_conv-1) input positions — ring buffer for causal conv |
| `ssm_state` | `[head_v_dim, head_v_dim, ssm_dt_rank]` | f32 | Linear recurrence state matrix, one per SSM head |

### Forward Pass

**Step 1 — Projections:**

```
qkv_mixed = reshape(attn_qkv @ cur, [conv_channels, n_tokens, 1])
z         = attn_gate @ cur                                          # [ssm_d_inner, n_tokens]
beta      = reshape(sigmoid(ssm_beta @ cur), [1, ssm_dt_rank, n_tokens, 1])
alpha     = reshape(ssm_alpha @ cur, [ssm_dt_rank, n_tokens, 1])
alpha     = softplus(alpha + ssm_dt_bias)
g         = reshape(alpha * ssm_a, [1, ssm_dt_rank, n_tokens, 1])
```

**Step 2 — Causal 1D convolution:**

Concatenate the conv state (last d_conv-1 positions) with current input along time, then convolve:

```
conv_input  = concat(conv_state, qkv_mixed^T, dim=0)  # [d_conv-1+n_tokens, conv_channels, 1]
conv_out    = silu(ssm_conv1d(conv_input))              # [conv_channels, n_tokens, 1] — channel-first output
new_conv_state = conv_input[-d_conv+1:, :, :]           # last (d_conv-1) positions for cache writeback
```

**Hazard:** `ggml_ssm_conv` input is `[time, channels, seqs]` but output is `[channels, n_tokens, seqs]` — time and channel axes transpose. Account for this when splitting Q/K/V.

**Step 3 — Split Q, K, V from conv output:**

`qkvDim = ssm_d_state × ssm_n_group × 2 + head_v_dim × ssm_dt_rank` (== `conv_channels`). `conv_out` is channel-first (`[qkvDim, n_tokens, 1]`), so Q/K/V are contiguous sub-ranges of ne[0]. Extract as 4D views with nb2=qkvDim×sizeof(f32):

```
# row_bytes(n) = n × sizeof(f32)
# All views share nb2 = row_bytes(qkvDim), nb3 = row_bytes(qkvDim × n_tokens)

q_ssm: ne=[d_state, n_group, n_tokens, 1], nb1=row_bytes(d_state), nb2=row_bytes(qkvDim), offset=0
k_ssm: ne=[d_state, n_group, n_tokens, 1], nb1=row_bytes(d_state), nb2=row_bytes(qkvDim), offset=row_bytes(d_state × n_group)
v_ssm: ne=[head_v_dim, dt_rank, n_tokens, 1], nb1=row_bytes(head_v_dim), nb2=row_bytes(qkvDim), offset=row_bytes(2 × d_state × n_group)
```

Key invariant: nb2 = row_bytes(qkvDim) for all three views (stride between tokens spans the full channel row).

**Step 4 — L2-normalize Q, K:**

```
q_ssm = l2_norm(q_ssm)    # normalize over d_state per (group, token, seq)
k_ssm = l2_norm(k_ssm)
```

**Step 5 — Group expansion (if ssm_n_group ≠ ssm_dt_rank):**

Repeat Q, K along the group dim → `[d_state, ssm_dt_rank, n_tokens, 1]`.

**Step 6 — Gated delta-net recurrence (fused op):**

Inputs:
- `q_ssm`: `[d_state, ssm_dt_rank, n_tokens, 1]`
- `k_ssm`: `[d_state, ssm_dt_rank, n_tokens, 1]`
- `v_ssm`: `[head_v_dim, ssm_dt_rank, n_tokens, 1]`
- `g`:     `[1, ssm_dt_rank, n_tokens, 1]` — decay gate
- `beta`:  `[1, ssm_dt_rank, n_tokens, 1]` — update gate
- `state`: `[head_v_dim, head_v_dim, ssm_dt_rank, 1]` — initial recurrence state (zero or cached)

The recurrence per SSM head h, per token t:
```
S_h[t] = (1 - beta_h[t] × k_h[t] × k_h[t]^T) × g_h[t] × S_h[t-1]  +  beta_h[t] × v_h[t] × k_h[t]^T
y_h[t] = q_h[t]^T × S_h[t]
```
(approximate: exact delta-net recurrence, see Yang et al. 2024)

The fused op returns a flat buffer containing two tensors at different offsets with **different shapes** (ne[1] differs) — two independent views, not slices along a common axis:

```
# output_bytes = head_v_dim × ssm_dt_rank × n_tokens × sizeof(f32)

ssm_output: ne=[head_v_dim, dt_rank, n_tokens, 1],
            nb1=row_bytes(head_v_dim), nb2=row_bytes(head_v_dim × dt_rank),
            nb3=row_bytes(head_v_dim × dt_rank × n_tokens),
            offset=0

new_state:  ne=[head_v_dim, head_v_dim, dt_rank, 1],
            nb1=row_bytes(head_v_dim), nb2=row_bytes(head_v_dim²),
            nb3=row_bytes(head_v_dim² × dt_rank),
            offset=output_bytes       ← starts immediately after the token outputs
```

**Step 7 — Gated RMS-norm output:**

```
z4d   = reshape(z, [head_v_dim, ssm_dt_rank, n_tokens, 1])
normed = rms_norm(ssm_output)                   # normalize over head_v_dim
if ssm_norm present: normed = normed * ssm_norm
cur   = normed * silu(z4d)
```

**Step 8 — Output projection:**

```
cur = reshape(cur, [ssm_d_inner, n_tokens])
cur = ssm_out @ cur                              # [n_embd, n_tokens]
```

### Cached Mode

- **Conv state**: prepend `conv_state` `[d_conv-1, conv_channels]` to QKV input; after compute, write back last (d_conv-1) positions.
- **SSM state**: pass cached `ssm_state` as initial recurrence state (instead of zeros); after compute, write back `final_state` from `delta_out`.

Both are flat copies (full tensor replaced, no head-striding).

---

## Builder: `swiglu`

Standard SwiGLU FFN. No params or config keys — dimensions inferred from weight shapes.

**Weights:** `gate` `[n_embd, n_ff]`, `up` `[n_embd, n_ff]`, `down` `[n_ff, n_embd]`

```
gate_out = silu(gate @ input)             # [n_ff, n_tokens]
up_out   = up @ input                     # [n_ff, n_tokens]
output   = down @ (gate_out * up_out)     # [n_embd, n_tokens]
```

---

## Builder: `moe`

Mixture-of-Experts FFN with optional shared expert and optional expert selection bias.

**Required weights:** `gate_inp`, `gate_exps`, `up_exps`, `down_exps`
**Optional weights:** `gate_inp_shexp`, `gate_shexp`, `up_shexp`, `down_shexp`, `exp_probs_b`
**Required params:** `n_expert`, `n_expert_used`

Expert routing: softmax → optional bias (`exp_probs_b` for selection only) → top-k → mul_mat_id → SwiGLU → aggregate. Weight normalization (sum_rows → clamp → div). Optional shared expert added with optional sigmoid gate.

---

## Builder: `attention`

Standard multi-head attention with GQA and standard RoPE. No gating on output. Used by Llama and standard transformer architectures.

**Required weights:** `attn_q`, `attn_k`, `attn_v`, `attn_output`
**Required params:** `head_dim`, `n_heads`, `n_kv_heads`, `rope_n_rot`, `rope_freq_base`
**Cache:** `k` and `v` tensors, same as `full_attention_gated`

---

## Builder: `mla_attention`

Multi-head Latent Attention (DeepSeek V2 / GLM-4). Low-rank Q and KV compression with Q-nope absorption and post-attention V decompression.

**Required weights:** `attn_q_a`, `attn_q_a_norm`, `attn_q_b`, `attn_kv_a_mqa`, `attn_kv_a_norm`, `attn_k_b`, `attn_v_b`, `attn_output`
**Required params:** `n_heads`, `rms_eps`, `rope_n_rot`, `rope_freq_base`, `kv_lora_rank`, `head_k_dim_mla`
**Cache:** K only (compressed + RoPE dims, 1 MQA head). V derived from K's compressed portion.

Q path: compress → norm → expand → split (nope + pe) → RoPE on pe → absorb nope into KV latent space via k_b.
KV path: compress → split (compressed + pe) → norm → RoPE on pe → concat.
Post-attention: V decompression via batched per-head matmul with v_b.

---

## Layer Routing (`[layers.routing]`)

Determines which `[blocks.*]` definition each layer uses. Three mutually exclusive modes:

### Uniform

All layers use a single block type (e.g. Llama, DeepSeek2 MLA):

```toml
[layers.routing]
uniform = "attention"        # must name a [blocks.*] key
```

### Rule

Expression evaluated per-layer; nonzero → `if_true`, zero → `if_false`:

```toml
[layers.routing]
rule     = "(@{layer_idx} + 1) % ${full_attn_interval} != 0"
if_true  = "recurrent_ssm"
if_false = "full_attention"
```

### Pattern

Integer array param indexed by layer; nonzero → `if_true`, zero → `if_false`. Used when routing is driven by an opaque per-layer array in GGUF metadata (e.g. Gemma4 ISWA):

```toml
[layers.routing]
pattern  = "swa_pattern"     # param name resolving to int[] from GGUF
if_true  = "swa_attention"
if_false = "full_attention"
```

**Validation:** Exactly one of `uniform`, `rule`, or `pattern` must be set. `uniform` is mutually exclusive with `if_true`/`if_false`. `rule` and `pattern` both require `if_true` and `if_false`. All block references must name a `[blocks.*]` key.

---

## Per-Layer FFN Routing (`[ffn_alt]`)

Optional `[ffn_alt]` section defines an alternative FFN builder for layers that have different weights in the GGUF. The model loader auto-detects which layers use the alt FFN based on whether the alt weights exist.

Example: DeepSeek2 uses dense SwiGLU for the first layer and MoE for the rest.

---

## Example (`[example]`)

Diagram-only values for `gen-arch-diagram` when no GGUF is loaded. No effect on inference.

| Key | Description |
|---|---|
| `n_layers` | Layer count for the layer-pattern strip |
| `full_attn_every` | Rule-based routing: interval N where `(i+1) % N == 0` selects `if_false`. Used by Qwen3.5. |
| `attn_pattern_true_every` | Pattern-based routing: `pattern[i] = 1` (`if_true`) when `(i+1) % N == 0`, else 0. |
| `attn_pattern_false_every` | Pattern-based routing: `pattern[i] = 0` (`if_false`) when `(i+1) % N == 0`, else 1. Used by Gemma4 ISWA. |

The three interval keys are **mutually exclusive** (validation error if more than one is set). Omit all three for uniform architectures (Llama, DeepSeek2).

```toml
# Rule-based (Qwen3.5): full attention every 4th layer
[example]
n_layers        = 32
full_attn_every = 4

# Pattern-based (Gemma4): SWA (if_false) every 5th layer
[example]
n_layers                 = 30
attn_pattern_false_every = 5
```

---

## Tokens (`[tokens]`)

Thinking control tokens and extra stop tokens. Auto-detected from vocab as fallback if not specified.

```toml
[tokens]
think_open   = "<think>"
think_close  = "</think>"
no_think     = "/nothink"
extra_eos    = ["<end_of_turn>"]    # non-standard EOS (GGUF EOS always included automatically)
```

---

## Expression Language

Used in `[params.derived]`, `[blocks.*.cache].dims`, `[layers].count`, and `[layers.routing].rule`.

**Syntax:**
- Integer literals: `128`, `4096`
- `${name}` — resolved param dereference (integers only; used in routing rules)
- Bare param names — allowed in derived expressions and cache dims
- `@{name}` — engine builtin (currently: `@{layer_idx}`, 0-based)
- Arithmetic: `+`, `-`, `*`, `/`, `%`; comparison (routing only): `==`, `!=`; parentheses
- Optional GGUF params: suffix key with `?` to silently skip (e.g., `"arch.key?"`)
- `tensor_name.ne[dim]` — reads GGUF tensor dimension (e.g., `token_embd.ne[1]` → `n_vocab`)

**Param defaults (`[params.defaults]`):** If a GGUF param resolves to 0, the defaults table provides a fallback. Applied after all params (including derived) are resolved.

```toml
[params.defaults]
n_kv_heads = "n_heads"    # GGUF convention: 0 means same as n_heads (no GQA)
```

**Array-to-scalar promotion:** Per-layer array params (e.g., `head_count_kv`) also store their first element as a scalar int for use in cache dims and derived expressions.

**Examples:**
```
conv_channels = "ssm_d_inner + 2 * ssm_n_group * ssm_d_state"
head_v_dim    = "ssm_d_inner / ssm_dt_rank"
n_vocab       = "token_embd.ne[1]"
rule          = "(@{layer_idx} + 1) % ${full_attn_interval} != 0"
```
