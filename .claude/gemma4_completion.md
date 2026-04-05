# Gemma 4 Architecture Support — Completion Handoff

## Objective

Get `gemma-4-E4B-it-Q4_K_M.gguf` (arch `gemma4`) to pass the llama.cpp equivalence test (`bash test_llama_equiv.sh`). The test compares our stateless forward pass logprobs against llama.cpp's output on the same prompt.

**Current logprob diff: -0.97** (threshold ~0.01). All other models (Llama 3.2, Qwen 3.5) pass.

## Implementation Status

Phases 1-6 of the plan (`/Users/benn/.claude/plans/snuggly-skipping-grove.md`) are complete. Phase 7 (runtime correctness) is blocked on a numerical divergence. Phase 8 (AGENTS.md) is complete.

All code compiles, all existing model tests pass, Gemma 4 runs without crashing and produces semi-coherent text — but logprobs don't match the llama.cpp reference.

## Architecture Overview

Gemma 4 E4B is a 42-layer dense transformer with these novel features (all implemented):

- **Two attention types**: full attention (head_dim=512, freq_base=1M, proportional RoPE via `rope_freqs`) and SWA (head_dim=256, freq_base=10K, sliding_window=512)
- **Layer routing**: bool array `swa_pattern` = `SSSSSF` repeating (35 SWA, 7 full at indices 5,11,17,23,29,35,41)
- **Shared KV cache**: 18 non-KV layers (24-41) share cache with KV layers (0-23). Determined by `n_kv_shared_layers=18` param, NOT by weight presence (all layers have K/V weights in GGUF)
- **kq_scale = 1.0** (no 1/sqrt(headDim) scaling — `f_attention_scale = 1.0` in llama.cpp)
- **RoPE NeoX mode** (mode=2) for all layers
- **GeGLU FFN** (GELU activation), **V-norm** (raw RMS), **post-attention/post-FFN norms**
- **Per-layer token embeddings** with GELU gating, projection, normalization
- **Embedding scaling** by sqrt(n_embd=2560), **logit softcapping** (cap=30)
- **Layer output scaling** (per-layer scalar)

## Key Files Modified

| File | What |
|---|---|
| `models/arch/gemma4.arch.toml` | Architecture definition |
| `src/internal/inference/arch/block_attention_std.go` | Config-based param overrides, optional K/V, SharedKV, V-norm, SWA mask, rope_freqs, NeoX mode, kq_scale |
| `src/internal/inference/arch/block_attention_util.go` | `selectSharedKV` for non-KV layers, `scaledDotProductAttention` takes kqScale float32 |
| `src/internal/inference/arch/graph.go` | Post-norms, embed scaling, softcapping, per-layer embeddings, SWA mask, SharedKV init, layer output scale |
| `src/internal/inference/arch/model.go` | Non-KV layer weight nulling (K/V set to NilTensor for layers >= nKVFromStart) |
| `src/internal/inference/arch/cache.go` | Param-driven shared cache allocation via `lastKVByBlock` |
| `src/internal/inference/arch/arch.go` | EmbedScale, CacheDef.Shared, RoutingDef.Pattern |
| `src/internal/inference/arch/arch_util.go` | `configIntOr`, `configFloatOr` helpers |
| `src/internal/inference/arch/params.go` | `GetArrBools` support |
| `src/internal/inference/arch/weights.go` | Pattern-based routing |
| `src/internal/inference/arch/blocks.go` | SharedKVState, geglu registration |
| `src/internal/inference/arch/block_ffn_geglu.go` | GeGLU FFN builder |
| `src/ggml_lib/src/ggml_ops.{h,c}` | Gelu, Tanh ops |
| `src/internal/inference/ggml/ops_graph.go` | Gelu, Tanh Go wrappers |

## GGUF Metadata (verified)

```
n_layers=42, n_embd=2560, n_heads=8, n_kv_heads=2
head_dim=512, head_dim_swa=256
rope_n_rot=512, rope_n_rot_swa=256
rope_freq_base=1000000, rope_freq_base_swa=10000
sliding_window=512, n_kv_shared_layers=18
n_embd_per_layer=256, logit_softcapping=30.0, n_ff=10240
rope_freqs.weight: global tensor, shape [256] (NOT per-layer)
```

## Verified Correct

- **Cache sharing**: Non-KV SWA layers → layer 22 cache, non-KV full layers → layer 23 cache. Matches llama.cpp's `layer_reuse_cb` mapping.
- **Layer routing**: swa_pattern array resolves correctly via pattern-based routing.
- **RopeNeoX constant**: `GGML_GO_ROPE_NEOX = 2` matches ggml's `GGML_ROPE_TYPE_NEOX`.
- **SWA mask construction**: Correct for short sequences (off-by-one vs llama.cpp at window boundary, but irrelevant for test prompt < 512 tokens).
- **Layer loop ordering**: Matches llama.cpp: attn_norm → attn → post_attn_norm → residual → ffn_norm → ffn → post_ffn_norm → residual → per-layer embed → layer_output_scale.
- **Per-layer embedding setup code**: Reviewed against llama.cpp `get_per_layer_inputs()` and `project_per_layer_inputs()` — appears to match.
- **Logit softcapping**: `scale(1/30) → tanh → scale(30)` matches llama.cpp.
- **Embedding scaling**: `scale(sqrt(2560))` matches llama.cpp.

## Historical Test Results

| Configuration | Logprob Diff | Notes |
|---|---|---|
| Standard rope (mode=0), all fixes | -0.57 | Semi-coherent output, echoes question |
| NeoX rope (mode=2), all fixes | -0.97 | Current state. NeoX is correct per llama.cpp |
| No per-layer embeds + standard rope | -1.17 | Per-layer embeds are helping |
| Before kq_scale/SharedKV fixes | Gibberish | Completely broken |

## Investigations Required

### 1. ~~Verify `rope_freqs` is actually loading~~ VERIFIED OK

**Tested 2026-04-04**: Added debug logging to `BuildCached`. Results confirm:
- Full attention layers (headDim=512): `rope_freqs.IsNil=false` — loading correctly via global fallback
- SWA layers (headDim=256): `rope_freqs.IsNil=true` — correctly absent
- All other params match expected values (see "Verified Correct" section above)

This was the #1 suspect but is **not the bug**.

### 2. Compare intermediate tensor values (HIGH PRIORITY)

Add debug instrumentation to dump intermediate tensor values at key points for layer 0, then compare against llama.cpp's debug output. Key checkpoints:

1. After embedding lookup + scaling (before layer loop)
2. After attn_norm (layer 0 input to attention)
3. After Q projection + reshape
4. After Q-norm
5. After RoPE on Q
6. After attention output
7. After post-attention norm + residual
8. After FFN output + residual
9. After per-layer embedding injection
10. After layer output scaling

Compare first few float values at each checkpoint. The first checkpoint where values diverge identifies the buggy component.

**Reference**: llama.cpp debug output can be obtained by building with `LLAMA_DEBUG=1` or using the callback mechanism. Alternatively, add `printf` statements to `/Users/benn/projects/llama.cpp/src/models/gemma4-iswa.cpp` at the `cb()` callsites.

### 3. Verify parameter resolution per block type (MEDIUM PRIORITY)

For full_attention blocks, verify at runtime that `attnParams()` resolves:
- `headDim = 512`
- `nHeads = 8`
- `nKVHeads = 2`
- `nRot = 512`
- `freqBase = 1000000.0`
- `kqScale = 1.0`
- `ropeMode = 2` (NeoX)

For swa_attention blocks, verify:
- `headDim = 256` (from config override `head_dim = "head_dim_swa"`)
- `nHeads = 8` (no override, uses global n_heads)
- `nKVHeads = 2` (from config override `n_kv_heads = "n_kv_heads"`)
- `nRot = 256` (from config override `rope_n_rot = "rope_n_rot_swa"`)
- `freqBase = 10000.0` (from config override `rope_freq_base = "rope_freq_base_swa"`)
- `kqScale = 1.0`
- `ropeMode = 2`

### 4. Investigate NeoX mode worsening results (MEDIUM PRIORITY)

Standard rope gives -0.57, NeoX gives -0.97. NeoX is confirmed correct per llama.cpp reference. The worsening suggests either:
- Another bug that interacts badly with NeoX (e.g., freq_factors not loading, so wrong frequencies + wrong pairing = worse than wrong frequencies + right pairing)
- Our `ggml_rope_ext` binding has a subtle issue with NeoX + freq_factors combination

If investigation #1 confirms freq_factors ARE loading, focus on whether `ggml_rope_ext` with mode=2 and non-null freq_factors produces the same output as llama.cpp for the same input tensor + positions + factors.

### 5. Check `n_heads` per-layer behavior (LOW PRIORITY)

In the GGUF, `gemma4.attention.head_count = 8` is a scalar. Both SWA and full attention use 8 heads, but with different head dims (512 vs 256). This means:
- Full attention: 8 heads * 512 = 4096 dim Q projection
- SWA: 8 heads * 256 = 2048 dim Q projection

Verify that Q/K/V projection weight shapes match these expectations. If `n_heads` is wrong for either block type, the reshape would produce garbage dimensions.

### 6. SWA mask off-by-one (LOW PRIORITY, fix regardless)

Our code: `kj < qi - swaWindow` (masks when distance > window)
llama.cpp: `p1 - p0 >= n_swa` (masks when distance >= window)

Off by one position. Only matters for sequences > 512 tokens, so won't affect the test. Fix after the main bug is resolved.

## llama.cpp Reference Code

The authoritative Gemma 4 implementation is at:
- `/Users/benn/projects/llama.cpp/src/models/gemma4-iswa.cpp` — full forward pass
- `/Users/benn/projects/llama.cpp/src/llama-graph.cpp` — ISWA `build_attn` (line ~2224)
- `/Users/benn/projects/llama.cpp/src/llama-model.cpp` — model init, `f_attention_scale=1.0`, `layer_reuse_cb`

## How to Run Tests

```bash
make                                    # build
bash test_llama_equiv.sh                # equivalence test (all models)
ALL_MODELS=true bash test_inference.sh "What is 2+2?"  # inference smoke test
make arch-diagrams                      # SVG diagrams (should pass)
```

## Suggested Investigation Order

1. ~~Verify `rope_freqs` loading and `attnParams` resolution~~ **DONE — all correct**
2. Add tensor value dumps at layer 0 checkpoints, compare with llama.cpp debug output
3. Identify first divergence point, fix, iterate
