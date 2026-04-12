# Inference Lab Bench

From-scratch Go LLM inference engine for R&D into inference mechanics. Multi-model API server, data-driven architecture definition via TOML DSL, KV-cached and stateless inference, weight culling infrastructure, and visualization tooling.

## Project Priorities

- **R&D focus** — clarity and tinkerability over production concerns; don't make choices that close doors
- **Consistency** — naming, data patterns, conventions. Surprises are expensive in R&D. TOML is the static data language except where constrained (e.g. JSON for OpenAPI payloads). Use discipline-specific terminology where high-consensus terms exist.
- **Data-driven architecture purity** — model-specific Go code subverts the design. The TOML DSL + block builder system is the whole point; adding architectures should be a data-writing operation, not a coding one.
- **DRY** — any functionality used in ≥2 places gets refactored into a utility:
  - project-wide → `src/internal/util/project_util.go`
  - package-internal, multi-file → `src/internal/[pkg]/[pkg]_util.go`
  - single-file → define in that file
- **Separation of Concerns**
  - systems should maintain ignorance of other system internals
  - one function or system 'pre-digesting' data for consumption by another function or system in a way only useful to the callee is a sign of violation of this principle. 
- **Complexity budget** — complex code for complex needs only. O(n) over 64 elements beats a 100-line O(1) every time.
- **Portability** — macOS only now; Linux/Windows ports coming. Don't accumulate portability debt. Platform-specific surface area is limited to: ggml cmake build options, `ggml_lib` compiler flags, and Metal-specific binding wiring.

## Current Status

Known working models (links in README.md):
- Llama-3.2-3B-Instruct-f16.gguf, llama-3.2-3b-instruct-q4_k_m.gguf (`llama` arch)
- qwen35-9b-opus46-mix-i1-Q4_K_M.gguf (`qwen35` arch)
- DeepSeek-V2-Lite-Chat.Q4_K_M.gguf (`deepseek2` arch)
- gemma-4-E4B-it-Q4_K_M.gguf (dense, `gemma4` arch)
- Gemma 4 MoE (`gemma4` arch, auto-detected via `[ffn_alt]` GGUF weights)
- LLaDA-MoE-7B-A1B-Instruct (`llada-moe` arch) — MoE diffusion model; working
- LLaDA-8B-Instruct (`llada` arch) — dense diffusion model; built, not yet tested against a live model

- KV-cached and stateless inference; TOML DSL drives GGUF loading, graph construction, and cache allocation. Only C code is a thin model-agnostic ggml op wrapper. Zero C++.
- HTTP server: Bearer auth, model listing, streaming SSE + non-streaming completions, logprobs, timing/throughput in `usage` response.
- **Zero model-specific code**: chat templates executed from GGUF `tokenizer.chat_template` via gonja; BOS/EOS from GGUF metadata. Thinking mode controlled entirely by template `enable_thinking` variable — no post-render prompt manipulation.
- BPE tokenizer: dual mode — GPT-2 byte-level (Llama, Qwen) + SentencePiece (Gemma); greedy + top-p sampling; logprob computation via stable log-softmax
- KV cache + SSM state: prefill in one pass, decode one token at a time
- Stateless mode (`"stateless": true`) — only mode supporting `ForwardCaptures`. Data collection and KV-cache optimization are intentionally separated; do not add capture to `ForwardCached`.
- SVG architecture visualizer: `bench gen-arch-diagram` generates `*.arch.svg` and `*.layers.svg` from TOML

## Architecture

- **Language**: Go + pure C via CGo (ggml op wrappers only)
- **Model definition**: TOML DSL (`models/arch/*.arch.toml`)
- **Inference backend**: ggml (git submodule), gpu-accelerated
- **Model format**: GGUF (`models/`)
- **API**: OpenAI-compatible (`/api/v1/models`, `/api/v1/chat/completions`) with extensions: `"stateless"`, `"cull_method"`, `"enable_thinking"`, `"elide_thinking"`, `"logprobs"`, `"top_logprobs"`, `"model":"default"`, `"diffusion"` (nested object: `steps`, `block_length`; ignored with a warning on non-diffusion models). Legacy `/v1/*` also supported. Diagnostic browser at `/diag/`. Control endpoint at `/ctl/` (`?memstats` = memory stats; `?quit` = graceful shutdown; `?quit&now` = immediate). Request-level overrides of server config defaults logged as `[req] param overrides: ...`.
- **Diagnostics**: `/diag/` serves files from `bin/diag/` — a useful location for dumping R&D diagnostic output (SVGs, culling maps, etc.) that can be viewed in-browser while the server is running.
- **Config**: `config/api_config.toml` — `[server]` (host, port, auth_token), `[models]` (directory, default), `[inference]` (max_seq_len, enable_thinking_default, elide_thinking_default, log_thinking, cull_method_default, single_resident_model, max_request_seq_len, strict_mode)
- **Default listen**: `0.0.0.0:11116`

## Source Layout

```
Makefile                        Top-level: test, integration-test, equiv-test, symlinks bin/ delegates the rest to src/
config/
  api_config.toml               API server config (models.default, auth, listen, max_seq_len)
  chat_config.toml              Chat client config (base_url=auto, model, system_prompt)
models/
  *.gguf                        Model files (not committed)
  arch/                         Architecture TOML definitions + generated SVGs
    MODEL_ARCH_TOML_DSL_SPEC.md DSL specification
    block_svg/                  Hand-crafted SVG fragments per block builder
src/
  Makefile                      Go build, ggml build, test
  go.mod / go.sum               Module: inference-lab-bench
  bench/                        CLI entry point (cobra subcommands — thin wrappers)
    main.go                     Root command
    serve_api.go                serve-api
    chat.go                     chat
    gen_arch_diagram.go         gen-arch-diagram: SVG from TOML
    gen_cull_metadata.go        gen-cull-metadata: writes .cullmeta sidecar
  internal/
    log/                        Structured leveled logger (Debug/Info/Warn/Error/Fatal); see ### Internal Logging
    util/                       Project-wide utilities (LoadTOML, WriteJSON, BenchPaths, extension constants)
    apiserver/                  HTTP handlers (OpenAI-compatible: /api/v1/*), /ctl endpoint
    chatclient/                 Interactive chat client
    model/                      GGUF scanning + metadata
    inference/
      engine.go                 Engine, GenerateParams, DiffusionParams, IsDiffusion(), Generate() dispatch
      generate_cached.go        generateCached — KV/SSM-cached autoregressive loop
      generate_stateless.go     generateStateless — full-sequence recompute autoregressive loop
      generate_diffusion.go     generateDiffusion — block-based iterative masked denoising
      metrics.go                InferenceMetrics (timing, throughput, cull ratio, FinishReason, optional Diagnostic)
      diagnostic.go             Post-generation diagnostic runners (additional forward passes, analysis)
      sampler.go                Greedy and top-p sampling, ComputeTopLogProbs (stable log-softmax)
      tokenizer.go              BPE tokenizer (GPT-2 byte-level + SentencePiece dual mode); chat template from GGUF via gonja; readGGUFTokensRaw direct GGUF binary reader
      arch/
        arch.go                 ArchDef, TOML parser, Validate() with builder contracts
        arch_util.go            Shared tensor-op helpers, cache key constants, configIntOr/configFloatOr/configStr, attentionScale
        block_attention_util.go Shared attention helpers (scaledDotProductAttention, RoPE, KV cache)
        params.go               Param resolver + routing expression eval
        weights.go              Weight resolver (template expansion, layer routing)
        blocks.go               BlockBuilder/FFNBuilder interfaces, ForwardCaptures, SharedKVState, GraphInputs, LayerCache, registry
        engagement.go           EngagementData: per-layer cosine similarity (block + FFN residual stream; populated by ForwardStateless)
        block_attention.go      full_attention_gated (Qwen3.5)
        block_attention_std.go  attention (Llama, Gemma4 SWA/global; supports sliding_window config, shared KV)
        block_attention_mla.go  mla_attention (DeepSeek2/GLM-4)
        block_ssm.go            gated_delta_net (Qwen3.5 SSM)
        block_ffn.go            swiglu
        block_ffn_geglu.go      geglu (GeGLU FFN: gelu(gate*x) * up * x → down)
        block_ffn_moe.go        moe (MoE + shared expert + expert bias)
        mask.go                 CullingMask (whole-tensor zeroing)
        modules.go              Module, ModuleMap
        module_map.go           BuildModuleMap, BuildTensorDimsMap
        weight_store.go         WeightStore: immutable GPU-resident tensor storage
        model.go                GenericModel: GGUF loading, per-layer FFN routing
        cache.go                Cache allocation (NewCache, GenericCache)
        graph.go                Forward pass (ForwardStateless, ForwardStatelessAllLogits, ForwardCached, forwardStatelessCore, runLayers)
        validate_lines.go       ResolveErrorLines (TOML key path → source line)
        diagram_util.go         Shared diagram palette (diagramPalette)
        arch_diagram.go         Architecture overview SVG renderer
        module_map_diagram.go   Module map SVG renderer (cull overlays)
      culling/
        culling.go              ApplyCulling dispatch
        culling_meta.go         CullingMeta, LoadCullingMeta (auto-detects binary formats + TOML)
        culling_util.go         WriteCullDiagnostics
        method_inattention.go   Inattention culling: (empty shell) prompt meta, block engagement
        method_random.go        Random test pattern culling
      ggml/                     Go wrappers for ggml ops (~43 functions)
  ggml_lib/                     C op wrappers + ggml build
  third_party/ggml/             ggml git submodule
test_inference.sh               Test harness (works with bench or llama-server via IS_LLAMA)
test_equiv.sh                   Logprob equivalence test: bench vs llama-server, stateless vs non-stateless, flash vs standard
```

All SVG renderers share a single palette from `diagram_util.go:diagramPalette()`. To change any color, update that map only.

## Build & Run

**Do not use `go build` directly.** Always use `make` targets — the Makefile handles ggml C compilation, cgo flags, symlinks, and output placement.

```bash
make                    # build bench binary, symlink config+models into bin/
make serve              # build + start API server (--log bin/bench.log --log-level INFO)
make chat               # build + start interactive chat client
make arch-diagrams      # rebuild SVG diagrams from models/arch/*.arch.toml
make test               # run go unit tests
make integration-test   # test inference end to end for all models

# CLI subcommands
./bin/bench serve-api
./bin/bench chat
./bin/bench gen-arch-diagram [--layers] [--blocks] <input.toml> [output.svg]
./bin/bench gen-cull-metadata --cull-method <method> [--cpu] <model.gguf>

# Test harness (assumes server running; FORCE_NEW_SERVER=true to kill+restart)
bash test_inference.sh "What is 2+2?"
bash test_inference.sh --loop
FORCE_NEW_SERVER=true bash test_inference.sh "Hi"
MODEL=Qwen3.5-4B_abliterated.Q4_K_M MAX_TOKENS=100 bash test_inference.sh "Hi"
THINK=true ELIDE_THINK=false bash test_inference.sh "Hi"
ALL_MODELS=true bash test_inference.sh "Hi"   # test every loaded model in sequence
DIFFUSION_STEPS=32 DIFFUSION_BLOCK_LENGTH=64 bash test_inference.sh "Hi"   # diffusion params (ignored on autoregressive models)

# API
curl localhost:11116/api/v1/models
curl -X POST localhost:11116/api/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","messages":[{"role":"user","content":"Hi"}]}'
```

**Validating Changes**
* `make test && make integration-test` must pass.
* When adding new model architectures or changing inference code, `make equiv-test` must also pass — it compares top-1 logprobs against `llama-server` (Homebrew) (among other checks) to verify inference correctness within GPU floating-point variance.
* Run `ALL_MODELS=true bash test_inference.sh "..."` before declaring any inference change complete. Llama is easy mode — edge cases surface on Qwen3.5, Qwen3.5-MoE, DeepSeek2, and Gemma4.

## Key Technical Details

### TOML Model DSL

`models/arch/*.arch.toml` declares:
- **Params**: GGUF metadata key mappings + derived arithmetic expressions + `[params.defaults]` fallbacks
- **Layer routing**: expression rules → block type per layer. `@{layer_idx}` = builtin; `${name}` = resolved GGUF param. Evaluated at load time.
- **Weight bindings**: GGUF tensor name templates with `blk.@{layer_idx}.` prefix expansion
- **Cache specs**: per-block-type tensor dimensions and dtypes; `shared` group name for layers that reuse a single cache tensor (e.g. non-KV layers in Gemma4)
- **FFN type**: `[ffn]` / `[ffn_alt]` for per-layer routing (e.g. dense for first N layers, MoE for rest — auto-detected from weights)
- **MoE FFN config** (`[ffn.config]` / `[ffn_alt.config]` for the `moe` builder):
  - `norm_w = "true"` — normalize router weights (static; always on)
  - `norm_w_param = "<param_name>"` — normalize router weights only when the named GGUF integer param is nonzero (dynamic; reads at load time). Mutually exclusive with `norm_w` — using both in the same `[ffn.config]` block is a validation error.
- **Layer routing**: `layers.routing.rule` (expression) OR `layers.routing.pattern` (array param name — nonzero → if_true)
- **Architecture flags**:
  - `embed_scale` — multiply input embeddings by `sqrt(n_embd)` before the layer loop
  - `non_causal` — bidirectional attention (no causal mask); required when `generation = "diffusion"`
  - `generation` — `""` (default, autoregressive) or `"diffusion"` (iterative masked denoising). Setting `generation = "diffusion"` without `non_causal = true` is a validation error.
  - `shift_logits` — diffusion only: output position `p` reads logits from position `p-1` rather than `p`. Required for models whose output tensor is offset by one vs. input.
- **Tokens**: `[tokens]` — `think_open`, `think_close`, `stop_tokens` (string array; each entry added to the generation stop set alongside EOS)

Optional GGUF params use `?` suffix (silently skipped if missing). Full spec: `models/arch/MODEL_ARCH_TOML_DSL_SPEC.md` — **read before writing or modifying any `.arch.toml`**.

### ggml Semantics (verified)
- `ggml_permute(ctx, a, ax0..ax3)`: input dim i → output position ax_i. `ne[ax_i] = a->ne[i]`.
- `ggml_mul_mat(A, B)`: contracts over `A->ne[0] == B->ne[0]`; result `[A->ne[1], B->ne[1], ...]`; GQA broadcasting when `B->ne[2] % A->ne[2] == 0`.
- `ggml_rms_norm`: normalizes over `ne[0]` per slice.
- `ggml_soft_max_ext(a, mask, scale, max_bias)`: `softmax(scale * a + mask)` over `ne[0]`.
- Quantized matmul (Q4_K, Q6_K) dequantizes on the fly — verified correct via isolated test.

### Qwen3.5 Architecture
- **Hybrid**: 32 layers — every 4th (3,7,11,...,31) is full softmax attention; rest are delta-net SSM
- **Full attention**: joint Q+gate projection, separate K/V, QK-norm, MRoPE ([11,11,10,0]), GQA (16Q/4KV heads, head_dim=256), sigmoid-gated output
- **Delta-net**: combined QKV projection, conv1d, L2-normalized Q/K, fused gated delta net op, rms_norm × silu(z) output gate
- **Common**: RMSNorm (eps=1e-6), SwiGLU FFN, post-attention norm; tied embeddings
- Exact params: `models/arch/qwen35.arch.toml`

### Gemma4 Architecture
- **ISWA**: layers alternate between sliding-window attention (SWA) and global attention; pattern-based routing via `layers.routing.pattern` (array param `swa_pattern`)
- **SWA**: `attention` builder with `sliding_window = "true"` config; SWA mask built from `sliding_window` GGUF param
- **Shared KV cache**: non-KV layers (SWA layers) reuse the K/V cache of the nearest global-attention layer via `n_kv_shared_layers` param; `CacheDef.Shared` group name for TOML-declared sharing
- **`SharedKVState`**: `{K, V map[string]Tensor}` — in-graph K/V propagated from KV layers to non-KV layers within the same forward pass; keyed by `shared_kv_group` config string
- **GeGLU FFN**: `geglu` builder — `gelu(gate * x) * (up * x) → down`; uses `ggml.Gelu`
- **Embed scaling**: `architecture.embed_scale = true` → multiply token embeddings by `sqrt(n_embd)` before first layer
- **Logit softcapping**: `logit_softcapping` param → `cap * tanh(logits / cap)` applied after final norm; uses `ggml.Tanh`
- **Per-layer embeddings**: `pe_inp_gate` weight per layer (injected into residual stream after FFN); handled by `perLayerEmbedInject` in `graph.go`
- **Post-attention/FFN norms**: `attn_post_norm`, `ffn_post_norm` common weights; applied after block output before residual add
- **Layer output scaling**: `layer_output_scale` weight; multiplied into residual stream after FFN + post-norm
- **`MulMatSetPrecF32`**: forces F32 accumulation on a matmul op (required for correct Gemma4 attention scores)
- Exact params: `models/arch/gemma4.arch.toml`

### Culling Metadata Generation
Hollow system for user exploration/implementation

### Internal Logging

**Package**: `inference-lab-bench/internal/log` — replaces all prior `log.Printf` / `fmt.Fprintf(os.Stderr, ...)` usage.

**Call-site API** (use these everywhere):
```go
log.Debug(format string, args ...any)
log.Info(format string, args ...any)
log.Warn(format string, args ...any)
log.Error(format string, args ...any)
log.Fatal(format string, args ...any)  // Error + os.Exit(1)
```

**Output format**: `<HH:MM:SS>[LEVEL] message` — compact, no key-value spam.

**Level type**: `log.Level` (own type, not `slog`). Constants: `LevelDebug`, `LevelInfo`, `LevelWarn`, `LevelError`, `LevelNone`. `ValidLevelNames []string` is exported — CLI flag descriptions must reference it (DRY, not a hardcoded string).

**Initialization** — call once per CLI entry point, before any goroutines start:
```go
level, ok := log.ParseLevel(flagValue)  // (log.Level, bool)
log.InitLogger(logPath string, stderrLevel log.Level, logFileLine bool) error
```
On unrecognized input, `ParseLevel` returns `(LevelInfo, false)` — always check `ok`; ignoring it silently defaults to INFO.
- `logPath` empty → stderr only at `stderrLevel`
- `logPath` non-empty → stderr at `stderrLevel` + file at DEBUG (always full). File opens with a session boundary marker line.
- Pre-init default: stderr at INFO, no panic.

**CLI flags** (on `serve-api`, `chat`):
- `--log <path>` — log file path
- `--log-level <level>` — stderr level (`DEBUG|INFO|WARN|ERROR|NONE`), default `INFO`
- --log-file-line — enables logging of source file names and line numbers 
- `make serve` passes `--log bin/bench.log --log-level INFO`
- Batch commands (`gen-arch-diagram`, `gen-cull-metadata`) do not expose these flags

**ggml C library log routing** (`src/internal/inference/ggml/logging.go`):

`ggml.InitLogging()` registers a CGo callback via `ggml_log_set` that routes all ggml C library diagnostic output through the Go logger (DEBUG for most levels; WARN/ERROR for ggml levels 3/4). Call it once in `serve_api.go` and `chat.go` immediately after `log.InitLogger`. This eliminates all uncontrolled stderr output — `--log-level NONE` fully suppresses everything including ggml noise.

**CGo export constraint** — do not violate:

`ggmlGoLogCallback` in `logging.go` carries the `//export` directive. CGo auto-generates its C declaration in `_cgo_export.h`. A forward declaration for this symbol must NOT appear in `ggml_ops.h`. If it does, the CGo auto-generated declaration and the hand-written one will conflict at compile time. The symbol's C declaration is local to `ggml_ops.c` only.

**Constraints** (enforce these):
- `log.Fatal` is only permitted in `bench/` cobra entry points. Never call it from `internal/apiserver/` or `internal/inference/`.
- Never pass user-controlled values in the format string position — always use `%s`/`%v` args.
- Think content must be capped at 500 chars before logging; request string fields at 64 chars.
- `LOG_LEVEL` env var does not exist — level comes only from `--log-level`.
- No slog dependency; no third-party logging dependency.
- `ModuleMap.CullLog []string` in the culling package is a data structure, not a logger — do not confuse it with this package.

### Shell Scripting Style
- Shebang: `#!/usr/bin/env bash`
- Indent: 2 spaces
- Variables: always `${VAR}` not `$VAR`
- Tests: `[[ ]]` not `[ ]`; equality: `==` not `=`
- Functions: `function funcname() {` not bare `funcname() {`

## Adding a Model Architecture

**Critical invariant**: zero model-specific Go code. Never write `if arch == "foo"` anywhere in Go. All architecture differences must be expressed in the TOML DSL and generic block builders.

### Phase 0: Research — Understand the Architecture

Before writing any code, build a complete picture of the model.

1. **Read the llama.cpp reference implementation** — the authoritative source for how the architecture actually works at inference time:
   - `../llama.cpp/src/models/<arch>.cpp` — full forward pass (layer loop, attention, FFN, any novel ops)
   - `../llama.cpp/src/llama-arch.cpp` — GGUF tensor name mappings
   - `../llama.cpp/src/llama-hparams.h` — parameter names, defaults, helper functions (e.g., `is_swa()`, `has_kv()`)
   - `../llama.cpp/src/llama-model.cpp` — model init (look for arch-specific overrides like `f_attention_scale`, `rope_type`, `layer_reuse_cb`)
   - `../llama.cpp/src/llama-graph.cpp` — shared graph construction (attention, KV cache writes, mask construction)

2. **Scan the GGUF file** — verify actual metadata keys and tensor names:
   ```python
   # Use the python gguf library to inspect metadata and tensor inventory
   import gguf
   reader = gguf.GGUFReader("models/<model>.gguf")
   for field in reader.fields.values(): print(field.name, ...)
   for t in reader.tensors: print(t.name, t.shape)
   ```
   Common surprises: tensors named without `.weight` suffix, global vs per-layer tensors, bool arrays stored as GGUF bool type, scalar params stored as arrays.

3. **Read existing arch TOMLs** — find the closest existing architecture and understand the DSL patterns:
   - `models/arch/MODEL_ARCH_TOML_DSL_SPEC.md` — authoritative DSL spec
   - `models/arch/*.arch.toml` — existing architectures as templates

4. **Identify novel features** — catalog what this architecture does that no existing builder supports. Common categories:
   - New activation functions (need ggml op bindings)
   - New attention variants (config overrides on existing `attention` builder vs new builder)
   - New layer routing patterns (expression-based vs array-based)
   - New cache sharing patterns (param-driven shared KV, sliding window)
   - New pre/post-processing (embedding scaling, logit capping, per-layer embeddings)

### Phase 1: Add Missing ggml Ops (if needed)

If the architecture needs ggml ops not yet exposed:
1. `src/ggml_lib/src/ggml_ops.h` — add function declaration
2. `src/ggml_lib/src/ggml_ops.c` — add implementation (typically a one-line cast+forward to the underlying ggml function)
3. `src/internal/inference/ggml/ops_graph.go` — add Go wrapper function

Pattern: each op is 1 line .h + 1 line .c + 1 function in Go. Keep it mechanical.

### Phase 2: Add New Block Builders (if needed)

Only needed if the architecture uses a computation pattern not covered by existing builders. Check the builder registry in `blocks.go` first.

**Available builders** (as of this writing):
- `attention` — standard multi-head attention with GQA, RoPE. Highly configurable via block config: per-block head dim/count overrides, optional K/V (shared KV for non-KV layers), V-norm, sliding window mask, RoPE frequency factors, NeoX rope mode, custom kq_scale. This is the most flexible builder — prefer extending it via config over writing a new one.
- `full_attention_gated` — gated attention with joint Q+gate projection, MRoPE (Qwen3.5)
- `mla_attention` — multi-latent attention with low-rank Q/KV (DeepSeek2/GLM-4)
- `gated_delta_net` — delta net SSM (Qwen3.5 hybrid)
- `swiglu` — SiLU-gated FFN
- `geglu` — GELU-gated FFN (Gemma 4)
- `moe` — mixture of experts with optional shared expert and expert bias

To add a new builder:
1. `src/internal/inference/arch/block_<type>.go` — implement `BlockBuilder` or `FFNBuilder` (see interfaces in `blocks.go`). Must implement both `BuildStateless` and `BuildCached`.
2. Register in `init()` in `blocks.go`
3. Define `Contract()` — required/optional weights, required params, config schema with allowed values
4. `models/arch/block_svg/<name>.svg` — SVG snippet for diagram rendering

### Phase 3: Extend the TOML DSL (if needed)

If the architecture needs TOML/param features not yet supported:
- **New param types**: `params.go` handles int, float, string, int arrays, bool arrays. Add new `Get*` methods to `GGUFReader` interface if needed.
- **New routing patterns**: `weights.go:resolveBlockName()` supports expression-based (`rule`) and array-based (`pattern`) routing. Array routing uses `IntArr[pattern][layer_idx]` — nonzero → `if_true`, zero → `if_false`.
- **New cache sharing**: `cache.go:NewCache()` supports param-driven sharing via `n_kv_shared_layers` (layers past `nLayers - nKVShared` reuse the last KV layer's cache per block type).
- **Architecture-level flags**: `arch.go:ArchMeta` — add fields like `EmbedScale bool`, `TiedEmbeddings bool`, `NonCausal bool`.

### Phase 4: Extend graph.go (if needed)

`graph.go` owns the layer loop and pre/post-processing. If the architecture has novel features at this level:
- **Embedding scaling** — `m.Def.Architecture.EmbedScale` flag, applied after `GetRows`
- **Logit softcapping** — checked via `m.Params.Floats["logit_softcapping"]`, applied after LM head
- **Post-attention/post-FFN norms** — checked via `lt["attn_post_norm"]` / `lt["ffn_post_norm"]`
- **Per-layer embeddings** — `buildPerLayerEmbedSetup` + `perLayerEmbedInject` (Gemma 4 pattern)
- **Layer output scaling** — `lt["layer_output_scale"]` per-layer scalar
- **SWA mask** — second mask tensor built when `sliding_window` param exists

The layer loop ordering in `runLayers` is: `attn_norm → attention block → post_attn_norm → residual → ffn_norm → FFN → post_ffn_norm → residual → per-layer embed → layer_output_scale`. Verify this matches the reference implementation — ordering differences cause silent numerical divergence.

### Phase 5: Write the `.arch.toml` File

1. Copy the closest existing `.arch.toml` to `models/arch/<arch-name>.arch.toml`
2. Update `[architecture]` section (name, flags)
3. Update `[params]` — map param names to GGUF metadata keys. Use `?` suffix for optional params.
4. Update `[layers]` — count, prefix, routing
5. Update `[layers.common_weights]` — per-layer tensors shared across block types
6. Update `[blocks.*]` — per-block-type weights, config overrides, cache specs
7. Update `[ffn]` — FFN builder and weights

**Common pitfalls:**
- GGUF tensor names: check for `.weight` suffix presence/absence. Our loader tries both.
- Global vs per-layer tensors: per-layer tensors use `blk.@{layer_idx}.` prefix. Global tensors need `[weights.global]` entries. A per-layer weight entry that references a global-only tensor will use the fallback lookup (raw suffix as global name).
- Config overrides: block config values that are strings reference param names (`head_dim = "head_dim_swa"` → look up `params.Ints["head_dim_swa"]`). Literal numeric values are also supported (`kq_scale = 1.0`).
- Bool arrays from GGUF: stored as `IntArr` (0/1 values) after conversion in `resolveParam`.
- Shared KV groups: blocks that share cache must declare matching `shared_kv_group` config values.

### Phase 6: Debug Numerical Equivalence

This is typically the hardest phase. The model will load and run before the output is correct.

**Debugging strategy** — isolate the first layer/component where values diverge from llama.cpp:

1. Add temporary debug prints in `graph.go` and `block_attention_std.go` to dump first few float values of intermediate tensors at key checkpoints (after norm, after attention, after FFN, etc.)
2. Add matching debug prints to the llama.cpp reference model (at `cb()` callsites)
3. Run both on the same prompt, compare values checkpoint by checkpoint
4. The first checkpoint where values diverge identifies the buggy component

**Common sources of numerical divergence:**
- Wrong RoPE mode (standard vs NeoX — check `rope_type` in llama.cpp model init)
- Wrong attention scale (`f_attention_scale` — some models use 1.0 instead of 1/sqrt(headDim))
- Param resolution errors — wrong GGUF key, config override pointing to wrong param name
- Weight not loading — global tensor expected at per-layer path, fallback silently returns NilTensor
- Layer loop ordering mismatch — post-norm before vs after residual add
- Wrong mask — SWA layers getting full mask or vice versa
- Shared KV not wired — non-KV layers getting nil K/V from SharedKV because group names don't match

### Verification

All must pass before the work is complete:
```bash
make test && make integration-test
ALL_MODELS=true bash test_inference.sh "Hello"   # test against every loaded model
make equiv-test                                  # logprob equivalence vs llama-server (Homebrew)
make arch-diagrams                               # regenerate SVGs; confirm new arch renders correctly
```

The equivalence test (`test_equiv.sh`) is the critical gate. It compares our forward pass logprobs against llama.cpp's output on the same prompt. Threshold is ~0.01 logprob diff. GPU floating-point variance accounts for small differences; anything larger indicates a computation bug.
