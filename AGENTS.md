# Inference Lab Bench

From-scratch Go LLM inference engine for R&D into inference mechanics. Multi-model API server, data-driven architecture definition via TOML DSL, KV-cached and stateless inference, and visualization tooling.

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
- LLaDA-MoE-7B-A1B-Instruct gguf (`llada-moe` arch) — MoE diffusion model
- LLaDA-8B-Instruct gguf (`llada` arch) — dense diffusion model
- Qwen3.5-9B.st/ (`qwen3.5` arch)
- LLaDA-8B-Instruct.st/ (`llada` arch)

- **Two model formats**: GGUF (`*.gguf`) and safetensors (`*.st/` directories). Format is auto-detected at load time; inference code is identical above the `ModelReader` abstraction layer.
- KV-cached and stateless inference; TOML DSL drives loading, graph construction, and cache allocation regardless of source format. Only C code is a thin model-agnostic ggml op wrapper. Zero C++.
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
- **Model format**: GGUF (`*.gguf`) and safetensors (`*.st/` directories), auto-detected at load time
- **API**: OpenAI-compatible (`/api/v1/models`, `/api/v1/chat/completions`) with extensions: `"stateless"`, `"enable_thinking"`, `"elide_thinking"`, `"logprobs"`, `"top_logprobs"`, `"model":"default"`, `"diffusion"` (nested object: `steps`, `block_length`; ignored with a warning on non-diffusion models). Legacy `/v1/*` also supported. Diagnostic browser at `/diag/`. Control endpoint at `/ctl/` (`?memstats` = memory stats; `?quit` = graceful shutdown; `?quit&now` = immediate). Request-level overrides of server config defaults logged as `[req] param overrides: ...`.
- **Diagnostics**: `/diag/` serves files from `bin/diag/` — a useful location for dumping R&D diagnostic output (SVGs, etc.) that can be viewed in-browser while the server is running.
- **Config**: `config/api_config.toml` — `[server]` (host, port, auth_token), `[models]` (directory, default), `[inference]` (max_seq_len, enable_thinking_default, elide_thinking_default, log_thinking, single_resident_model, max_request_seq_len, strict_mode)
- **Default listen**: `0.0.0.0:11116`

## Source Layout

```
Makefile                        Top-level: test, integration-test, equiv-test, symlinks bin/ delegates the rest to src/
config/
  api_config.toml               API server config (models.default, auth, listen, max_seq_len)
  chat_config.toml              Chat client config (base_url=auto, model, system_prompt)
models/
  *.gguf                        Model files (not committed)
  *.st/                         Safetensors directories (not committed)
  arch/                         Architecture TOML definitions + generated SVGs
    MODEL_ARCH_TOML_DSL_SPEC.md DSL specification
    MODEL_ARCH_STMAP_TOML_DSL_SPEC.md STMap file DSL for safetensors name mapping
    block_svg/                  Hand-crafted SVG fragments per block builder
src/
  Makefile                      Go build, ggml build, test
  go.mod / go.sum               Module: inference-lab-bench
  bench/                        CLI entry point (cobra subcommands — thin wrappers)
    main.go                     Root command
    serve_api.go                serve-api
    chat.go                     chat
    gen_arch_diagram.go         gen-arch-diagram: SVG from TOML
  internal/
    log/                        Structured leveled logger (Debug/Info/Warn/Error/Fatal); see ### Internal Logging
    util/                       Project-wide utilities (LoadTOML, WriteJSON, BenchPaths, ResolvePaths, extension constants)
    apiserver/                  HTTP handlers (OpenAI-compatible: /api/v1/*), /ctl endpoint
    chatclient/                 Interactive chat client
    model/                      GGUF scanning + safetensors directory discovery
    inference/
      engine.go                 Engine, GenerateParams, DiffusionParams, IsDiffusion(), Generate() dispatch
      generate_cached.go        generateCached — KV/SSM-cached autoregressive loop
      generate_stateless.go     generateStateless — full-sequence recompute autoregressive loop
      generate_diffusion.go     generateDiffusion — block-based iterative masked denoising
      metrics.go                InferenceMetrics (timing, throughput, FinishReason)
      sampler.go                Greedy and top-p sampling, ComputeTopLogProbs (stable log-softmax)
      tokenizer.go              BPE tokenizer (GPT-2 byte-level + SentencePiece dual mode); chat template from GGUF via gonja; readGGUFTokensRaw direct GGUF binary reader
      arch/
        arch.go                 ArchDef, TOML parser, Validate() with builder contracts
        arch_util.go            Shared tensor-op helpers, cache key constants, configIntOr/configFloatOr/configStr, attentionScale
        model_reader.go         ModelReader interface (metadata + tensor loading)
        safetensors_index.go    LoadSafetensorsIndex() — JSON index parser + shard header reader
        safetensors_reader.go   Safetensors ModelReader implementation; BF16→F32 conversion
        stmap.go                .arch.stmap.toml parser (LoadArchSTMap, FindSTMapByHFClass)
        block_attention_util.go Shared attention helpers (scaledDotProductAttention, RoPE, KV cache)
        params.go               Param resolver + routing expression eval
        weights.go              Weight resolver (template expansion, layer routing)
        blocks.go               BlockBuilder/FFNBuilder interfaces, ForwardCaptures, SharedKVState, GraphInputs, LayerCache, registry
        block_attention.go      attention — standard multi-head (Llama, Gemma4 SWA/global; sliding_window, shared KV)
        block_attention_gated.go full_attention_gated — gated attention with QK-norm and MRoPE (Qwen3.5)
        block_attention_mla.go  mla_attention (DeepSeek2/GLM-4)
        block_ssm.go            gated_delta_net (Qwen3.5 SSM)
        block_ffn.go            swiglu + geglu (shared gluBuilder, activation-parameterized)
        block_ffn_moe.go        moe (MoE + shared expert + expert bias)
        modules.go              Module, ModuleMap
        module_map.go           BuildModuleMap, BuildTensorDimsMap
        weight_store.go         WeightStore: immutable GPU-resident tensor storage
        model.go                GenericModel: GGUF + safetensors loading via ModelReader, per-layer FFN routing
        cache.go                Cache allocation (NewCache, GenericCache)
        graph.go                Forward pass (ForwardStateless, ForwardStatelessAllLogits, ForwardCached, forwardStatelessCore, runLayers)
        validate_lines.go       ResolveErrorLines (TOML key path → source line)
        keys.go                 Canonical string constants for map keys, module identifiers, config keys
        model_reader_safetensors_xform.go  Load-time safetensors tensor transform pipeline (reorder, permute)
      archdiagram/              SVG diagram renderers (separate package)
        arch_diagram.go         Architecture overview SVG renderer (RenderArchDiagram)
        layers_diagram.go       Layers Diagram SVG renderer (RenderLayersDiagram)
        palette.go              Shared diagram palette (Pal())
        weights.go              Weight resolution for diagrams (ResolveWeightsForDiagram)
      ggml/                     Go wrappers for ggml ops (~43 functions)
  ggml_lib/                     C op wrappers + ggml build
  third_party/ggml/             ggml git submodule
test_inference.sh               Test harness (works with bench or llama-server via IS_LLAMA)
test_equiv.sh                   Logprob equivalence test: bench vs llama-server, stateless vs non-stateless, flash vs standard
```

All SVG renderers share a single palette from `archdiagram/palette.go:Pal()`. To change any color, update that map only.

## Build & Run

**Do not use `go build` directly.** Always use `make` targets — the Makefile handles ggml C compilation, cgo flags, symlinks, and output placement.

```bash
make                    # build bench binary, symlink config+models into bin/
make serve              # build + start API server (--log bin/bench.log --log-level INFO)
make chat               # build + start interactive chat client
make arch-diagrams      # rebuild SVG diagrams from models/arch/*.arch.toml
make st-tok-ggufs       # (re)generate tokenizer.gguf sidecar in every models/*.st/ dir
make test               # run go unit tests
make integration-test   # test inference end to end for all models

# CLI subcommands
./bin/bench serve-api
./bin/bench chat
./bin/bench gen-arch-diagram [--layers] [--blocks] <input.toml> [output.svg]

# Safetensors conversion tools
tools/hf_to_gguf_setup.sh       One-time setup: Python venv + llama.cpp convert script
tools/hf_to_gguf.sh             Tokenizer sidecar generation + full convert passthrough
  tools/hf_to_gguf.sh --bench-tokenizer <model-name>  # generates tokenizer.gguf in .st/ dir

# After adding a new models/<name>.st/ directory, build its tokenizer.gguf sidecar:
make st-tok-ggufs               # builds the sidecar in every .st/ dir missing one
#   — the sidecar is required to load a safetensors model (bench's tokenizer
#     path is GGUF-only). `make serve` invokes this automatically; running
#     `./bin/bench serve-api` directly does not, so run it manually after
#     dropping a new .st/ directory into models/.

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
* The acceptance gate for every change, without exception: `make test && make integration-test && make equiv-test`. No partial gates — this applies to mechanical fixes and refactors just as much as inference changes.
* Only one instance of `make integration-test` or `make equiv-test` can run at a time — port and GPU resource contention prevent concurrent runs. All validation is strictly sequential.
* `make equiv-test` compares top-1 logprobs against `llama-server` (Homebrew) and between the GGUF and safetensors loader paths to verify inference correctness within GPU floating-point variance. This is the authoritative correctness gate.
* Run `ALL_MODELS=true bash test_inference.sh "..."` before declaring any inference change complete. Llama is easy mode — edge cases surface on Qwen3.5, Qwen3.5-MoE, DeepSeek2, and Gemma4.

**Load-time defensive checks** — the loader runs three cheap sanity checks that turn silent numerical failures into loud load-time errors. Do not disable them without cause:
* `arch.ResolveParams` validates every required param is present and typed correctly; the loader aborts before VRAM allocation if any required key is missing or has a zero/garbage value that would cause silent divergence downstream (e.g. `rms_eps=0`).
* `ggml.ValidateRowData` inspects every tensor's raw bytes before upload — full element scan for float types, block scale/delta scan for quantized types (near-zero cost). Catches corrupt weight files and reader type/shape mismatches at load time rather than as NaN logits mid-generation.
* `ValidateLogits` runs at every sampler chokepoint (cached, stateless, diffusion) and fails the request if any logit is NaN/Inf. Diffusion wraps the error with `blockNum` / `step`; autoregressive paths wrap with `sample:`.

**Param dumps** — both GGUF and safetensors readers emit a sorted `[param] key = value` DEBUG dump of all metadata at load. Visual diff between the two is the fastest way to catch stmap errors when porting a new architecture to safetensors.

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
- Batch commands (`gen-arch-diagram`) do not expose these flags

**ggml C library log routing** (`src/internal/inference/ggml/logging.go`):

`ggml.InitLogging()` registers a CGo callback via `ggml_log_set` that routes all ggml C library diagnostic output through the Go logger (DEBUG for most levels; WARN/ERROR for ggml levels 3/4). Call it once in `serve_api.go` and `chat.go` immediately after `log.InitLogger`. This eliminates all uncontrolled stderr output — `--log-level NONE` fully suppresses everything including ggml noise.

**CGo export constraint** — do not violate:

`ggmlGoLogCallback` in `logging.go` carries the `//export` directive. CGo auto-generates its C declaration in `_cgo_export.h`. A forward declaration for this symbol must NOT appear in `ggml_ops.h`. If it does, the CGo auto-generated declaration and the hand-written one will conflict at compile time. The symbol's C declaration is local to `ggml_ops.c` only.

**Constraints** (enforce these):
- `log.Fatal` is only permitted in `bench/` cobra entry points. Never call it from `internal/apiserver/`, `internal/inference/`, or any utility package — including `internal/util/paths.go`. Library code returns errors; only the cobra entry decides whether to terminate. `util.ResolvePaths()` returns `(BenchPaths, error)` to honor this.
- Never pass user-controlled values in the format string position — always use `%s`/`%v` args.
- Think content must be capped at 500 chars before logging; request string fields at 64 chars.
- `LOG_LEVEL` env var does not exist — level comes only from `--log-level`.
- No slog dependency; no third-party logging dependency.

### Path Resolution and Injection

`util.BenchPaths` carries the standard directory layout (`ExeDir`, `ConfigDir`, `ModelsDir`, `ArchDir`, `DiagDir`) derived from the running executable's location. Two rules:

- **Resolution happens once, in `bench/`**. `util.ResolvePaths() (BenchPaths, error)` is called from each cobra entry point (`serve_api.go`, `chat.go`, `gen_arch_diagram.go`). The result is cached for the process lifetime via `sync.Once`. On error, the cobra entry calls `log.Fatal` itself.
- **Injection downstream**. Library code never calls `ResolvePaths`. `BenchPaths` is passed into `apiserver.NewServer(paths, cfg, manager)`, then forwarded to `inference.NewEngine(...)`. `Server` stores `paths util.BenchPaths` and reads `paths.DiagDir` / `paths.ArchDir` from it. This keeps utility and library packages free of `os.Exit` and the executable-path dance.

The `BENCH_EXE_DIR` env var is honored by `ResolvePaths` for debug/IDE configurations where source CWD and runtime CWD diverge.

### Graph Context Sizing (`arch` ↔ `ggml`)

Two named pieces of state govern every ggml graph the arch package builds:

- `arch.maxGraphNodes = 16384` — single source of truth for the per-pass cgraph node budget. Sized for the widest forward pass we build (stateless + cached, all architectures). Drives both context-arena sizing and `NewGraph` / `NewSched` allocations — all three must agree.
- `arch.graphCtxSize() int` — returns `ggml.GraphContextSize(maxGraphNodes)`. The underlying `ggml.GraphContextSize(maxNodes)` is principled: `GraphOverheadCustom(maxNodes, false) + TensorOverhead*maxNodes + 64` alignment slop, computed from ggml's own accounting. No empirical multipliers, no magic numbers.

`ggml.NewGraphContext(memSize int, allocPerm AllocPerm)` is non-variadic — every caller declares `AllocPermDisallow` (graph-build context, descriptors only — the normal case) or `AllocPermAllow` (data-arena scratch context, e.g. load-time type conversion). There is no default. The named `AllocPerm` type catches accidental swaps with `memSize`.

Canonical logical weight names live in `arch_util.go`: `WeightAttnNorm`, `WeightFFNNorm`, and the `Cache*` keys. Never inline these literal strings in graph or module-map code.

`ResolvedLayerWeights.Prefix` (set by `ResolveWeights` to the expanded per-layer prefix, e.g. `"blk.5."`, with trailing dot) is the canonical source for any per-layer prefix consumer — module map, tensor dims, diagram code. Never reconstruct `blk.<N>.` from `layer_idx` ad hoc.

### ggml Wrapper Conventions

- `ggml.GGMLType` is a named integer type for tensor element types (`TypeF32`, `TypeF16`, `TypeQ4_K`, etc.). Distinct from `int` — tensor type arguments cannot be silently swapped with unrelated ints (ne dimensions, mode flags, etc.).
- Nullable tensor parameters in op wrappers use the `opt` prefix to flag that callers may pass `NilTensor()`: `SoftMaxExt(ctx, a, optMask, ...)`, `RopeExt(ctx, a, pos, optFreqFactors, ...)`, `FlashAttnExt(ctx, q, k, v, optMask, ...)`. Required tensor parameters have no prefix.
- `Buffer.Clear(byte)` zeroes (or fills) an entire backend buffer in a single C call. `GenericCache.Clear()` uses this instead of per-tensor zero loops;

### Model Loader Phase Structure (`arch/model.go`)

`newGenericModelFromReader` builds a `genericModelBuilder` and calls `b.build()`. The builder carries partial state across phase methods (`checkMemory`, `resolveArch`, `initBackendsAndArena`, `uploadWeights`, `buildWeightStore`, `assignBuilders`, `createComputeResources`) and uses `WeightStore` as the ownership cut line:

- Before `buildWeightStore`: the builder owns `gpu`, `cpu`, `weightCtx`, `weightBuf` individually — `cleanupOnError` frees them one by one.
- After `buildWeightStore`: `store` owns all four — `cleanupOnError` calls `store.Close()` and returns.
- After `createComputeResources`: the model owns `cachedCtx` / `cachedSched` — `cleanupOnError` frees them before closing the store.

This avoids the double-free hazard of the previous open-coded loader. When adding new phases, place them in the sequence and update the cleanup logic; do not pass loose state by parameter.

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
   - ```make llama-cpp``` will clone our reference version 
   - `tools/llama.cpp/src/models/<arch>.cpp` — full forward pass (layer loop, attention, FFN, any novel ops)
   - `tools/llama.cpp/src/llama-arch.cpp` — GGUF tensor name mappings
   - `tools/llama.cpp/src/llama-hparams.h` — parameter names, defaults, helper functions (e.g., `is_swa()`, `has_kv()`)
   - `tools/llama.cpp/src/llama-model.cpp` — model init (look for arch-specific overrides like `f_attention_scale`, `rope_type`, `layer_reuse_cb`)
   - `tools/llama.cpp/src/llama-graph.cpp` — shared graph construction (attention, KV cache writes, mask construction)

2. **Scan the model file or directory** — verify actual metadata keys and tensor names:

   **GGUF**:
   ```python
   # Use the python gguf library to inspect metadata and tensor inventory
   import gguf
   reader = gguf.GGUFReader("models/<model>.gguf")
   for field in reader.fields.values(): print(field.name, ...)
   for t in reader.tensors: print(t.name, t.shape)
   ```

   **Safetensors** — inspect `config.json` and `model.safetensors.index.json` (or shard headers):
   ```python
   import json
   with open("models/<model>.st/config.json") as f: cfg = json.load(f)
   for t_name, t_info in index["weight_map"].items(): print(t_name, t_info["dtype"], t_info["shape"])
   ```

   Common surprises: tensors named without `.weight` suffix, global vs per-layer tensors, bool arrays stored as GGUF bool type, scalar params stored as arrays.

   **If using safetensors**, you will also need to create a `.arch.stmap.toml` mapping HF names → GGUF-equivalent names (see `models/arch/MODEL_ARCH_STMAP_TOML_DSL_SPEC.md`). The stmap is per-architecture, not per-model — all variants of the same architecture share one stmap file.

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
- **New param types**: `params.go` handles int, float, string, int arrays, bool arrays. Add new `Get*` methods to the `ModelReader` interface if needed (and to both GGUF and safetensors adapters). Note: safetensors models only provide scalar params from `config.json` — array types (`GetArrInts`, `GetArrBools`) are not available from that format.
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

**Per-layer hook threshold**: `runLayers` implements optional per-layer behaviors as weight- or param-presence nil-checks (see its doc comment for the current list). Adding the (N+1)th hook at N ≥ 5 should trigger a design review: is this the right layer for the feature, or does it warrant an explicit per-layer hook registry? At that scale the readability cost of another nil-check outweighs the benefit of inlining another feature.

### Phase 5: Write the `.arch.toml` File

1. Copy the closest existing `.arch.toml` to `models/arch/<arch-name>.arch.toml`
2. Update `[architecture]` section (name, flags)
3. Update `[params]` — map param names to GGUF metadata keys. Use `?` suffix for optional params.
4. Update `[layers]` — count, prefix, routing
5. Update `[layers.common_weights]` — per-layer tensors shared across block types
6. Update `[blocks.*]` — per-block-type weights, config overrides, cache specs
7. Update `[ffn]` — FFN builder and weights

**If using safetensors**, also write `models/arch/<arch-name>.arch.stmap.toml` mapping HF names → GGUF-equivalent names (see `models/arch/MODEL_ARCH_STMAP_TOML_DSL_SPEC.md`). The stmap covers:
- `[params]` — HF `config.json` keys → GGUF metadata keys
- `[layer_prefix].hf` — HF per-layer prefix with `{N}` substitution
- `[tensors]` — our short tensor names → HF short tensor names
- `[tensors.global]` — our short global names → HF full tensor names

**Common pitfalls:**
- GGUF tensor names: check for `.weight` suffix presence/absence. Our loader tries both.
- Global vs per-layer tensors: per-layer tensors use `blk.@{layer_idx}.` prefix. Global tensors need `[weights.global]` entries. A per-layer weight entry that references a global-only tensor will use the fallback lookup (raw suffix as global name).
- Config overrides: block config values that are strings reference param names (`head_dim = "head_dim_swa"` → look up `params.Ints["head_dim_swa"]`). Literal numeric values are also supported (`kq_scale = 1.0`).
- Bool arrays from GGUF: stored as `IntArr` (0/1 values) after conversion in `resolveParam`.
- Shared KV groups: blocks that share cache must declare matching `shared_kv_group` config values.

### Phase 6: Debug Numerical Equivalence

This is typically the hardest phase. The model will load and run before the output is correct.

**Debugging strategy** — isolate the first layer/component where values diverge from llama.cpp:

1. Add temporary debug prints in `graph.go` and `block_attention.go` to dump first few float values of intermediate tensors at key checkpoints (after norm, after attention, after FFN, etc.)
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
