# Go Inference Lab Bench

A from-scratch Go language LLM inference lab bench for doing R&D related to the mechanics of inference.
Provides local multi-model inference server, multiple ways of testing inference, data-driven model architecture (definition, loading, and visualization), and general utilities.

## Project Priorities
- Focus 
  - this is an R&D tool. No optimizing for production concerns at the expense of clarity & tinkerability.
  - this is meant to be a fairly general tool, don't make choices that close lots of doors
- Consistency - naming patterns, data patterns, conventions - surprises are bad. R&D work means focusing on the problem, not keeping the tooling's mess straight in your head or token window
  - the static data language of choice is TOML, when not constrained by requirements, (e.g. json for OpenAPI payloads)
  - discipline-specific terminology is adhered to for names where high-consensus terms exist (e.g. 'module' for portion of a layer)
- Preserve data-driven model architecture support purity - model-specific code is BAD and subverts the design
  - A great deal of effort has been expended in making the modular block system & toml-based DSL mechanism work. 
  - It has repeatedly demonstrated its value
    - at the bootstrapping of this project getting qwen3.5 model working in pure C++ took 2 days of solid work.
    - adding the last 2 architectures (so far), one of which was deepseek2, took 2 hours for both.
- Cleanliness & Consision
  - DRY adherence is top priority
    - For all its advantages, go is verbose language
    - Code reuse is paramount to prevent wasting token window space on redundancy
  - any functionality needed two or more times needs to get refactored into a utility function.
    - if it is a general utility it goes into a project-wide utility file (`src/internal/util/project_util.go`)
    - if it is package-specific and utilization spans multiple files it goes into `src/internal/[package]/[package]_util.go`
    - if it is specific to a single file, put the util function implementation in that file
  - Complex code must be reserved for complex needs. A 100 line algorithm for O(1) efficiency over a 64 element data set is a waste when O(n) is 5 lines and will never make a noticeable difference
  - Separation of Concerns
    - systems should maintain ignorance of other system internals
    - one function or system 'pre-digesting' data for consumption by another function or system in a way only useful to the callee is a sign of violation of this principle.
- Currently only macOS supported but ports to Linux and Windows are coming
  - Do not accummulate tech debt related to portability
  - The only platform-specific debt at this time are around the ggml binding layer
    - ggml 3rd party C++ project build options provided to cmake
    - ggml_lib C project compiler flags
    - binding layer functions for other platforms need wiring (only mac/metal exposed right now)

## Current Status


Known working models (from huggingface.co, links in README.md):
* Llama-3.2-3B-Instruct-f16.gguf, llama-3.2-3b-instruct-q4_k_m.gguf (dense, `llama` arch)
* Qwen3.5-4B_Abliterated.f16.gguf (dense, `qwen35` arch)
* Qwen3.5-9B-abliterated.f16.gguf (dense, `qwen35` arch)

In-progress models:
* gemma-4-E4B-it-Q4_K_M.gguf (dense, `gemma4` arch) — runs and generates coherent text but fails equivalence test (logprob diff -0.97). See `.claude/gemma4_completion.md` for investigation handoff.

- **Inference** KV cached and stateless inference. Model architectures are defined via a TOML DSL (`models/arch/*.arch.toml`) that drives GGUF loading, graph construction, and cache allocation. Block builders implement the graph-level ops in Go. The only C code is a thin, model-agnostic ggml op wrapper layer (`ggml_ops.h/.c`). Zero C++ in the project.
- HTTP server with Bearer auth, model listing, streaming SSE + non-streaming completions, logprobs
- **TOML DSL-driven model loading**: architecture definitions in `models/arch/*.arch.toml` declare params, layer routing, weight bindings, cache specs, and FFN type
- **Block builder registry**: `attention` (standard, with config-based param overrides, optional K/V, SharedKV, V-norm, SWA mask, RoPE freq factors), `full_attention_gated` (Qwen3.5), `mla_attention` (DeepSeek2/GLM-4), `gated_delta_net`, `swiglu`, `geglu`, `moe_with_shared` — pluggable graph construction
- **Qwen3.5 dense**: hybrid forward pass (attention + delta-net SSM), logit-perfect vs reference
- **Qwen3.5 MoE**: softmax router, top-k expert dispatch via `ggml_mul_mat_id`, sigmoid-gated shared expert, weight normalization
- **Llama 3.2**: standard attention with GQA, standard RoPE, SwiGLU FFN
- **DeepSeek2/GLM-4**: MLA attention (low-rank Q/KV, Q-nope absorption), hybrid dense/MoE FFN, expert bias
- GGUF loading entirely in Go (metadata parsing, weight loading to Metal VRAM)
- Unsupported architectures auto-filtered at scan time
- **Zero model-specific code**: chat templates executed from GGUF `tokenizer.chat_template` via gonja; BOS/EOS from GGUF metadata; no per-architecture tokenizer branches. Thinking mode controlled entirely by template `enable_thinking` variable — no post-render prompt manipulation.
- BPE tokenizer loaded from GGUF metadata (Qwen3.5/tiktoken-compatible, 248K vocab)
- Greedy + top-p sampling; logprob computation via stable log-softmax
- KV cache + SSM state persistence: prefill in one pass, decode one token at a time
- Stateless fallback via `"stateless": true`
- Default model resolution from config, `"model": "default"` supported
- **SVG architecture visualizer**: `bench gen-arch-diagram` generates architecture diagrams (`*.arch.svg`) and per-layer module map diagrams (`*.layers.svg`) from TOML definitions

## Architecture
- **Language**: Go (all model logic + HTTP server) + pure C via CGO (ggml op wrappers only)
- **Model definition**: TOML DSL (`models/arch/*.arch.toml`) — declares params, layer routing, weight bindings, cache specs
- **Inference backend**: ggml (git submodule), Metal-accelerated
- **Model format**: GGUF (files in `models/`)
- **Target models**: Llama 3.2 (3B), Qwen3.5 dense (4B, 9B), Qwen3.5 MoE (35B-A3B), DeepSeek2/GLM-4 (4.7B)
- **API**: OpenAI-compatible (`/api/v1/models`, `/api/v1/chat/completions`) with extensions: `"stateless"`, `"enable_thinking"`, `"elide_thinking"`, `"logprobs"`, `"top_logprobs"`, `"model":"default"`. Legacy `/v1/*` also supported. Diagnostic browser at `/diag/`. Control endpoint at `/ctl` (graceful shutdown via `?quit` or immediate via `?quit&now`).
- **Diagnostics**: `/diag/` serves files from `bin/diag/` — a useful location for dumping R&D diagnostic output (SVGs, data files, etc.) that can be viewed in-browser while the server is running. The directory is auto-created at startup.
- **Config**: `api_config.toml` — server, models, inference settings (`max_seq_len`, `enable_thinking_default`, `elide_thinking_default`, `log_thinking`)
- **Default listen**: `0.0.0.0:11116`

## Source Layout

```
Makefile                        Top-level: test, integration-test, equiv-test, symlinks bin/ delegates the rest to src/
config/
  api_config.toml               API server config (models.default, auth, listen, max_seq_len)
  chat_config.toml              Chat client config (base_url=auto, model, system_prompt)
models/
  *.gguf                        Model files (not committed)
  arch/                         Architecture TOML definitions (*.arch.toml) + generated SVGs (*.arch.svg, *.layers.svg)
    model_arch_toml_dsl_spec.md               DSL specification
    block_svg/                  Hand-crafted SVG fragments per block builder
    editor/                     Web-based TOML editor (HTML/JS/CSS)
src/
  Makefile                      Go build, ggml build, test
  go.mod / go.sum               Module: inference-lab-bench
  bench/                        CLI entry point (cobra subcommands — thin wrappers that dispatch to internal/)
    main.go                     Root command (cobra)
    serve_api.go                serve-api: dispatches to apiserver
    chat.go                     chat: dispatches to chatclient
    gen_arch_diagram.go         gen-arch-diagram: SVG from TOML
    arch_editor.go              arch-editor: dispatches to archeditor
  internal/
    util/                       Project-wide utilities (LoadTOML, WriteJSON, BenchPaths, extension constants)
    apiserver/                  HTTP handlers (OpenAI-compatible: /api/v1/*), /ctl endpoint
    chatclient/                 Interactive chat client implementation
    model/                      GGUF scanning + metadata
    archeditor/                 Arch editor web server
    inference/
      engine.go                 Generation loop (tokenize → forward → sample)
      metrics.go                InferenceMetrics (timing, throughput), TokenLogProb, ByteArray
      sampler.go                Greedy and top-p sampling, ComputeTopLogProbs (stable log-softmax)
      tokenizer.go              BPE tokenizer; chat template executed from GGUF via gonja (no model-specific code)
      arch/
        arch.go                 ArchDef, TOML parser, Validate() with builder contracts
        arch_util.go            Shared tensor-op helpers (rmsNormApply, projectReshape3D, attentionScale), cache key constants (CacheK, CacheV, ...)
        block_attention_util.go Shared attention helpers (scaledDotProductAttention, writeCacheKV, selectCachedKV, selectSharedKV, applyRoPEPair)
        params.go               Param resolver + routing expression eval
        weights.go              Weight resolver (template expansion, layer routing)
        blocks.go               BlockBuilder/FFNBuilder interfaces, registry
        block_attention.go      full_attention_gated (Qwen3.5)
        block_attention_std.go  attention (standard MHA+GQA+RoPE; prepareQKV pipeline, selectMask, KV/SharedKV)
        block_attention_mla.go  mla_attention (DeepSeek2/GLM-4)
        block_ssm.go            gated_delta_net (Qwen3.5 SSM)
        block_ffn.go            swiglu
        block_ffn_moe.go        moe_with_shared (MoE + expert bias)
        modules.go              Module, ModuleMap
        module_map.go           BuildModuleMap, BuildTensorDimsMap
        weight_store.go         WeightStore: immutable weight storage
        model.go                GenericModel: GGUF loading, per-layer FFN routing
        cache.go                Cache allocation (NewCache, GenericCache)
        graph.go                Forward pass (ForwardStateless, ForwardCached, runLayers, buildCausalMaskData, buildSWAMaskData)
        validate_lines.go       ResolveErrorLines (TOML key path → source line mapping)
        diagram_util.go         Shared diagram palette (diagramPalette, palPrefix, palPrefixBuilder)
        arch_diagram.go         Architecture overview SVG renderer (gen-arch-diagram)
        module_map_diagram.go   Module map SVG renderer (per-layer tensor details)
      ggml/                     Go wrappers for ggml ops (~36 functions)
  ggml_lib/                     C op wrappers + ggml build
  third_party/ggml/             ggml git submodule
test_inference.sh               Test harness
```

### Diagram Style System
All SVG diagram renderers (`arch_diagram.go`, `module_map_diagram.go`) share a unified color palette defined in `diagram_util.go:diagramPalette()`. Block-type colors (attention, recurrent, FFN, global, norm), gradient definitions, and UI chrome colors (text hierarchy, borders, backgrounds, arrows) are sourced exclusively from this palette. To change a color, update the single palette map — both diagram types will reflect the change.

## Build & Run

```bash
make                    # builds bench binary, symlinks config+models into bin/
make serve              # build + start API server (stderr to bin/bench.log)
make chat               # build + start interactive chat client
make arch-diagrams      # (re)builds SVG architecture diagrams from models/arch/*.arch.toml

# CLI subcommands
./bin/bench serve-api           # run inference API server
./bin/bench chat                # interactive chat client
./bin/bench gen-arch-diagram    # generate SVG from TOML definition
./bin/bench arch-editor         # launch web-based TOML editor

# Test - by default assumes server is already running; set FORCE_NEW_SERVER=true to kill+restart
bash test_inference.sh "What is 2+2?"           # one-shot
bash test_inference.sh --loop                   # interactive (help/stateless/think/quit)

# Environment overrides for test_inference.sh
FORCE_NEW_SERVER=true bash test_inference.sh "Hi"
MODEL=Qwen3.5-4B_abliterated.Q4_K_M MAX_TOKENS=100 bash test_inference.sh "Hi"
THINK=true ELIDE_THINK=false bash test_inference.sh "Hi" # enable thinking and show <think> content (for thinking models) 
ALL_MODELS=true bash test_inference.sh "Hi"     # applies the same query to each loaded model sequentially in one invocation

# In --loop mode: /all-models toggles ALL_MODELS mode; /model [n] selects a specific model
# IMPORTANT: always test with ALL_MODELS=true before considering a change complete.
# Llama is easy mode — Qwen3.5, Qwen3.5-MoE, and DeepSeek2/GLM4 are where issues surface.

# API
curl localhost:11116/api/v1/models
curl -X POST localhost:11116/api/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","messages":[{"role":"user","content":"Hi"}]}'
```

**Validating Changes**
* `make test && make integration-test` must pass.
* When adding new model architectures or changing inference code, `make equiv-test` must also pass — it compares top-1 logprobs against `llama-server` (Homebrew) to verify inference correctness within Metal floating-point variance.

## Key Technical Details

### TOML Model DSL

Model architectures are declared in TOML files (`models/arch/*.arch.toml`). The DSL captures:
- **Params**: GGUF metadata key mappings + derived arithmetic expressions
- **Layer routing**: expression-based rules determining which block type each layer uses. `@{name}` references builtins (`@{layer_idx}` = 0-based layer index), `${name}` dereferences a resolved GGUF param. Evaluated at model load time to determine the static layer structure.
- **Weight bindings**: GGUF tensor name templates with `blk.@{layer_idx}.` prefix expansion
- **Cache specs**: per-block-type cache tensor dimensions and dtypes
- **FFN type**: which feed-forward builder to use. Optional `[ffn_alt]` for per-layer FFN routing (e.g., dense SwiGLU for first N layers, MoE for the rest — auto-detected from GGUF weights).
- **Tokens**: `[tokens]` section declares `think_open`, `think_close` per architecture.
- **Param defaults**: `[params.defaults]` provides fallback values when GGUF params resolve to 0.

Block builders (`full_attention`, `full_attention_gated`, `mla_attention`, `gated_delta_net`, `swiglu`, `moe_with_shared`) implement the ggml graph construction. Adding a new model that uses existing block types requires only a new `.arch.toml` file. A genuinely new block type requires a new builder implementation. Optional GGUF params use `?` suffix (e.g., `"arch.key?"` — silently skipped if missing).

See `models/arch/model_arch_toml_dsl_spec.md` for the full DSL spec.

### ggml Semantics (verified)
- `ggml_permute(ctx, a, ax0, ax1, ax2, ax3)`: input dim i goes to output position ax_i. `ne[ax_i] = a->ne[i]`.
- `ggml_mul_mat(A, B)`: contracts over `A->ne[0] == B->ne[0]`. Result shape `[A->ne[1], B->ne[1], ...]`. Supports GQA broadcasting when `B->ne[2] % A->ne[2] == 0`.
- `ggml_rms_norm`: normalizes over ne[0] independently for each slice.
- `ggml_soft_max_ext(a, mask, scale, max_bias)`: computes `softmax(scale * a + mask)` over ne[0].
- Quantized matmul (Q4_K, Q6_K) dequantizes on the fly — verified correct via isolated test.

### Qwen3.5 Architecture
- **Hybrid**: 32 layers — every 4th layer (3,7,11,...,31) is full softmax attention, rest are delta-net SSM
- **Full attention layers**: joint Q+gate projection, separate K/V, QK-norm, MRoPE (sections [11,11,10,0]), GQA (16Q/4KV heads, head_dim=256), gated output (sigmoid gate)
- **Delta-net layers**: combined QKV projection, conv1d, L2-normalized Q/K, gated delta net (fused ggml op), gated normalization (rms_norm × silu(z))
- **Common**: RMSNorm (eps=1e-6), SwiGLU FFN, post-attention norm before FFN
- Tied embeddings: LM head reuses `token_embd.weight`
- Exact param values: see `models/arch/qwen35.arch.toml`

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
   - `models/arch/model_arch_toml_dsl_spec.md` — authoritative DSL spec
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
- `moe_with_shared` — mixture of experts with shared expert and expert bias

To add a new builder:
1. `src/internal/inference/arch/block_<type>.go` — implement `BlockBuilder` or `FFNBuilder` (see interfaces in `blocks.go`). Must implement both `BuildStateless` and `BuildCached`.
2. Register in `init()` in `blocks.go`
3. Define `Contract()` — required/optional weights, required params, config schema with allowed values
4. `models/arch/block_svg/<name>.svg` — SVG snippet for diagram rendering
5. `models/arch/editor/editor.js` — `BUILDER_` table entries for the editor

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
bash test_llama_equiv.sh                         # logprob equivalence vs llama-server (Homebrew)
make arch-diagrams                               # regenerate SVGs; confirm new arch renders correctly
```

The equivalence test (`test_llama_equiv.sh`) is the critical gate. It compares our stateless forward pass logprobs against llama.cpp's output on the same prompt. Threshold is ~0.01 logprob diff. Metal floating-point variance accounts for small differences; anything larger indicates a computation bug.

### Shell Scripting Style
- Shebang: `#!/usr/bin/env bash`
- Indent: 2 spaces
- Variable dereferencing: always use curly braces (`${VAR}`, not `$VAR`)
- Test enclosure: `[[ ]]`, not `[ ]`
- Equality: `==`, not `=`
- define functions using keyword syntax `function funcname() {` not bare syntax `funcname() {`
