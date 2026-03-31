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

- **Inference** KV cached and stateless inference. Model architectures are defined via a TOML DSL (`models/arch/*.arch.toml`) that drives GGUF loading, graph construction, and cache allocation. Block builders implement the graph-level ops in Go. The only C code is a thin, model-agnostic ggml op wrapper layer (`ggml_ops.h/.c`). Zero C++ in the project.
- HTTP server with Bearer auth, model listing, streaming SSE + non-streaming completions
- **TOML DSL-driven model loading**: architecture definitions in `models/arch/*.arch.toml` declare params, layer routing, weight bindings, cache specs, and FFN type
- **Block builder registry**: `full_attention`, `full_attention_gated`, `mla_attention`, `gated_delta_net`, `swiglu`, `moe_with_shared` — pluggable graph construction
- **Qwen3.5 dense**: hybrid forward pass (attention + delta-net SSM), logit-perfect vs reference
- **Qwen3.5 MoE**: softmax router, top-k expert dispatch via `ggml_mul_mat_id`, sigmoid-gated shared expert, weight normalization
- **Llama 3.2**: standard attention with GQA, standard RoPE, SwiGLU FFN
- **DeepSeek2/GLM-4**: MLA attention (low-rank Q/KV, Q-nope absorption), hybrid dense/MoE FFN, expert bias
- GGUF loading entirely in Go (metadata parsing, weight loading to Metal VRAM)
- Unsupported architectures auto-filtered at scan time
- **Zero model-specific code**: tokenizer chat templates executed directly from GGUF `tokenizer.chat_template` Jinja2 via gonja; BOS/EOS IDs from GGUF metadata. No per-architecture tokenizer branches anywhere in the codebase.
- BPE tokenizer loaded from GGUF metadata (Qwen3.5/tiktoken-compatible, 248K vocab)
- Token sampling (greedy + top-p)
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
- **API**: OpenAI-compatible (`/api/v1/models`, `/api/v1/chat/completions`) with extensions: `"stateless"`, `"enable_thinking"`, `"model":"default"`. Legacy `/v1/*` routes also supported. Diagnostic file browser at `/diag/`.
- **Config**: `api_config.toml` — server, models, inference settings (`max_seq_len`, `enable_thinking_default`, `elide_thinking_default`, `log_thinking`)
- **Default listen**: `0.0.0.0:11116`

## Source Layout

```
Makefile                        Top-level: delegates to src/, symlinks bin/
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
    apiserver/                  HTTP handlers (OpenAI-compatible: /api/v1/*)
    chatclient/                 Interactive chat client implementation
    model/                      GGUF scanning + metadata
    archeditor/                 Arch editor web server
    inference/
      engine.go                 Generation loop (tokenize → forward → sample)
      metrics.go                InferenceMetrics (timing, throughput)
      sampler.go                Greedy and top-p sampling
      tokenizer.go              BPE tokenizer; chat template executed from GGUF via gonja (no model-specific code)
      arch/
        arch.go                 ArchDef, TOML parser, Validate() with builder contracts
        arch_util.go            Shared tensor-op helpers (rmsNormApply, projectReshape3D, attentionScale), cache key constants (CacheK, CacheV, ...)
        block_attention_util.go Shared attention helpers (scaledDotProductAttention, writeCacheKV, selectCachedKV, applyRoPEPair)
        params.go               Param resolver + routing expression eval
        weights.go              Weight resolver (template expansion, layer routing)
        blocks.go               BlockBuilder/FFNBuilder interfaces, registry
        block_attention.go      full_attention_gated (Qwen3.5)
        block_attention_std.go  full_attention (Llama)
        block_attention_mla.go  mla_attention (DeepSeek2/GLM-4)
        block_ssm.go            gated_delta_net (Qwen3.5 SSM)
        block_ffn.go            swiglu
        block_ffn_moe.go        moe_with_shared (MoE + expert bias)
        modules.go              Module, ModuleMap
        module_map.go           BuildModuleMap, BuildTensorDimsMap
        weight_store.go         WeightStore: immutable weight storage
        model.go                GenericModel: GGUF loading, per-layer FFN routing
        cache.go                Cache allocation (NewCache, GenericCache)
        graph.go                Forward pass (ForwardStateless, ForwardCached, runLayers)
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

## Key Technical Details

### TOML Model DSL

Model architectures are declared in TOML files (`models/arch/*.arch.toml`). The DSL captures:
- **Params**: GGUF metadata key mappings + derived arithmetic expressions
- **Layer routing**: expression-based rules determining which block type each layer uses. `@{name}` references builtins (`@{layer_idx}` = 0-based layer index), `${name}` dereferences a resolved GGUF param. Evaluated at model load time to determine the static layer structure.
- **Weight bindings**: GGUF tensor name templates with `blk.@{layer_idx}.` prefix expansion
- **Cache specs**: per-block-type cache tensor dimensions and dtypes
- **FFN type**: which feed-forward builder to use. Optional `[ffn_alt]` for per-layer FFN routing (e.g., dense SwiGLU for first N layers, MoE for the rest — auto-detected from GGUF weights).
- **Tokens**: `[tokens]` section declares think/no-think tokens per architecture.
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

### Shell Scripting Style
- Shebang: `#!/usr/bin/env bash`
- Indent: 2 spaces
- Variable dereferencing: always use curly braces (`${VAR}`, not `$VAR`)
- Test enclosure: `[[ ]]`, not `[ ]`
- Equality: `==`, not `=`
- define functions using keyword syntax `function funcname() {` not bare syntax `funcname() {`

### Deferred Work (priority order)
1. **Structured logging** — replace `fmt.Fprintf(os.Stderr, ...)` with a real logging package (slog or zerolog). Leveled output, consistent format, eliminate Makefile/test_inference.sh stderr redirects. Top tech debt priority.
2. **Chat client streaming + acontextual mode** — add `--no-history` flag for stateless per-prompt testing with real-time SSE streaming output. Replaces need for test_inference.sh for interactive debugging of thinking models. refactor chat client to separate implementation from CLI entry point
3. Batch inference
4. Multiple concurrent models
5. Linux/CUDA support (rename GPU init, add CUDA backend)
6. **Palette unification** — export a .css color set from the SVG diagram palette (`diagramPalette()`) that the arch-editor can use for block coloring, eliminating the duplicated color constants in editor.js
7. **Diffusion generation loop** — iterative masked denoising for non-causal models (LLaDA-MoE). Architecture definition at `models/arch/llada-moe.arch.toml.nyi`; builder support (attention QK-norm, non-causal mask, MoE FFN) already in place. Needs new generation strategy in engine.go.
