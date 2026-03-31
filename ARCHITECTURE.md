# Architecture — go-inference-lab-bench

Companion to `AGENTS.md`. This document describes the codebase structure, data flows,
and invariants an agent needs to navigate and modify the project safely.

---

## Central Invariant

**Zero model-specific Go code.** Every supported model architecture is fully described by
a `models/arch/<name>.arch.toml` file. No Go code mentions "llama", "qwen", "deepseek"
by name except inside arch TOML files and tests. Adding a new model architecture that uses
existing block builders requires only a new `.arch.toml` — no Go changes.

Violating this invariant is the worst category of mistake in this codebase. When you see
yourself writing `if arch == "qwen35"` in Go, stop — that logic belongs in the TOML DSL.

---

## System Diagram (text)

```
HTTP request
    │
    ▼
apiserver/completions.go          parse JSON, extract GenerateParams
    │
    ▼
apiserver/server.go (Engine())    look up loaded model → *inference.Engine
    │
    ▼
inference/engine.go (Generate())
    ├─ tokenizer.go               EncodeChat (Jinja2 chat template from GGUF, via gonja)
    ├─ ForwardCached / ForwardStateless
    │       │
    │       ▼
    │   arch/graph.go             build ggml graph per-layer
    │       ├─ BlockBuilders[il]  BuildStateless / BuildCached per layer
    │       └─ FFNBuilders[il]    BuildFFN per layer
    │           │
    │           ▼
    │       ggml/ops_graph.go     Go wrappers → C ggml ops (Metal)
    │
    ├─ sample()                   Greedy or Top-P
    └─ onToken callback           SSE streaming or buffered
```

---

## Package Map

### `bench/` — CLI entry point
Thin cobra command wrappers. Each subcommand delegates immediately to `internal/`.
No business logic lives here. `serve_api.go` builds config and calls `apiserver.NewServer`.

### `internal/config/`
Loads `api_config.toml` into `Config`.

### `internal/model/`
- `gguf.go` — GGUF file scanning and metadata extraction. Reads GGUF header KV pairs
  to produce `ModelInfo` (architecture name, layer count, etc.). Zero inference logic.
- `manager.go` — `ModelManager`: directory scan, `ModelInfo` list, filters unsupported
  architectures (those without a matching `.arch.toml`).

### `internal/apiserver/`
- `server.go` — `Server` holds the loaded `engines` map (`modelName → *inference.Engine`),
  `httpServer` for graceful shutdown, and `pending` WaitGroup for in-flight request tracking.
  No mutex on `engines` currently (single-client R&D use case). `Engine()` method looks up
  by model name; `"default"` resolves to the configured default model.
- `completions.go` — POST `/api/v1/chat/completions`. Builds `GenerateParams` from JSON body
  (pointer semantics: nil = use server config). SSE streaming via `onToken` callback.
  Logprobs: `logprobs: true` + `top_logprobs: N` in request; response includes
  `choices[].logprobs.content[].top_logprobs[]` per OpenAI spec. Logs `[req] param overrides: ...`
  when any field differs from server config.
- `models.go` — GET `/api/v1/models`. Lists loaded engines.
- `ctl.go` — GET `/ctl`. Control endpoint: `?quit` waits for in-flight inference then shuts down;
  `?quit&now` shuts down immediately (100ms timeout).

### `internal/inference/engine.go`
The main generation loop. Key types:

```go
type GenerateParams struct {
    MaxTokens       int
    Temperature     float32
    TopP            float32
    Stateless       bool   // bypass KV cache
    ThinkingEnabled bool   // passed to template as enable_thinking variable
    LogProbs        bool   // include log-probabilities in response
    TopLogProbs     int    // number of top log-probabilities per token
}
```

`Generate()` flow:
1. `EncodeChat` → token IDs (template handles thinking mode natively via `enable_thinking`)
2. Run `generateCached` or `generateStateless` — each step: forward → sample → logprobs (if enabled) → onToken

### `internal/inference/arch/` — Core inference package

This is where the bulk of the complexity lives. Structural types (ModuleMap), TOML DSL parsing, graph construction, forward pass. Subsections below.

### `internal/inference/ggml/`
Go wrappers for ggml C ops. ~36 functions. All CGo is isolated to this package.
See "CGo Layer" section. Never call CGo from outside this package.

---

## TOML DSL (`ArchDef`)

An `*.arch.toml` file maps directly onto `arch.ArchDef`. Sections:

| Section | Go field | Purpose |
|---|---|---|
| `[architecture]` | `ArchMeta` | Name, tied_embeddings flag |
| `[params]` | `ParamsDef` | GGUF metadata key mappings; `[params.derived]` = arithmetic expressions; `[params.defaults]` = fallback values |
| `[weights.global]` | `WeightsDef` | Global tensor names: `token_embd`, `output_norm`, `output` |
| `[layers]` | `LayersDef` | `count` (param ref), `prefix` (must contain `@{layer_idx}`), `[layers.routing]`, `[layers.common_weights]` |
| `[layers.routing]` | `RoutingDef` | Binary routing: `rule` (Go expression), `if_true`/`if_false` (block names) |
| `[layers.common_weights]` | `map[string]string` | Per-layer tensors shared by all block types (e.g. `attn_norm`) |
| `[blocks.<name>]` | `BlockDef` | Block-type weights, config, cache specs — one section per block type |
| `[ffn]` / `[ffn_alt]` | `FFNDef` | FFN builder + weights. `ffn_alt` for per-layer routing (dense vs MoE) |
| `[tokens]` | `TokensDef` | `think_open`, `think_close` token strings |

**Expression syntax (brief):** `@{layer_idx}` is the engine-provided layer index builtin;
`${param}` dereferences a resolved param in routing rules; bare names reference params in
`derived` expressions; `tensor.ne[dim]` reads a tensor shape dimension.
Full expression language spec — including arithmetic operators, optional `?` suffix,
array-to-scalar promotion, and `[params.defaults]` semantics — is in
`models/arch/model_arch_toml_dsl_spec.md`, which also specifies each builder's params, weights, config
keys, cache layout, and step-by-step forward pass algorithm. **Read `model_arch_toml_dsl_spec.md` for the
complete DSL picture before writing or modifying any `.arch.toml` file.**

**Validation happens at parse time** (`Validate()` in `arch.go`). Unknown keys, missing required
weights, bad param references, and contract violations all produce errors with source line numbers
(`ResolveErrorLines`). If `arch.Load()` succeeds, the definition is structurally sound.

**Adding a new architecture:** Copy the closest existing `.arch.toml`. The validator will tell
you exactly what is wrong. The builder contracts (see below) will catch weight mismatches.

---

## Block Builder System

Defined in `arch/blocks.go`. Two interfaces:

```go
type BlockBuilder interface {
    Contract() BuilderContract           // declares expected weights, params, config
    BuildStateless(...) ggml.Tensor      // full-sequence forward pass
    BuildCached(...) ggml.Tensor         // KV-cache decode step
}

type FFNBuilder interface {
    Contract() BuilderContract
    BuildFFN(...) ggml.Tensor
}
```

Registered in `init()`:
```
block builders: attention, full_attention_gated, mla_attention, gated_delta_net
FFN builders:   swiglu, moe_with_shared
```

`BuilderContract` lists `RequiredWeights`, `OptionalWeights`, `RequiredParams`, and a
`ConfigSchema`. `Validate()` in `arch.go` calls each builder's `Contract()` at TOML load time
and reports mismatches. This is the primary protection against stale TOML definitions.

**Adding a new block builder:**
1. Write `type MyBuilder struct{}` implementing `BlockBuilder` (or `FFNBuilder`)
2. Implement `Contract()` — be explicit about required weights
3. Register in `init()` in `blocks.go`
4. Reference by name in the TOML `[blocks.<name>] builder = "my_builder"`

---

## Model Loading (`arch/model.go`)

`NewGenericModel(archName, ggufPath, archDir)` → `*GenericModel`:

1. Load and parse `.arch.toml` → `ArchDef`
2. Open GGUF file for KV metadata → resolve all params to `ResolvedParams`
3. Open GGUF file again for tensor shapes → `ResolvedLayerWeights` per layer
4. Determine per-layer FFN routing (`ffn` vs `ffn_alt`)
5. Assign block builder and FFN builder per layer → `BlockBuilders[]`, `FFNBuilders[]`
6. Build `CanonicalModuleMap` (structural map of all weight modules)
7. Load all tensors into Metal VRAM via ggml backend

`GenericModel` key fields:
```go
type GenericModel struct {
    Def              *ArchDef
    Params           *ResolvedParams
    Weights          ResolvedWeights          // GGUF tensor name maps per layer
    Store            *WeightStore             // GPU-resident tensors
    BlockBuilders    []BlockBuilder           // per-layer block builder
    FFNBuilders      []FFNBuilder             // per-layer FFN builder
    CanonicalModuleMap *ModuleMap             // immutable structural map (never mutated)
    TensorDims       TensorDimsMap            // tensor shape info for diagnostics
    ModelPath        string                   // path to GGUF file
}
```

`CanonicalModuleMap` is built once at load time and **never mutated**.

---

## Forward Pass (`arch/graph.go`)

Two entry points: `ForwardStateless(tokenIDs)` and `ForwardCached(cache, tokenIDs)`.

Both follow the same frame:
1. Extract params (`nLayers`, `nEmbd`, `nVocab`, `rmsEps`)
2. Allocate graph context + Metal scheduler
3. Build input tensors (`inpTokens`, `inpPos`, `inpMask`)
4. Per-layer loop:
   - `BlockBuilders[il].Build{Stateless,Cached}()` → attention/SSM output
   - `buildFFNBlock()` → FFN output + residual
5. Final norm + LM head + logit extraction
6. Execute graph (`sched.Compute`)
7. Read back logits → `[]float32`

The divergence between stateless and cached is: input tensor setup, KV cache writeback,
and cache-position tracking. Everything else is shared structure.

---

## CGo Layer (`internal/inference/ggml/`)

All CGo is confined to this package. The C layer (`ggml_lib/`) is a thin wrapper over ggml
ops — no model-specific logic, no C++.

**Verified ggml semantics** (do not change without re-verifying):
- `ggml_mul_mat(A, B)`: contracts over `A→ne[0] == B→ne[0]`. Result `[A→ne[1], B→ne[1], ...]`.
  GQA broadcasting: `B→ne[2] % A→ne[2] == 0`.
- `ggml_permute(ctx, a, ax0, ax1, ax2, ax3)`: input dim i → output position ax_i.
- `ggml_rms_norm`: normalizes over `ne[0]` independently per slice.
- `ggml_soft_max_ext(a, mask, scale, max_bias)`: `softmax(scale * a + mask)` over `ne[0]`.

---

## Visualization System

Two SVG renderers sharing a unified palette.

### Palette (`arch/diagram_util.go`)
`diagramPalette()` returns the single canonical `map[string]string` for all colors.
Both renderers call this. To change any color, update `diagramPalette()` only —
both diagrams reflect the change. Do not hardcode color strings in renderer code.

### Architecture diagram (`arch/arch_diagram.go`)
`RenderArchDiagram(def, outPath)` — generates `*.arch.svg` from a parsed `ArchDef`.
Shows the layer structure, block types, weight modules, and cache specs.
Invoked by `bench gen-arch-diagram`.

### Module map diagram (`arch/module_map_diagram.go`)
`RenderModuleMapDiagram(mm, svgPath, tensorDims)` — generates a per-layer tensor detail
diagram.
`tensorDims` provides shape info for rendering tensor size annotations.

Both renderers use `svgPath` as output; `module_map_diagram.go` normalizes the `.svg`
extension if not already present.

---

## Weight Store (`arch/weight_store.go`)

`WeightStore` holds GPU-resident tensors. Immutable after construction — no tensor is
ever added or removed from the store at inference time. The store provides:
- `Global(name) ggml.Tensor` — global tensors by logical name
- `Layer(idx) map[string]ggml.Tensor` — per-layer tensor map by logical name

Logical names are the short names from the TOML definition (e.g. `attn_q`, `ffn_gate`),
not GGUF tensor names. The translation layer is `ResolvedLayerWeights` in `arch/weights.go`.

---

## Key Invariants for Agents

1. **No model-specific Go code.** Any branch on architecture name belongs in TOML.

2. **`CanonicalModuleMap` is never mutated.**

4. **`os.Exit` is never acceptable below the CLI entry point.** Return errors up the chain.

5. **CGo stays in `ggml/`.** No CGo imports outside `internal/inference/ggml/`.

6. **Builder contracts are enforced at TOML parse time.** If you add a required weight to
   a builder's `Contract()`, all existing arch TOML files that use that builder must be
   updated — and the validator will tell you exactly which ones fail.

7. **Palette is single-source.** All colors come from `diagramPalette()`. No inline hex strings.

8. **TOML is the data language.** JSON only for OpenAI-compatible API payloads. Config files,
   architecture definitions, and module maps are all TOML.

9. **Utility placement rules** (from AGENTS.md):
   - Used in ≥2 files, project-wide → `internal/util/project_util.go`
   - Used in ≥2 files, package-internal → `internal/<pkg>/<pkg>_util.go`
   - Single-file use → define in that file

10. **Test with all models.** Llama is easy. Qwen3.5, Qwen3.5-MoE, and DeepSeek2/GLM-4
    are where architectural edge cases surface. Always run `ALL_MODELS=true` before
    declaring a change complete.

11. **Template owns thinking mode.** `enable_thinking` is passed as a variable to the
    gonja template. No post-render prompt manipulation (no `/no_think` injection, no
    think-block stripping). The template's output is the correct prompt. This matches
    llama.cpp behavior and is required for logprob equivalence testing.

---

## Logprobs

`ComputeTopLogProbs` in `sampler.go` computes log-probabilities via stable log-softmax
on raw logits after each `sample()` call. Works in both cached and stateless modes.

```go
type TokenLogProb struct {
    ID       int32
    Token    string
    LogProb  float64
    Bytes    ByteArray  // JSON-marshals as integer array, not base64
    TopProbs []TopLogProb
}
```

`ByteArray` is a custom `[]byte` type that marshals as `[51, 52]` instead of Go's
default base64 encoding — matches the OpenAI/llama.cpp response format.

### Equivalence Testing

`make equiv-test` (via `test_llama_equiv.sh`) sends identical prompts to bench and
`llama-server` (Homebrew), compares top-1 logprobs. All models match within Metal
floating-point variance (~0.1% relative error on logprobs). Validates: tokenization,
chat template rendering, forward pass correctness, and sampling.

---

## Open Architecture Issues

(none)
