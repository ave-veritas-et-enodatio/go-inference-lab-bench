# Architecture — go-inference-lab-bench

Companion to `AGENTS.md`: codebase structure, data flows, and invariants.

---

## Central Invariant

**Zero model-specific Go code.** Every supported architecture is fully described by
`models/arch/<name>.arch.toml`. No Go code mentions "llama", "qwen", "deepseek" by name
except inside arch TOML files and tests. Adding a new model that uses existing block
builders requires only a new `.arch.toml`. Writing `if arch == "qwen35"` in Go is the
worst category of mistake in this codebase — that logic belongs in the TOML DSL.

---

## System Diagram

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
    ├─ culling/culling.go         ApplyCulling(tokenIDs, prompt) → clone ModuleMap → method → CullingMask
    ├─ ForwardCached / ForwardStateless
    │       │
    │       ▼
    │   arch/graph.go             build ggml graph per-layer
    │       ├─ MaskedLayer()      apply CullingMask (nil tensors for culled weights)
    │       ├─ BlockBuilders[il]  BuildStateless / BuildCached per layer
    │       └─ FFNBuilders[il]    BuildFFN per layer
    │           │
    │           ▼
    │       ggml/ops_graph.go     Go wrappers → C ggml ops (GPU)
    │
    ├─ sample()                   Greedy or Top-P
    └─ onToken callback           SSE streaming or buffered
```

---

## Package Map

### `bench/` — CLI entry point
Thin cobra command wrappers delegating to `internal/`. `serve-api`, `chat` expose `--log`/`--log-level` flags; batch commands (`gen-arch-diagram`, `gen-cull-metadata`) use default INFO-to-stderr logger.

### `internal/log/` — structured logging (leaf package)
Zero-external-dependency. Imports stdlib only — no project package may be imported by it. See **Logging** section.

### `internal/config/`
Loads `config/api_config.toml` into `Config`: `[server]`, `[models]`, `[inference]` subsections. Per-request JSON fields override server defaults for `cull_method`, `enable_thinking`, `elide_thinking`.

### `internal/model/`
- `gguf.go` — GGUF header scanning → `ModelInfo`
- `manager.go` — `ModelManager`: directory scan, filters architectures without matching `.arch.toml`
- `culling.go` — `CullingMap` (`[layer][head]float32` weight relevance; currently all-ones stub)

### `internal/apiserver/`
- `server.go` — `Server` holds `engines` map (`modelName → *inference.Engine`), `httpServer`, `pending` WaitGroup. No mutex on `engines` (single-client R&D).
- `completions.go` — POST `/api/v1/chat/completions`. Per-request overrides, SSE streaming, logprobs, timing/throughput in `usage` response.
- `models.go` — GET `/api/v1/models`.
- `ctl.go` — GET `/ctl/` (`?memstats`, `?quit`, `?quit&now`).

### `internal/inference/engine.go`

```go
type GenerateParams struct {
    MaxTokens       int
    Temperature     float32
    TopP            float32
    Stateless       bool   // bypass KV cache (for testing/comparison)
    CullMethod      string // "" or "none" = no culling; "random" = random test pattern
    ThinkingEnabled bool   // passed to template as enable_thinking variable
    LogProbs        bool   // include log-probabilities in response
    TopLogProbs     int    // number of top log-probabilities per token
}
```

`Generate()` flow:
1. `EncodeChat` → token IDs (template handles thinking mode natively via `enable_thinking`)
2. If culling active: `ApplyCulling` → `CullingMask` + `ModuleMap`
3. `generateCached` or `generateStateless` — each step: forward → sample → logprobs (if enabled) → onToken
4. Post-generation diagnostics if applicable (see `diagnostic.go`)
5. Write cull diagnostics if a cull map was produced

### `internal/inference/arch/`
Structural types (`ModuleMap`, `CullingMask`), TOML DSL parsing, graph
construction, forward pass, masking. See subsections below.

### `internal/inference/culling/`
Culling method dispatch, metadata loading, and per-method implementations. Imports `arch`
for structural types; `engine.go` calls `ApplyCulling` as the entry point.

### `internal/inference/ggml/`
Go wrappers for ggml C ops (~43 functions). All CGo for graph ops is isolated here.
See "CGo Layer" section. All CGo is confined to this package.

---

## TOML DSL (`ArchDef`)

An `*.arch.toml` file maps directly onto `arch.ArchDef`:

| Section | Go field | Purpose |
|---|---|---|
| `[architecture]` | `ArchMeta` | Name, tied_embeddings flag |
| `[params]` | `ParamsDef` | GGUF metadata key mappings; `[params.derived]` = arithmetic expressions; `[params.defaults]` = fallback values |
| `[weights.global]` | `WeightsDef` | Global tensor names: `token_embd`, `output_norm`, `output` |
| `[layers]` | `LayersDef` | `count` (param ref), `prefix` (must contain `@{layer_idx}`), `[layers.routing]`, `[layers.common_weights]` |
| `[layers.routing]` | `RoutingDef` | Binary routing: `rule` expression or `pattern` array param, `if_true`/`if_false` block names |
| `[layers.common_weights]` | `map[string]string` | Per-layer tensors shared by all block types (e.g. `attn_norm`) |
| `[blocks.<name>]` | `BlockDef` | Block-type weights, config, cache specs |
| `[ffn]` / `[ffn_alt]` | `FFNDef` | FFN builder + weights; `ffn_alt` for per-layer routing (dense vs MoE) |
| `[tokens]` | `TokensDef` | `think_open`, `think_close` token strings; `stop_tokens` string array — each entry added to the generation stop set alongside EOS |

**Routing:** `RoutingDef` supports two mutually exclusive modes:
- `rule` — expression evaluated per-layer (`@{layer_idx}`, `${param}` refs); nonzero → `if_true` block
- `pattern` — GGUF array param name; `pattern[layer_idx]` nonzero → `if_true` block

**`CacheDef.Shared`:** a string group name. All layers with the same `shared` value in a
given cache entry share one physical cache tensor (allocated on the first layer in the
group). Used for TOML-declared sharing; see also param-driven sharing via `n_kv_shared_layers`.

**`ArchMeta.EmbedScale`:** boolean. When true, input token embeddings are multiplied by
`sqrt(n_embd)` before the layer loop.

**Expressions:** `@{layer_idx}` = engine builtin; `${param}` = resolved GGUF param;
bare names = derived param refs; `tensor.ne[dim]` = tensor shape. Full spec including
operators, `?` optional suffix, and `[params.defaults]` semantics:
`models/arch/MODEL_ARCH_TOML_DSL_SPEC.md`. **Read it before writing or modifying any `.arch.toml`.**

**Validation** runs at parse time (`Validate()` in `arch.go`): unknown keys, missing
weights, bad param refs, and contract violations all produce errors with source line
numbers. A successful `arch.Load()` means the definition is structurally sound.

**Adding a new architecture:** Copy the closest existing `.arch.toml`. The validator
and builder contracts will identify exactly what is wrong.

---

## Block Builder System

Defined in `arch/blocks.go`:

```go
type BlockBuilder interface {
    Contract() BuilderContract           // declares expected weights, params, config
    BuildStateless(...) ggml.Tensor      // full-sequence forward pass
    BuildCached(ctx, gf, input, weights, params, config, inputs, cache) ggml.Tensor
}

type FFNBuilder interface {
    Contract() BuilderContract
    BuildFFN(...) ggml.Tensor
}
```

`BuildCached` receives a `*ggml.Graph` (`gf`) in addition to the graph context. Builders call `gf.BuildForwardExpand(ggml.Cpy(...))` to emit in-graph copy ops directly into KV cache tensors on the GPU — no CPU writeback is involved.

Registered in `init()`:
```
block builders: attention, full_attention_gated, mla_attention, gated_delta_net
FFN builders:   swiglu, geglu, moe
```

`attention` is the standard multi-head attention builder. Sliding-window attention is not
a separate builder — it is `attention` with `sliding_window = "true"` in block config,
which causes `selectMask` to route to `inputs.InpMaskSWA` instead of `inputs.InpMask`.

**`AttentionBuilder` contract:**
- Required weights: `attn_q`, `attn_output`
- Optional weights: `attn_k`, `attn_v`, `attn_q_norm`, `attn_k_norm`, `rope_freqs`
- Config schema (all optional): `head_dim`, `n_heads`, `n_kv_heads`, `rope_n_rot`,
  `rope_freq_base`, `kq_scale` — per-block overrides for attention dims; `sliding_window`
  — set to `"true"` to use SWA mask; `shared_kv_group` — group name for SharedKV;
  `v_norm` — `"rms"` applies RMS norm to V without learned weights; `rope` — `"standard"`
  or `"neox"` rope mode

`BuilderContract` declares `RequiredWeights`, `OptionalWeights`, `RequiredParams`, and
`ConfigSchema`. `Validate()` calls each builder's `Contract()` at TOML load time — the
primary protection against stale definitions.

**Adding a new block builder:**
1. Implement `BlockBuilder` (or `FFNBuilder`) with an explicit `Contract()`
2. Register in `init()` in `blocks.go`
3. Reference by name in TOML: `[blocks.<name>] builder = "my_builder"`

**NilTensor sentinel:** Culled weights are absent from the map `MaskedLayer()` returns.
Absent keys yield the zero-value `ggml.Tensor`, which satisfies `IsNil()`. Every builder
checks `IsNil()` and skips the op — culling is transparent to graph construction.

---

## Model Loading (`arch/model.go`)

`NewGenericModel(archName, ggufPath, archDir)` → `*GenericModel`:
1. Parse `.arch.toml` → `ArchDef`
2. Read GGUF KV metadata → `ResolvedParams`
3. Read GGUF tensor shapes → `ResolvedLayerWeights` per layer
4. Determine per-layer FFN routing (`ffn` vs `ffn_alt`)
5. Assign `BlockBuilders[]`, `FFNBuilders[]` per layer
6. Build `CanonicalModuleMap`
7. Load tensors into GPU VRAM
8. Create persistent `cachedCtx` and `cachedSched` for reuse across decode tokens
9. Allocate `ffnScratch` and `logitBuf` scratch buffers for the hot decode path

```go
type GenericModel struct {
    Def               *ArchDef
    Params            *ResolvedParams
    Weights           *ResolvedWeights       // GGUF tensor name maps per layer
    Store             *WeightStore           // GPU-resident tensors
    LayerBlockNames   []string               // which block builder each layer uses
    BlockBuilders     []BlockBuilder
    FFNBuilders       []FFNBuilder
    FFNConfigs        []map[string]any       // per-layer FFN config ([ffn.config] / [ffn_alt.config])
    CanonicalModuleMap *ModuleMap            // immutable; always Clone() before modifying
    HeadDim           int                    // attention head dimension
    TensorDims        TensorDimsMap          // tensor shape info for diagnostics
    ModelPath         string

    // Persistent compute resources for ForwardCached. Created at load, reused across tokens.
    cachedCtx   *ggml.GraphContext
    cachedSched *ggml.Sched

    // Pre-allocated scratch buffers for the hot decode path.
    ffnScratch map[string]ggml.Tensor // reused by buildFFNBlock each layer
    logitBuf   []float32              // reused by readLogits each token
}
```

---

## Cache (`arch/cache.go`)

`GenericModel.NewCache(maxSeqLen)` allocates per-layer cache tensors on GPU VRAM and pre-allocates `maskBuf` and `swaMaskBuf` (each sized to `maxSeqLen`) for reuse in `buildCausalMaskData` / `buildSWAMaskData` during cached decode. `ForwardStateless` allocates its own mask buffers and does not use these fields.

Two sharing mechanisms keep non-KV layers from allocating redundant buffers:

**TOML-declared sharing (`CacheDef.Shared`)**  
Set `shared = "groupName"` on a cache entry in a block definition. All layers whose block
type carries that entry with the same `shared` value reuse a single tensor, allocated on
the first layer in the group. `LayerCache.SharedGroup` is set to the group name for
diagnostic purposes.

**Param-driven sharing (`n_kv_shared_layers`)**  
When the GGUF param `n_kv_shared_layers` is present and nonzero, the last `N` layers
(indices `nLayers − N` through `nLayers − 1`) are treated as non-KV layers and reuse the
most recently allocated KV cache tensor for their block type. The last KV layer's
`LayerCache` is found by tracking `lastKVByBlock[blockName]` as allocation proceeds. The
non-KV layer's `LayerCache.SharedGroup` is set to the block name.

In both cases, non-KV layers hold a reference to the same `ggml.Tensor` as the KV layer.
During the forward pass, `selectSharedKV` reads from this shared cache tensor rather than
a per-layer buffer.

---

## Forward Pass (`arch/graph.go`)

Entry points:
- `ForwardStateless(tokenIDs, mask, caps) ([]float32, *EngagementData, error)` — full recompute; returns logits, per-layer engagement data, and error
- `ForwardCached(gc, tokenIDs, mask) ([]float32, error)` — KV-cached; returns logits and error

Both follow the same frame:
1. Extract params; allocate graph context + GPU scheduler
2. Build input tensors (`inpTokens`, `inpPos`, `inpMask`)
3. Embedding scaling: if `ArchMeta.EmbedScale`, multiply token embeddings by `sqrt(n_embd)`
4. SWA mask: if `sliding_window` param present and nonzero, allocate `inpMaskSWA` and set it on `GraphInputs`
5. Per-layer embedding setup: `buildPerLayerEmbedSetup` prepares the combined per-layer embedding tensor (returns NilTensor if unused)
6. `runLayers`: for each layer, `MaskedLayer()` → `BlockBuilders[il].Build*()` → optional `attn_post_norm` → residual add → `buildFFNBlock()` (includes optional `ffn_post_norm`) → optional per-layer embedding injection → optional `layer_output_scale` multiply; engagement cosine similarities computed at each step when not nil
7. Final norm + LM head (`buildLogits`); execute (`sched.Compute`); read back logits + any captures + engagement scalars

Stateless vs cached diverge in input setup, KV cache writeback strategy, and position tracking.

**`ForwardCached` reuse model:** `ForwardCached` calls `cachedCtx.Reset()` and `cachedSched.Reset()` at the top of each call and reuses the persistent `cachedCtx`/`cachedSched` instances created at model load — no scheduler or graph context is allocated per token. KV writes (K, V, conv_state, ssm_state, MLA K) are emitted as `ggml_cpy` ops directly into the persistent cache buffer via `writeCacheKV` / `gf.BuildForwardExpand(ggml.Cpy(...))` and execute on the GPU during `sched.Compute()`. There is no post-compute CPU writeback loop.

### `runLayers` — per-layer features

```
for each layer il:
    lt = MaskedLayer(il)
    if !attn_norm.IsNil():
        xPreBlock = x
        cur = rmsNormApply(attn_norm) → BlockBuilder.Build*()
        if !attn_post_norm.IsNil(): cur = rmsNormApply(attn_post_norm)
        x += cur
        if engagement != nil: engagement.blockTensors[il] = cosineSim(xPreBlock, x)
    xPreFFN = x
    x = buildFFNBlock(x)     // includes ffn_post_norm if present
    if engagement != nil && x != xPreFFN: engagement.ffnTensors[il] = cosineSim(xPreFFN, x)
    if perLayerEmbd present and !pe_inp_gate.IsNil():
        x = perLayerEmbedInject(x)
    if !layer_output_scale.IsNil():
        x *= layer_output_scale
```

`perLayerEmbedInject` slices this layer's embedding from the combined tensor prepared by
`buildPerLayerEmbedSetup`, gates it through `pe_inp_gate` (GELU), optionally applies
`pe_post_norm`, and adds the result to the residual stream.

### `buildLogits` — logit softcapping

After the final norm and LM head projection, if the `logit_softcapping` param is present
and nonzero, logits are capped:

```
logits = logits * (1/cap) → tanh → * cap
```

This is `cap * tanh(logits / cap)`, implemented as three sequential `ggml.Scale` /
`ggml.Tanh` ops.

### `SharedKVState` — in-graph K/V sharing

`SharedKVState` is allocated fresh on each forward pass and passed through `GraphInputs`.
It carries in-graph tensors (not weights) from KV layers to non-KV layers within the same
`runLayers` call:

1. KV layers (`attn_k` present): after projecting K and V, write both into
   `SharedKV.K[group]` and `SharedKV.V[group]` keyed by the block's `shared_kv_group`
   config value.
2. Non-KV layers (`attn_k` absent): read `SharedKV.K[group]` and `SharedKV.V[group]`
   for attention. In cached mode, `selectSharedKV` additionally reads from the shared
   `LayerCache` tensor rather than a per-layer KV buffer.

`SharedKVState` tensors are live only for the duration of the graph build. The cache
backing for non-KV layers is handled separately — see "Cache" section below.

### SWA mask

`buildSWAMaskData(nQuery, nKV, startPos, window)` produces a float32 mask where positions
outside the sliding window (key > query or key < query − window) are set to −Inf. The
mask is allocated as `inpMaskSWA` in `GraphInputs` only when the `sliding_window` GGUF
param is present and nonzero; otherwise `InpMaskSWA` is a NilTensor. `selectMask` in
`AttentionBuilder` chooses between `InpMask` and `InpMaskSWA` based on the block's
`sliding_window` config key.

### `ForwardCaptures` — optional tensor capture

`ForwardStateless` accepts `*ForwardCaptures` (nil = no capture):

```go
type CaptureFlags uint32
const CaptureAttnWeights CaptureFlags = 1 << iota

type ForwardCaptures struct {
    Flags       CaptureFlags
    AttnWeights [][]float32  // [n_layers][nHeads*nTokens*nKV], nKV fastest
    NHeads      int64
    NTokens     int64
}
```

When `CaptureAttnWeights` is set, `scaledDotProductAttention` marks the post-softmax
tensor as a graph output; data is read back after `sched.Compute`. Layers without
`scaledDotProductAttention` (MLA, delta-net) leave their slot nil.

**Capture is stateless-only by design.** `ForwardCached` has no capture parameter —
cached mode has no single clean matrix to extract across prefill + per-token decode calls.
Data collection and KV-cache optimization must not be entangled. Any research question
answerable via a stateless pass should use that path. Do not add capture to
`ForwardCached` without explicit research into cached-mode mechanics as a dedicated goal.

### `engagement.go` — residual stream cosine similarity

`EngagementData` records per-layer cosine similarity of the residual stream before and after the attention block (`BlockCosSim`) and FFN (`FFNCosSim`). Unconditionally populated by `ForwardStateless` — overhead is ~10 scalar ggml ops per layer with ~200 scalar GPU→CPU readbacks after compute. Culled/skipped layers get NaN. Results are returned alongside logits from `ForwardStateless`.

```go
type EngagementData struct {
    BlockCosSim []float32   // [nLayers]; NaN if block culled/skipped
    FFNCosSim   []float32   // [nLayers]; NaN if FFN culled/skipped
}
```

### `diagnostic.go` — post-generation analysis

Contains runners that execute additional forward passes after generation and populate
`InferenceMetrics.Diagnostic`. Triggered conditionally from `engine.Generate()`.
Add new analysis routines here rather than to `engine.go`.

---

## Culling Architecture

Two packages: `culling/` (algorithms, dispatch, metadata) and `arch/` (structural types,
mask compilation, forward-pass masking).

### Structural types (`arch/`)
- `ModuleMap` — canonical map built at load; cloned per-request
- `CullingMask` — compiled mask: `ZeroTensors` (whole-tensor zeroing)
- `Compile()` — annotated `ModuleMap` → `CullingMask`
- `MaskedLayer()` — applies mask at graph-build time: culled tensors absent from map → NilTensor

### Culling algorithms (`culling/`)
```
culling.ApplyCulling(canonical, method, tokenIDs, prompt, meta)
    ├─ Clone() → mutable ModuleMap
    ├─ apply method (random, inattention)
    └─ Compile() → CullingMask
```

Methods annotate the cloned map (mark modules culled, append to
`mm.CullLog`). `Compile()` converts to `CullingMask` for the forward pass.

`CullingMeta` is loaded at engine startup from `.cullmeta` sidecars next to the GGUF.
`LoadCullingMeta` tries each format loader in turn. Returns `(nil, nil)` if absent — callers must handle nil.

### Culling metadata generation
`bench gen-cull-metadata --cull-method <method> <model.gguf>` writes a `.cullmeta`
sidecar. CPU and GPU paths are both implemented (GPU primary, `--cpu` fallback).
See AGENTS.md: CPU path first, GPU second, never GPU-only.

### Culling diagnostics
`WriteCullDiagnostics()` writes to `<exeDir>/diag/`:
- `<model>.<timestamp>.cullmap.toml` — serialized ModuleMap
- `<model>.<timestamp>.cullmap.svg` — visual diagram (`RenderModuleMapDiagram`)
- `<model>.cullmap.toml` / `.svg` — "latest" symlinks for browser refresh

---

## CGo Layer (`internal/inference/ggml/`)

All CGo is confined here. `ggml_lib/` is a thin C wrapper over ggml ops — no model-specific
logic, no C++.

**ggml log routing (`ggml/logging.go`):**  
`ggml_log_set` registers a CGo callback in the C layer (`ggml_ops.c`). All ggml diagnostic
output — previously printed directly to stderr — is intercepted and forwarded to the Go
`internal/log` logger at the appropriate level. This eliminates uncontrolled stderr writes
from the C layer; all log output passes through the single Go logger and is subject to
the same level gate and file sink. Registration is not automatic on import — each CLI
entry point that loads models must call `ggmlmod.InitLogging()` explicitly after
`log.InitLogger`. `logging.go` is the only file in `ggml/` that imports `internal/log`;
the rest of the package has no dependency on it.

**Go wrappers added for Gemma4 support (in `ggml/`):**
- `Gelu(ctx, a)` — element-wise GELU activation (used by GeGLU and per-layer embed injection)
- `Tanh(ctx, a)` — element-wise tanh (used by logit softcapping)
- `MulMatSetPrecF32(t)` — sets F32 accumulation precision on a `mul_mat` tensor (no return value; mutates in place)

**GGUF metadata reading**: `arch/model.go` uses pure-Go `gguf-parser-go` — no CGo. The `goGGUFReader` adapter implements the `GGUFReader` interface with type-checked KV access (guards against gguf-parser-go's panic-on-wrong-type behavior). Split GGUF files are rejected at parse time.

**Verified ggml semantics** (do not change without re-verifying):
- `ggml_mul_mat(A, B)`: contracts over `A→ne[0] == B→ne[0]`; result `[A→ne[1], B→ne[1], ...]`; GQA broadcasting when `B→ne[2] % A→ne[2] == 0`
- `ggml_permute(ctx, a, ax0..ax3)`: input dim i → output position ax_i
- `ggml_rms_norm`: normalizes over `ne[0]` per slice
- `ggml_soft_max_ext(a, mask, scale, max_bias)`: `softmax(scale * a + mask)` over `ne[0]`

---

## Logging

See **Internal Logging** section in `AGENTS.md` — authoritative for format, initialization, CLI flags, ggml routing, and constraints. Summary: `internal/log` package, `<HH:MM:SS>[LEVEL]` format, dual stderr/file output, `ggml.InitLogging()` routes C library output. Dependency leaf — imports stdlib only.

---

## Visualization System

Two SVG renderers share a unified palette (`diagramPalette()` in `arch/diagram_util.go`).
To change any color, update the palette map — both renderers reflect it. No inline hex strings.

- `RenderArchDiagram(def, outPath)` — `*.arch.svg` from `ArchDef`; invoked by `bench gen-arch-diagram`
- `RenderModuleMapDiagram(mm, svgPath, tensorDims)` — per-layer tensor detail with cull overlays; called from `WriteCullDiagnostics`

---

## Weight Store (`arch/weight_store.go`)

`WeightStore` holds GPU-resident tensors, immutable after construction. Logical names
(e.g. `attn_q`, `ffn_gate`) are short names from TOML, not GGUF tensor names; translation
is via `ResolvedLayerWeights` in `arch/weights.go`.
- `Global(name) ggml.Tensor`
- `Layer(idx) map[string]ggml.Tensor`

---

## Key Invariants

1. **No model-specific Go code.** Architecture branches belong in TOML.
2. **`CanonicalModuleMap` is never mutated.** Always `Clone()` before modification.
3. **NilTensor means culled.** Absent map keys return zero-value `ggml.Tensor`. Every builder checks `IsNil()`.
4. **`os.Exit` never below CLI entry point.** Return errors up the chain.
5. **CGo stays in `ggml/`.** No other package imports `"C"`. GGUF metadata is read via pure-Go `gguf-parser-go`.
6. **Builder contracts enforced at parse time.** Adding a required weight to `Contract()` requires updating all `.arch.toml` files using that builder.
7. **Palette is single-source.** All diagram colors from `diagramPalette()`.
8. **TOML is the data language.** JSON only for OpenAI-compatible API payloads.
9. **`internal/log` is a dependency leaf.** Imports stdlib only.
10. **Utility placement:** project-wide → `util/project_util.go`; package-internal → `<pkg>/<pkg>_util.go`; single-file → same file.
11. **Test with all models.** Llama is easy; Qwen3.5, Qwen3.5-MoE, DeepSeek2, Gemma4 edge-case.
12. **Arch editor: TOML is canonical, palette/builders dynamic.** No hardcoded colors/names in JS.
13. **Data capture is stateless-only.** `ForwardCaptures` on `ForwardStateless` only.
14. **Template owns thinking mode.** `enable_thinking` passed to gonja template. No post-render manipulation.
15. **SharedKV ordering validated at parse time.** `shared_kv_group` with no `attn_k` producer is a validation error.

## Metrics and Timing

`InferenceMetrics` captures timing/throughput for each generation, returned by `engine.Generate()` and serialized in the API `usage` field:

```go
type InferenceMetrics struct {
    PromptTokens     int                  // input tokens
    CompletionTokens int                  // output tokens
    FinishReason     string               // "stop" or "length"
    PrefillDuration  time.Duration        // prefill wall-clock
    DecodeDuration   time.Duration        // all decode steps wall-clock
    TotalDuration    time.Duration        // total generation wall-clock
    ZeroedTensors    int                  // tensors zeroed by culling
    TotalTensors     int                  // total weight tensors
    Diagnostic       *InferenceDiagnostic // nil unless inattention culling active
    TokenLogProbs    []TokenLogProb       // nil if not requested
    Engagement       *arch.EngagementData  // stateless only
}
```

Throughput methods: `TokensPerSec()` (decode), `PrefillTokensPerSec()`, `TotalTokensPerSec()`, `CullRatio()`. All return 0 when denominator ≤ 0.

Non-streaming response `usage` object (all timing fields `omitempty`):
```json
{
  "usage": {
    "prompt_tokens": 12, "completion_tokens": 48, "total_tokens": 60,
    "prompt_tokens_per_sec": 240.5, "completion_tokens_per_sec": 35.2, "total_tokens_per_sec": 42.1,
    "prefill_seconds": 0.049, "decode_seconds": 1.374, "total_seconds": 1.423
  }
}
```
Streaming responses do not include usage (OpenAI convention — use `stream_options.include_usage` if supported).

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

`test_equiv.sh` sends identical prompts to bench and `llama-server` (Homebrew),
compares top-1 logprobs. All models match within GPU floating-point variance (~0.1%
relative error on logprobs). Validates: tokenization, chat template rendering, forward
pass correctness, and sampling.
