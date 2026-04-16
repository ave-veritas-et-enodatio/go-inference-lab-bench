# Architecture — inference-lab-bench

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
    ├─ generateCached / generateStateless / generateDiffusion
    │       │
    │       ▼
    │   arch/graph.go             ForwardCached / ForwardStateless / ForwardStatelessAllLogits → build ggml graph per-layer
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
- `manager.go` — `ModelManager`: two-pass directory scan — `*.gguf` glob first, then `*.st/` directory enumeration via `os.ReadDir`; `ModelFormat` type (`FormatGGUF` / `FormatSafetensors`); `ModelInfo.Format` field; `List()` rescans for new files and `.st/` dirs; `Get()`/`tryLoadOne()` checks `.gguf` before falling back to `.st/`; filters architectures without matching `.arch.toml`
- `gguf.go` — GGUF header scanning → `GGUFMetadata`
- `safetensors.go` — `ParseSafetensorsDir()`: reads safetensors index for tensor inventory, `config.json` for architecture class and numeric params, resolves architecture via `arch.FindSTMapByHFClass()`; returns "unknown" architecture and zeroed numeric fields if `config.json` is absent; corrupt `config.json` logs debug warning and continues with partial data
- `culling.go` — `CullingMap` (`[layer][head]float32` weight relevance; currently all-ones stub)

### `internal/apiserver/`
- `server.go` — `Server` holds `engines` map (`modelName → *inference.Engine`) guarded by `enginesMu sync.Mutex`, `httpServer`, `pending` WaitGroup, and an injected `util.BenchPaths`. The mutex is held while looking up, evicting, or creating an engine — single-client R&D, but the lock keeps the eviction/creation race honest. `NewServer(paths, cfg, manager)` takes `BenchPaths` from the cobra entry point — `apiserver` never calls `util.ResolvePaths()` itself.
- `completions.go` — POST `/api/v1/chat/completions`. Per-request overrides, SSE streaming, logprobs, timing/throughput in `usage` response.
- `models.go` — GET `/api/v1/models`.
- `ctl.go` — GET `/ctl/` (`?memstats`, `?quit`, `?quit&now`).

### `internal/inference/engine.go`

```go
type DiffusionParams struct {
    Steps       int    // 0 = use default (64)
    BlockLength int    // 0 = single block (global); >0 = block-based left-to-right
    Algorithm   string // "" or "confidence" = max-softmax; stub for future variants
}

type GenerateParams struct {
    MaxTokens       int
    Temperature     float32
    TopP            float32
    Stateless       bool             // bypass KV cache (for testing/comparison)
    CullMethod      string           // "" or "none" = no culling; "random" = random test pattern
    ThinkingEnabled bool             // passed to template as enable_thinking variable
    LogProbs        bool             // include log-probabilities in response
    TopLogProbs     int              // number of top log-probabilities per token
    FlashAttention  *bool            // nil = use server default; true/false = per-request override
    Diffusion       *DiffusionParams // nil = not diffusion (ignored for autoregressive models)
}
```

`Generate()` flow:
1. `EncodeChat` → token IDs (template handles thinking mode natively via `enable_thinking`)
2. If culling active: `ApplyCulling` → `CullingMask` + `ModuleMap`
3. Dispatch on generation strategy:
   - `e.IsDiffusion()` → `generateDiffusion` (block-based iterative masked denoising)
   - `params.Stateless` → `generateStateless` (full-sequence recompute per token)
   - default → `generateCached` (KV/SSM-cached autoregressive decode)
4. Post-generation diagnostics if applicable (see `diagnostic.go`)
5. Write cull diagnostics if a cull map was produced

### Generation strategies

Three generation paths live in separate files under `internal/inference/`:

| File | Strategy | Forward call | Notes |
|---|---|---|---|
| `generate_cached.go` | Autoregressive + KV cache | `ForwardCached` | Prefill in one pass; decode one token at a time |
| `generate_stateless.go` | Autoregressive, no cache | `ForwardStateless` | Full recompute each step; supports `ForwardCaptures` |
| `generate_diffusion.go` | Diffusion (iterative denoising) | `ForwardStatelessAllLogits` | Block-based masked denoising; no prefill/decode split |

`Engine.Generate()` dispatches via `IsDiffusion()` first; stateless flag is consulted only on the autoregressive path.

**Diffusion generation** (`generateDiffusion`):
1. Initialize output positions with `mask_token_id` (read from GGUF via `MaskTokenID()`).
2. Divide output into left-to-right blocks of size `DiffusionParams.BlockLength` (0 = single global block).
3. For each block: run `DiffusionParams.Steps` denoising steps. Each step calls `ForwardStatelessAllLogits` on the full sequence (prompt + all output positions), extracts logits for masked positions in the current block, scores each position by max-softmax confidence, and unmasks the highest-confidence positions according to a linear schedule.
4. Emit output tokens in sequence order after all blocks are resolved, stopping at the first stop token.

Streaming requests on diffusion models produce a single burst response after all denoising steps complete (not token-by-token); a warning is logged.

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
| `[architecture]` | `ArchMeta` | Name, `tied_embeddings`, `embed_scale`, `non_causal`, `generation`, `shift_logits` — see **`ArchMeta` flags** below |
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

**`ArchMeta` flags:**
- `EmbedScale` (`embed_scale`) — when true, input token embeddings are multiplied by `sqrt(n_embd)` before the layer loop.
- `NonCausal` (`non_causal`) — when true, attention uses a zero mask (all positions attend to all positions) instead of a causal lower-triangular mask. Required for diffusion models.
- `Generation` (`generation`) — `""` (default, autoregressive) or `"diffusion"`. Controls which generation path `Engine.Generate()` dispatches to via `IsDiffusion()`. Setting `generation = "diffusion"` without `non_causal = true` is a validation error.
- `ShiftLogits` (`shift_logits`) — diffusion only: output position `p` reads logits at index `p-1` rather than `p`. Required for models whose output tensor is offset by one position relative to the input.

**Expressions:** `@{layer_idx}` = engine builtin; `${param}` = resolved GGUF param;
bare names = derived param refs; `tensor.ne[dim]` = tensor shape. Full spec including
operators, `?` optional suffix, and `[params.defaults]` semantics:
`models/arch/MODEL_ARCH_TOML_DSL_SPEC.md`. **Read it before writing or modifying any `.arch.toml`.**

**Validation** runs at parse time (`Validate()` in `arch.go`): unknown keys, missing
weights, bad param refs, and contract violations all produce errors with source line
numbers. A successful `arch.Load()` means the definition is structurally sound.

**Adding a new architecture:** Copy the closest existing `.arch.toml`. The validator
and builder contracts will identify exactly what is wrong.

**For safetensors models:** also create a matching `.arch.stmap.toml` file that maps
HF config.json keys → GGUF metadata keys and HF tensor names → our short names. The
stmap's `architecture.hf_class` must match `config.json["architectures"][0]` for the
target architecture. Without it, `NewSafetensorsReader` will fail to locate a stmap.

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

**`ModelReader` interface** (`arch/model_reader.go`): abstracts metadata + tensor loading behind a single contract. Both GGUF and safetensors implement `ModelReader` — `newGenericModelFromReader()` is format-agnostic:

```go
type ModelReader interface {
    GetU32, GetF32, GetArrInts, GetArrBools  // GGUFReader metadata access
    GetTensorDim(tensorName, dim)            // shape query
    TensorCount() int                         // total tensor count
    TensorNames() []string                    // all tensor names
    TensorSpec(name string) (TensorSpec, bool) // shape, dtype, byte size
    ReadTensor(name string, buf []byte) error // raw bytes into caller buffer
    Close() error                            // release file handles
}

type TensorSpec struct {
    Type int               // ggml type constant (e.g., ggml.TypeF16)
    Ne   [4]int64          // dimensions, padded to 4 (unused = 1)
    Size int               // total bytes in ggml format
}
```

**Construction**: `NewModelReader(path, *GGUFFile)` for GGUF; `NewSafetensorsReader(stDir, archDir)` for safetensors. Both return `ModelReader`.

**`NewGenericModel(archName, modelPath, archDir, gf)`** → `*GenericModel` (GGUF path):
1. Parse `.arch.toml` → `ArchDef`
2. Create `ModelReader` from GGUF file (reuses pre-parsed `gf` if provided)
3. Delegates to `newGenericModelFromReader()` (steps 3–9 below)

**`NewSafetensorsModel(archName, stDir, archDir)`** → `*GenericModel` (safetensors path):
1. Parse `.arch.toml` → `ArchDef`
2. Create `ModelReader` via `NewSafetensorsReader(stDir, archDir)`:
   a. Load `model.safetensors.index.json` (500KB guard) or parse single `model.safetensors`
   b. Read `config.json`, extract `architectures[0]` as HF class
   c. Find matching `.arch.stmap.toml` via `FindSTMapByHFClass()` — scans all stmap files, returns the one with `architecture.hf_class` matching
   d. Build param value map: stmap says `hf_param → GGUF_key`, values come from `config.json`
   e. Precompute tensor specs (dtype → ggml type mapping; BF16 mapped to F32)
   f. Open all shard files for `ReadAt`
   g. Return `stReaderAdapter` implementing `ModelReader`
3. Delegates to `newGenericModelFromReader()` (steps 3–9 below)

**`newGenericModelFromReader(reader, def, modelPath)`** → `*GenericModel` (shared path):

The loader is structured as a `genericModelBuilder` struct with phase methods, called sequentially from `build()`. The builder owns partial state until each phase succeeds; on failure, `cleanupOnError` releases everything allocated so far. The `WeightStore` is the ownership cut line: before `buildWeightStore` runs the builder owns `gpu`/`cpu`/`weightCtx`/`weightBuf` individually; after, `store.Close()` frees all four in one call. Compute resources (`cachedCtx`, `cachedSched`) created in the final phase are owned by the model and freed before `store.Close()`.

Phases (each a method on `genericModelBuilder`):
1. `checkMemory` — verify the model fits available VRAM/RAM via `reader.MinMemoryRequired(maxSeqLen)`
2. `resolveArch` — resolve params → `ResolvedParams`, then **validate**: every required param is present, typed, and nonzero where zero would cause silent downstream divergence (e.g. `rms_eps`, `rope_freq_base`). Bails out with a named error before any allocation. Then resolve weights → `ResolvedLayerWeights` per layer, build `CanonicalModuleMap` and `TensorDimsMap`.
3. `initBackendsAndArena` — bring up GPU/CPU backends, create the weight context (`AllocPermDisallow`), build the tensor-name → tensor index from reader specs, allocate weight VRAM via `AllocCtxTensors`.
4. `uploadWeights` — for each weight tensor: `reader.ReadTensor()` into a byte buffer → `ggml.ValidateRowData(spec.Type, buf)` → `ggml.TensorSetBytes(t, buf, 0)`. Validation inspects block scale/delta fields for quantized types (near-zero cost) and every element for float types; a NaN/Inf byte is a hard load error with the offending tensor name. This catches corrupt weight files and reader type/shape mismatches at load time, not as NaN logits mid-generation.
5. `buildWeightStore` — wrap the backends + arena in a `WeightStore`, transferring ownership; resolve global and per-layer tensor maps; handle tied embeddings; validate required globals are present.
6. `assignBuilders` — determine per-layer FFN routing (`ffn` vs `ffn_alt`), assign `BlockBuilders[]` and `FFNBuilders[]` per layer.
7. `createComputeResources` — create persistent `cachedCtx` (`AllocPermDisallow`, sized via `graphCtxSize()`) and `cachedSched` (sized to `maxGraphNodes`) for reuse across decode tokens; allocate `ffnScratch` and `logitBuf` scratch buffers for the hot decode path.

```go
type GenericModel struct {
    Def               *ArchDef
    Params            *ResolvedParams
    Weights           *ResolvedWeights       // tensor name maps per layer (format-agnostic above ModelReader);
                                             // ResolvedLayerWeights.Prefix is the canonical per-layer prefix source
    Store             *WeightStore           // GPU-resident tensors
    LayerBlockNames   []string               // which block builder each layer uses
    BlockBuilders     []BlockBuilder
    FFNBuilders       []FFNBuilder
    FFNConfigs        []map[string]any       // per-layer FFN config ([ffn.config] / [ffn_alt.config])
    CanonicalModuleMap *ModuleMap            // immutable; always Clone() before modifying
    HeadDim           int                    // attention head dimension
    TensorDims        TensorDimsMap          // tensor shape info for diagnostics
    ModelPath         string                 // path to GGUF or .st/ directory

    // Persistent compute resources for ForwardCached. Created at load, reused across tokens.
    cachedCtx   *ggml.GraphContext  // sized via arch.graphCtxSize(); AllocPermDisallow
    cachedSched *ggml.Sched         // sized to arch.maxGraphNodes

    // Pre-allocated scratch buffers for the hot decode path.
    ffnScratch map[string]ggml.Tensor // reused by buildFFNBlock each layer
    logitBuf   []float32              // reused by readLogits each token
}
```

---

## Cache (`arch/cache.go`)

`GenericModel.NewCache(maxSeqLen)` allocates per-layer cache tensors on GPU VRAM and pre-allocates `maskBuf` and `swaMaskBuf` (each sized to `maxSeqLen`) for reuse in `buildCausalMaskData` / `buildSWAMaskData` during cached decode. `ForwardStateless` allocates its own mask buffers and does not use these fields.

`GenericCache.Clear()` zeroes the entire cache backing buffer in a single `cacheBuf.Clear(0)` backend call rather than iterating per-tensor.

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
- `ForwardStateless(tokenIDs, mask, caps) ([]float32, *EngagementData, error)` — full recompute; returns logits for the last token position, per-layer engagement data, and error
- `ForwardStatelessAllLogits(tokenIDs, mask, flashAttn) ([]float32, error)` — full recompute; returns logits for **all** token positions, row-major by position: position `p` occupies `allLogits[p*nVocab:(p+1)*nVocab]`. Used exclusively by `generateDiffusion` — no engagement data or captures are collected.
- `ForwardCached(gc, tokenIDs, mask) ([]float32, error)` — KV-cached; returns logits and error

Both stateless entry points share a private `forwardStatelessCore` helper that builds the graph context, scheduler, inputs, and runs all layers. The two stateless entry points differ only in how they extract logits after `sched.Compute`.

All three entry points follow the same layer-execution frame:
1. Extract params; allocate graph context + GPU scheduler. Stateless paths build a fresh `GraphContext` per call sized via `graphCtxSize()` (= `ggml.GraphContextSize(maxGraphNodes)` — derived from ggml's own cgraph + tensor-overhead accounting). `ForwardCached` reuses the persistent `cachedCtx` / `cachedSched` allocated at model load. All call sites pass `maxGraphNodes` (= 16384) consistently to the context, `NewGraph`, and `NewSched` — these three must agree.
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

**Type and signature conventions** (enforced as the API for every Go caller):
- `ggml.GGMLType` — named integer type for tensor element types (e.g. `TypeF32`, `TypeQ4_K`). Distinct from `int` so that tensor type arguments cannot be silently swapped with unrelated ints (ne dimensions, mode flags, etc.).
- `ggml.AllocPerm` — named type controlling whether a `GraphContext` arena holds tensor *data* (`AllocPermAllow`) or only tensor *descriptors* (`AllocPermDisallow`, the normal graph-build case). `NewGraphContext(memSize int, allocPerm AllocPerm)` is non-variadic — every caller must declare its intent explicitly. There is no `AllocPermDefault`.
- `opt*` parameter prefix — nullable tensor parameters in op wrappers use the `opt` prefix to flag that callers may pass `NilTensor()` (e.g. `SoftMaxExt(ctx, a, optMask, ...)`, `RopeExt(ctx, a, pos, optFreqFactors, ...)`, `FlashAttnExt(ctx, q, k, v, optMask, ...)`). Required tensor parameters have no prefix.

**Principled context sizing** (`backend.go`):
- `GraphOverheadCustom(size int, grads bool) int` — exact bytes ggml requires for a cgraph structure (nodes array, leafs array, hash tables) of the given size. Thin wrapper over `ggml_graph_overhead_custom`.
- `GraphContextSize(maxNodes int) int` — minimum context arena bytes needed to build a graph of up to `maxNodes` nodes: `GraphOverheadCustom + TensorOverhead*maxNodes + 64` alignment slop. Use this with `NewGraphContext` to size graph contexts precisely from the declared maxNodes budget — never ad-hoc multipliers.
- `Buffer.Clear(byte)` — single backend call that writes `value` to every byte in the buffer (including alignment padding). Used by `GenericCache.Clear` instead of per-tensor zero loops.

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

**Go wrappers for load-time validation (in `ggml/`):**
- `ValidateRowData(typ int, data []byte) bool` — thunks to upstream `ggml_validate_row_data`. Full element scan for float types; per-block scale/delta scan for quantized types. Returns true on empty input. Called by `newGenericModelFromReader` before every `TensorSetBytes`. Unit tested in `ggml/validate_test.go` (F32 NaN/Inf, I32 pass-through, Q4_K block-scale NaN).

**GGUF metadata reading**: `arch/model.go` uses pure-Go `gguf-parser-go` — no CGo. The `goGGUFReader` adapter implements the `GGUFReader` interface with type-checked KV access (guards against gguf-parser-go's panic-on-wrong-type behavior). Split GGUF files are rejected at parse time.

**Safetensors parsing**: `arch/safetensors_index.go` and `arch/safetensors_reader.go` are pure Go — JSON index parsing, shard header reading, and BF16→F32 conversion are all done without CGo. The `stReaderAdapter` implements `ModelReader` using `os.ReadAt` on shard files.

**Verified ggml semantics** (do not change without re-verifying):
- `ggml_mul_mat(A, B)`: contracts over `A→ne[0] == B→ne[0]`; result `[A→ne[1], B→ne[1], ...]`; GQA broadcasting when `B→ne[2] % A→ne[2] == 0`
- `ggml_permute(ctx, a, ax0..ax3)`: input dim i → output position ax_i
- `ggml_rms_norm`: normalizes over `ne[0]` per slice
- `ggml_soft_max_ext(a, mask, scale, max_bias)`: `softmax(scale * a + mask)` over `ne[0]`

---

## Defensive Checks

Silent numerical failures are the highest-cost class of bug in this codebase — a
single NaN in a Q4_K block scale propagates through every row the block touches
and surfaces as "the model produces gibberish," forcing hours of bisection
through the load and forward paths. The checks below convert that class of
failure into loud load-time or sampler-time errors with the offending
tensor/position named, plus two debugging aids (param dumps and the
GGUF↔safetensors equivalence test).

**Param validation (load time, `arch/params.go`)**  
After `ResolveParams`, the loader asserts that every required param is present,
typed correctly, and nonzero where zero would cause silent downstream
divergence. Concrete failures this catches:
- `rms_eps = 0` (from untyped JSON `1e-6` silently truncated to `uint32 0`) →
  `1/sqrt(0) = Inf` in every RMS norm
- `rope_freq_base` present in `Ints` but not `Floats` → positional encoding
  computed against zero base → NaN positions → first-token-EOS output
- Missing `n_heads` / `n_kv_heads` / `head_dim` → attention scale computation
  fails later rather than at load

Fails before any VRAM allocation; the error names the param and the reader
that produced it.

**Weight validation (load time, `arch/model.go`)**  
Every tensor's raw bytes are validated via `ggml.ValidateRowData(spec.Type,
buf)` before `TensorSetBytes` uploads them to VRAM. The underlying
`ggml_validate_row_data`:
- Float types (F32, F16, BF16, F64) — scans every element for NaN/Inf
- Quantized types (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K,
  Q8_K) — scans each block's scale/delta fields for NaN/Inf (near-zero cost;
  quants themselves are integers so they cannot be NaN)
- MXFP4 — E8M0 scale validation
- Integer types (I8, I16, I32, I64) — pass-through (no representation for NaN)

On failure the C layer prints `ggml_validate_row_data: found nan value at
block N` to stderr (routed through the Go logger) and the Go wrapper returns
false. `newGenericModelFromReader` converts that into a named load error that
identifies the tensor and the likely cause (corrupt file or reader type/shape
mismatch).

Unit tested in `internal/ggml/validate_test.go`: F32 finite/empty/NaN/Inf,
I32 pass-through, Q4_K well-formed block passes, Q4_K F16 NaN d-scale fails.

**Logit validation (sample time, `inference/sampler.go`)**  
`ValidateLogits` runs at every sampler chokepoint. Autoregressive paths
(`generate_cached`, `generate_stateless`, `generate_rlb`) hit it transitively
via `Engine.sample()` in `engine.go`, which validates before any sampling
math. The diffusion path (`generate_diffusion`) calls it explicitly after
each `ForwardStatelessAllLogits`, wrapped with `blockNum` / `step` context.
If any logit is NaN/Inf the request fails with a named error rather than
emitting garbage tokens. Cost is one linear scan of `n_vocab` per call —
memory-bandwidth bound, negligible vs. a full forward pass.

**Param dumps (debugging aid, `arch/model_reader_safetensors.go`,
`arch/model_reader_gguf.go`)**  
Both readers emit a sorted `[param] key = value` dump at DEBUG level listing
every metadata KV seen from the source file. When porting a new architecture
to safetensors, visual diff of the two dumps is the fastest way to find a
stmap error: any param present in one but absent/wrong in the other is the
bug. Keep these dumps symmetric — a difference in dump format between loaders
hides exactly the class of bug they are there to surface.

**Equivalence testing (acceptance gate, `make equiv-test`)**  
`test_equiv.sh gguf-st` compares logprobs between the GGUF and safetensors
loaders for the same model, at the same sampler step, on the same prompt.
Non-identical above GPU FP variance is a hard failure. This test is on the
default Makefile testing path and is the authoritative gate for any change
touching the load or inference path — it must be green before a change is
considered complete.

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

## Safetensors Loading Pipeline

Safetensors models from HuggingFace use a different directory layout and naming convention
than GGUF. The loading pipeline translates both into the format-agnostic `ModelReader`
interface so that inference code is identical above the abstraction boundary.

### Directory Convention

```
models/
  Llama-3.2-3B-Instruct.Q4_K_M.gguf          ← GGUF model
  Llama-3.2-3B-Instruct.st/                   ← safetensors directory (.st suffix)
    model-00001-of-00002.safetensors          ← shard file(s)
    model.safetensors.index.json              ← JSON index (weight → shard mapping)
    config.json                               ← HF model config
    tokenizer.gguf                            ← tokenizer sidecar (vocab only, no weights)
```

Format is auto-detected: `*.gguf` → GGUF parser; `*.st/` → safetensors directory.
`ModelInfo.Format` (`FormatGGUF` / `FormatSafetensors`) carries the result into
`NewEngine()`.

### Index Parsing (`safetensors_index.go`)

`LoadSafetensorsIndex(stDir)` handles both sharded and single-file models:
1. **Sharded**: reads `model.safetensors.index.json` (500KB guard), parses `weight_map`
   to build ordered shard list, then reads each shard's 8-byte header length + JSON header
   to populate `STTensorEntry` per tensor (dtype, shape, absolute data offset, shard assignment).
2. **Single-file**: reads `model.safetensors` directly, parses its JSON header.

Both paths produce a `SafetensorsIndex` with `Tensors` map and `Shards` list.

### STMap Resolution (`stmap.go`, `safetensors_reader.go`)

Safetensors tensor/param names differ from GGUF conventions. A per-architecture
`.arch.stmap.toml` file provides the translation (see
`models/arch/MODEL_ARCH_STMAP_TOML_DSL_SPEC.md`). The lookup chain:

1. `config.json["architectures"][0]` → HF class (e.g. `"LlamaForCausalLM"`)
2. `FindSTMapByHFClass(archDir, hfClass)` scans `*.arch.stmap.toml` files, returns the
   one with `architecture.hf_class` matching
3. STMap provides:
   - `[params]`: HF config.json key → GGUF metadata key mapping
   - `[layer_prefix].hf`: per-layer prefix with `{N}` substitution
   - `[tensors]`: our short tensor name → HF short tensor name
   - `[tensors.global]`: our short global name → HF full tensor name

Param resolution: HF config.json value is stored under the GGUF key name, so the
existing `ResolveParams()` machinery works without changes. Note: safetensors models
only provide scalar params from `config.json` — array types (`GetArrInts`, `GetArrBools`)
are not available from that format and will always return `(nil, false)`.

Tensor name resolution at load time:
- Per-layer: `layer_prefix.hf` (with `{N}` resolved) + `tensors[our_short_name]` = full HF tensor key
- Global: `tensors.global[our_short_name]` used directly

### Reader Construction (`safetensors_reader.go`)

`NewSafetensorsReader(stDir, archDir)` builds an `stReaderAdapter` implementing
`ModelReader`:

1. Parse safetensors index → `SafetensorsIndex`
2. Read `config.json`
3. Extract HF class → find matching stmap
4. Build param value map from stmap + config.json
5. Precompute `TensorSpec` for all tensors (dtype → ggml type mapping via `stDtypeToGGML`)
6. Open all shard files for `ReadAt`-based random access

**BF16 conversion**: ggml has no native BF16 type. At load time, BF16 data is expanded
to F32 — the upper 16 bits of BF16 (sign + 8-bit exponent + 7-bit mantissa) are placed
in the upper half of the F32 value with 16 zero bits appended as the lower half. This is
a byte-level operation: read 2 bytes, write 4 bytes (bottom-up to avoid overwriting).

### Engine Integration (`engine.go`)

`NewEngine()` dispatches on `info.Format`:
- `FormatSafetensors`: loads tokenizer from `tokenizer.gguf` sidecar in the `.st/`
  directory (a minimal GGUF with only tokenizer metadata), then calls
  `arch.NewSafetensorsModel(archName, stDir, archDir)`
- `FormatGGUF`: existing GGUF path; reuses pre-parsed GGUF to avoid double-parsing

For safetensors models, `NLayers` may be 0 if `config.json` lacked the field — falls
back to the resolved model param (`m.Params.GetInt("n_layers")`).

### Model Discovery (`model/safetensors.go`)

`ParseSafetensorsDir(stDir, archDir)` produces `GGUFMetadata` for the model manager:
1. Load safetensors index for tensor inventory
2. Convert `STTensorEntry` → `TensorInfo` (offset=0, no unified safetensors offset concept)
3. Read `config.json` for architecture class and numeric params (`num_hidden_layers`,
   `hidden_size`, etc.)
4. Resolve HF class → architecture name via `FindSTMapByHFClass()`
5. Return `GGUFMetadata` with resolved architecture name

The manager (`manager.go`) scans both `*.gguf` files and `*.st/` directories during
`scan()` and `List()`, filtering by whether the resolved architecture has a matching
`.arch.toml` file. `tryLoadOne()` checks GGUF first, then falls back to `.st/`.

---

## Key Invariants

1. **No model-specific Go code.** Architecture branches belong in TOML.
2. **`CanonicalModuleMap` is never mutated.** Always `Clone()` before modification.
3. **NilTensor means culled.** Absent map keys return zero-value `ggml.Tensor`. Every builder checks `IsNil()`.
4. **`os.Exit` never below CLI entry point.** Return errors up the chain. `log.Fatal` is permitted only in `bench/` cobra entry points; utility and library packages (including `util/paths.go`) return errors. `util.ResolvePaths()` returns `(BenchPaths, error)`; `BenchPaths` is injected from the cobra entry into `Server`, `Engine`, and culling helpers — no library code calls `ResolvePaths` itself.
5. **CGo stays in `ggml/`.** No other package imports `"C"`. GGUF metadata is read via pure-Go `gguf-parser-go`; safetensors index and tensor data are parsed in pure Go.
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
16. **`ModelReader` is the format boundary.** Everything above `newGenericModelFromReader()` is format-agnostic. Adding a new model format requires only a new `ModelReader` implementation.
17. **Tokenizer is GGUF-only.** Safetensors models use a `tokenizer.gguf` sidecar; no HuggingFace `tokenizer.json` parsing path exists.
18. **`NewGraphContext` requires explicit `AllocPerm`.** No default. Graph-build contexts pass `AllocPermDisallow`; data-arena scratch contexts (load-time type conversion) pass `AllocPermAllow`. Caller intent must be visible at the call site.
19. **Single source of truth for the cgraph node budget.** `arch.maxGraphNodes = 16384` drives both the context arena (via `arch.graphCtxSize()` → `ggml.GraphContextSize`) and every `NewGraph` / `NewSched` call in the arch package. Never use a literal node count.
20. **Canonical logical weight names live in `arch_util.go`.** `WeightAttnNorm`, `WeightFFNNorm` (and the `Cache*` keys) are constants; never inline these literal strings in graph or module code.
21. **`ResolvedLayerWeights.Prefix` is the canonical per-layer prefix.** All consumers (module map, tensor dims, diagram code) read `lw.Prefix` (already includes the trailing dot). Never reconstruct `blk.<N>.` from `layer_idx` ad hoc.

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
