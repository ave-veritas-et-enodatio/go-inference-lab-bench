package arch

import (
	"fmt"

	ggufparser "github.com/gpustack/gguf-parser-go"

	"inference-lab-bench/internal/ggml"
	"inference-lab-bench/internal/log"
)

// GenericModel is a model loaded from a GGUF file using an architecture definition.
type GenericModel struct {
	Def             *ArchDef
	Params          *ResolvedParams
	Weights         *ResolvedWeights
	Store           *WeightStore     // immutable weight storage (read-only after load)
	LayerBlockNames []string         // which block each layer uses
	BlockBuilders   []BlockBuilder   // per-layer block builder
	FFNBuilders     []FFNBuilder     // per-layer FFN builder
	FFNConfigs      []map[string]any // per-layer FFN config (from [ffn.config] / [ffn_alt.config])

	// Canonical module map and supporting data for per-query culling.
	CanonicalModuleMap *ModuleMap
	HeadDim            int // elements per attention head (for flash attention geometry check)
	TensorDims         TensorDimsMap
	ModelPath          string

	// Persistent compute resources for ForwardCached. Created at model load, reused across tokens.
	cachedCtx   *ggml.GraphContext
	cachedSched *ggml.Sched

	// Pre-allocated scratch buffers for the hot decode path (ForwardCached).
	ffnScratch map[string]ggml.Tensor // reused by buildFFNBlock each layer
	logitBuf   []float32              // reused by readLogitsInto each token

	// rlb bundles all RLB-specific state (block ranges, lazy scratch
	// context/scheduler). Defined in graph_rlb.go so RLB internals stay out
	// of this file.
	rlb rlbState
}

// NewGenericModelFromGGUF loads a GGUF model using the named architecture definition.
// archDir is the directory containing architecture definition TOML files.
// If gf is non-nil it is used directly; otherwise the GGUF is parsed from modelPath.
func NewGenericModelFromGGUF(memStats ggml.MemoryStats, maxSeqLen int, archDef *ArchDef, modelPath, archDir string, gf *ggufparser.GGUFFile) (*GenericModel, error) {
	if archDef == nil {
		return nil, fmt.Errorf("valid ArchDef required")
	}
	reader, err := NewModelReaderGGUF(archDef, modelPath, gf)
	if err != nil {
		return nil, err
	}

	return newGenericModelFromReader(memStats, maxSeqLen, reader, archDef, modelPath)
}

// NewGenericModelFromSafetensors loads a safetensors model using the named architecture definition.
// archDir is the directory containing architecture definition TOML and stmap files.
// stDir is the directory containing the safetensors shards and config.json.
func NewGenericModelFromSafetensors(memStats ggml.MemoryStats, maxSeqLen int, archDef *ArchDef, stDir, archDir string) (*GenericModel, error) {
	if archDef == nil {
		return nil, fmt.Errorf("valid ArchDef required")
	}
	reader, err := NewModelReaderSafetensors(archDef, stDir, archDir)
	if err != nil {
		return nil, err
	}

	return newGenericModelFromReader(memStats, maxSeqLen, reader, archDef, stDir)
}

// newGenericModelFromReader is the shared loading path used by both GGUF and safetensors.
// Takes an already-constructed ModelReader and loaded ArchDef. Always closes the
// reader before returning. On error, releases any resources allocated so far.
func newGenericModelFromReader(memStats ggml.MemoryStats, maxSeqLen int, reader ModelReader, def *ArchDef, modelPath string) (*GenericModel, error) {
	b := &genericModelBuilder{
		memStats:  memStats,
		maxSeqLen: maxSeqLen,
		reader:    reader,
		def:       def,
		modelPath: modelPath,
	}
	return b.build()
}

// genericModelBuilder owns the partial state accumulated while constructing a
// GenericModel. Splitting the loader into phase methods on this struct keeps
// the top-level control flow readable without passing a dozen locals around.
//
// Ownership model for cleanup:
//   - Before buildWeightStore(): gpu, cpu, weightCtx, weightBuf (if allocated)
//     are owned by the builder and must be freed individually on error.
//   - After buildWeightStore(): store owns all of the above; store.Close()
//     frees everything in one call.
//   - After createComputeResources(): m.cachedCtx / m.cachedSched are owned by
//     the model; cleanupOnError frees them before closing store.
type genericModelBuilder struct {
	// Inputs
	memStats  ggml.MemoryStats
	maxSeqLen int
	reader    ModelReader
	def       *ArchDef
	modelPath string

	// Resolved architecture
	params      *ResolvedParams
	weights     *ResolvedWeights
	headDim     int
	canonicalMM *ModuleMap
	tensorDims  TensorDimsMap

	// Backends + weight arena — ownership transfers to store once built.
	gpu               *ggml.Backend
	cpu               *ggml.Backend
	weightCtx         *ggml.GraphContext
	weightBuf         *ggml.Buffer
	weightTensorIndex map[string]ggml.Tensor

	// Built model — owns the store and compute resources from phase H onward.
	store *WeightStore
	m     *GenericModel
}

func (b *genericModelBuilder) build() (m *GenericModel, err error) {
	defer b.reader.Close()
	defer func() {
		if err != nil {
			b.cleanupOnError()
		}
	}()

	if err = b.checkMemory(); err != nil {
		return nil, err
	}
	if err = b.resolveArch(); err != nil {
		return nil, err
	}
	if err = b.initBackendsAndArena(); err != nil {
		return nil, err
	}
	if err = b.uploadWeights(); err != nil {
		return nil, err
	}
	if err = b.buildWeightStore(); err != nil {
		return nil, err
	}
	if err = b.assignBuilders(); err != nil {
		return nil, err
	}
	if err = b.createComputeResources(); err != nil {
		return nil, err
	}
	return b.m, nil
}

// cleanupOnError releases any resources the builder has allocated so far.
// Safe to call at any phase — uses store ownership as the cut line.
func (b *genericModelBuilder) cleanupOnError() {
	// Model-scoped compute resources (created after store).
	if b.m != nil {
		if b.m.cachedSched != nil {
			b.m.cachedSched.Free()
			b.m.cachedSched = nil
		}
		if b.m.cachedCtx != nil {
			b.m.cachedCtx.Free()
			b.m.cachedCtx = nil
		}
		b.m.rlb.Free()
	}
	// Store owns gpu/cpu/weightCtx/weightBuf — its Close frees everything.
	if b.store != nil {
		b.store.Close()
		return
	}
	// Store not yet built — free raw resources individually.
	if b.weightBuf != nil {
		b.weightBuf.Free()
	}
	if b.weightCtx != nil {
		b.weightCtx.Free()
	}
	if b.gpu != nil {
		b.gpu.Free()
	}
	if b.cpu != nil {
		b.cpu.Free()
	}
}

// checkMemory verifies the model fits in available VRAM/RAM.
func (b *genericModelBuilder) checkMemory() error {
	memReq := b.reader.MinMemoryRequired(b.maxSeqLen)
	if b.memStats.IsUMA {
		if memReq.UnifiedRAM() > uint64(b.memStats.VRAM.Available()) {
			return fmt.Errorf("insufficient unified RAM min required: %d  available: %d",
				memReq.UnifiedRAM(),
				b.memStats.VRAM.Available(),
			)
		}
		return nil
	}
	if memReq.TotalVRAM() >= uint64(b.memStats.VRAM.Available()) || memReq.OverheadRAM >= uint64(b.memStats.RAM.Available()) {
		return fmt.Errorf("insufficient VRAM/RAM min required: %d/%d  available: %d/%d",
			memReq.TotalVRAM(), memReq.OverheadRAM,
			b.memStats.VRAM.Available(), b.memStats.RAM.Available(),
		)
	}
	return nil
}

// resolveArch resolves params, weights, and builds the canonical module map
// and tensor dims (used by per-query culling).
func (b *genericModelBuilder) resolveArch() error {
	params, err := ResolveParams(b.def, b.reader)
	if err != nil {
		return fmt.Errorf("resolving params: %w", err)
	}
	weights, err := ResolveWeights(b.def, params)
	if err != nil {
		return fmt.Errorf("resolving weights: %w", err)
	}
	b.params = params
	b.weights = weights
	b.headDim = params.Ints["head_dim"]
	b.canonicalMM = BuildModuleMap(weights)
	b.tensorDims = BuildTensorDimsMap(weights, func(name string) (int64, int64, int64, bool) {
		if s, ok := b.reader.TensorSpec(name); ok {
			return s.Ne[0], s.Ne[1], int64(s.Size), true
		}
		return 0, 0, 0, false
	}, b.headDim)
	return nil
}

// initBackendsAndArena brings up the GPU/CPU backends, creates the weight
// context, builds the tensor-name → tensor index from reader specs, and
// allocates all weight storage on the GPU.
func (b *genericModelBuilder) initBackendsAndArena() error {
	b.gpu = ggml.GPUInit()
	if b.gpu == nil {
		return fmt.Errorf("failed to init GPU backend")
	}
	b.cpu = ggml.CPUInit()

	nTensors := int64(b.reader.TensorCount())
	ctxSize := int64(ggml.TensorOverhead())*nTensors + 1*1024*1024
	b.weightCtx = ggml.NewGraphContext(int(ctxSize), ggml.AllocPermDisallow)
	if b.weightCtx == nil {
		return fmt.Errorf("failed to create weight context")
	}

	b.weightTensorIndex = make(map[string]ggml.Tensor, nTensors)
	for _, name := range b.reader.TensorNames() {
		spec, ok := b.reader.TensorSpec(name)
		if !ok {
			continue
		}
		t := makeTensorFromSpec(b.weightCtx, spec.Type, spec.Ne[0], spec.Ne[1], spec.Ne[2], spec.Ne[3])
		ggml.SetName(t, name)
		b.weightTensorIndex[name] = t
	}

	b.weightBuf = ggml.AllocCtxTensors(b.weightCtx, b.gpu)
	if b.weightBuf == nil {
		return fmt.Errorf("failed to alloc GPU VRAM")
	}
	return nil
}

// uploadWeights reads each tensor's raw bytes from the reader, runs
// ggml_validate_row_data on them, and uploads them to GPU VRAM.
//
// ggml_validate_row_data inspects each block's scale and delta fields for
// NaN/Inf at near-zero cost on quantized types; for float types it scans every
// element. A single bad block scale in a Q4_K weight propagates silently to
// every output row that block touches during inference, producing the "first
// token EOS / garbage text" symptom that took hours to diagnose in the
// safetensors load path. Catching it here turns that into a loud one-line
// load-time error naming the offending tensor.
func (b *genericModelBuilder) uploadWeights() error {
	for _, name := range b.reader.TensorNames() {
		t, ok := b.weightTensorIndex[name]
		if !ok {
			continue
		}
		spec, _ := b.reader.TensorSpec(name)
		buf := make([]byte, spec.Size)
		if err := b.reader.ReadTensor(name, buf); err != nil {
			return fmt.Errorf("read tensor %s: %w", name, err)
		}
		if !ggml.ValidateRowData(spec.Type, buf) {
			return fmt.Errorf("tensor %s failed ggml_validate_row_data "+
				"(NaN/Inf in raw bytes — see stderr for offending block index; "+
				"likely cause: corrupted weight file or reader type/shape mismatch)", name)
		}
		ggml.TensorSetBytes(t, buf, 0)
	}
	return nil
}

// buildWeightStore wraps the backends + weight arena in a WeightStore,
// resolves global and per-layer tensor maps, handles tied embeddings, and
// validates that each block builder's required weights are present.
//
// On success, the store takes ownership of gpu, cpu, weightCtx, weightBuf —
// subsequent cleanup runs through store.Close() only.
func (b *genericModelBuilder) buildWeightStore() error {
	store := &WeightStore{
		global: make(map[string]ggml.Tensor),
		ctx:    b.weightCtx,
		Buffer: b.weightBuf,
		GPU:    b.gpu,
		CPU:    b.cpu,
	}

	for logicalName, tensorName := range b.weights.Global {
		t, ok := b.weightTensorIndex[tensorName]
		if !ok {
			t = ggml.NilTensor()
		}
		store.global[logicalName] = t
	}

	if b.def.Architecture.TiedEmbeddings && store.Global("output").IsNil() {
		store.global["output"] = store.Global("token_embd")
	}

	// Transfer ownership to store before any error return so cleanupOnError
	// uses store.Close() instead of double-freeing via individual fields.
	b.store = store

	if store.Global("token_embd").IsNil() || store.Global("output_norm").IsNil() {
		return fmt.Errorf("missing required global weight tensor: token_embd or output_norm")
	}
	if store.Global("output").IsNil() {
		return fmt.Errorf("missing required global weight tensor: output")
	}

	nLayers := b.params.Ints["n_layers"]
	store.layers = make([]map[string]ggml.Tensor, nLayers)

	for i, lw := range b.weights.Layers {
		lt := make(map[string]ggml.Tensor)

		for logicalName, tensorName := range lw.Common {
			t, ok := b.weightTensorIndex[tensorName]
			if !ok {
				t = ggml.NilTensor()
			}
			lt[logicalName] = t
		}

		blockDef := b.def.Blocks[lw.BlockName]
		for logicalName, tensorName := range lw.Block {
			t, ok := b.weightTensorIndex[tensorName]
			if !ok {
				// Fallback: try the raw suffix as a global tensor name (e.g. rope_freqs.weight)
				if suffix, has := blockDef.Weights[logicalName]; has {
					t, ok = b.weightTensorIndex[suffix]
				}
			}
			if !ok {
				t = ggml.NilTensor()
			}
			lt[logicalName] = t
		}

		// n_kv_shared_layers: non-KV layers still have weights in the model but
		// must NOT compute their own K/V or write to cache. Null out K/V weights
		// so the builder's hasKV check correctly identifies them as non-KV.
		if nKVShared, ok := b.params.Ints["n_kv_shared_layers"]; ok && nKVShared > 0 {
			nKVFromStart := nLayers - nKVShared
			if i >= nKVFromStart {
				lt["attn_k"] = ggml.NilTensor()
				lt["attn_v"] = ggml.NilTensor()
			}
		}

		for logicalName, tensorName := range lw.FFN {
			t, ok := b.weightTensorIndex[tensorName]
			if !ok {
				t = ggml.NilTensor()
			}
			lt["ffn_"+logicalName] = t
		}

		// FFN alt weights (also prefixed with ffn_)
		for logicalName, tensorName := range lw.FFNAlt {
			t, ok := b.weightTensorIndex[tensorName]
			if !ok {
				t = ggml.NilTensor()
			}
			lt["ffn_"+logicalName] = t
		}

		store.layers[i] = lt

		// Validate that all required weights for this layer's block builder are present.
		if bb, ok := GetBlockBuilder(blockDef.Builder); ok {
			contract := bb.Contract()
			for _, req := range contract.RequiredWeights {
				if lt[req].IsNil() {
					return fmt.Errorf("layer %d (block %q): required weight %q is missing from model", i, lw.BlockName, req)
				}
			}
		}
	}

	return nil
}

// assignBuilders instantiates the model struct, assigns per-layer block and
// FFN builders (including per-layer alt-FFN routing via layerHasAltFFN), and
// emits the load summary log.
func (b *genericModelBuilder) assignBuilders() error {
	nLayers := b.params.Ints["n_layers"]

	m := &GenericModel{
		Def:                b.def,
		Params:             b.params,
		Weights:            b.weights,
		Store:              b.store,
		CanonicalModuleMap: b.canonicalMM,
		HeadDim:            b.headDim,
		TensorDims:         b.tensorDims,
		ModelPath:          b.modelPath,
	}
	b.m = m

	m.LayerBlockNames = make([]string, nLayers)
	m.BlockBuilders = make([]BlockBuilder, nLayers)
	for i, lw := range b.weights.Layers {
		m.LayerBlockNames[i] = lw.BlockName
		bb, ok := GetBlockBuilder(b.def.Blocks[lw.BlockName].Builder)
		if !ok {
			return fmt.Errorf("layer %d: unknown block builder %q", i, b.def.Blocks[lw.BlockName].Builder)
		}
		m.BlockBuilders[i] = bb
	}

	fb, ok := GetFFNBuilder(b.def.FFN.Builder)
	if !ok {
		return fmt.Errorf("unknown FFN builder %q", b.def.FFN.Builder)
	}
	var fbAlt FFNBuilder
	if b.def.FFNAlt != nil {
		fbAlt, ok = GetFFNBuilder(b.def.FFNAlt.Builder)
		if !ok {
			return fmt.Errorf("unknown FFN alt builder %q", b.def.FFNAlt.Builder)
		}
	}
	m.FFNBuilders = make([]FFNBuilder, nLayers)
	m.FFNConfigs = make([]map[string]any, nLayers)
	for i, lw := range b.weights.Layers {
		// Use alt FFN if this layer has alt weights in the model
		if fbAlt != nil && len(lw.FFNAlt) > 0 && layerHasAltFFN(b.store, lw) {
			m.FFNBuilders[i] = fbAlt
			m.FFNConfigs[i] = b.def.FFNAlt.Config
		} else {
			m.FFNBuilders[i] = fb
			m.FFNConfigs[i] = b.def.FFN.Config
		}
	}

	// Compute block boundaries for per-block RLB. InitBlockRanges is a no-op
	// when full_attn_interval is absent or zero; the per-block driver then
	// falls back to a single block covering all layers.
	m.rlb.InitBlockRanges(nLayers, b.params.Ints["full_attn_interval"])

	blockCounts := make(map[string]int)
	for _, name := range m.LayerBlockNames {
		blockCounts[name]++
	}
	var blockSummary string
	for name, count := range blockCounts {
		if blockSummary != "" {
			blockSummary += " + "
		}
		blockSummary += fmt.Sprintf("%d %s", count, name)
	}
	log.Info("%s: %d layers (%s), n_embd=%d, n_heads=%d/%d, head_dim=%d",
		b.def.Architecture.Name, nLayers, blockSummary,
		b.params.Ints["n_embd"], b.params.Ints["n_heads"], b.params.Ints["n_kv_heads"], b.params.Ints["head_dim"])
	return nil
}

// createComputeResources allocates the cached forward-pass graph context,
// scheduler, and scratch buffers reused across decode tokens.
func (b *genericModelBuilder) createComputeResources() error {
	b.m.cachedCtx = ggml.NewGraphContext(graphCtxSize(), ggml.AllocPermDisallow)
	if b.m.cachedCtx == nil {
		return fmt.Errorf("failed to create cached graph context")
	}
	b.m.cachedSched = ggml.NewSched(b.store.GPU, b.store.CPU, maxGraphNodes)
	if b.m.cachedSched == nil {
		return fmt.Errorf("failed to create cached scheduler")
	}
	b.m.ffnScratch = make(map[string]ggml.Tensor)
	b.m.logitBuf = make([]float32, b.params.Ints["n_vocab"])
	return nil
}

// MemoryStats returns the model's current memory allocation.
func (m *GenericModel) MemoryStats() (weightBytes uint64, cacheBytes uint64) {
	if m.Store == nil || m.Store.Buffer == nil {
		return 0, 0
	}
	// WeightStore.Buffer.Size() returns the total allocated GPU buffer.
	bufSize := m.Store.Buffer.Size()

	// We don't have a precise split between weights and cache in the buffer,
	// but we can estimate: weight size is the sum of all tensor specs.
	// For now, the whole buffer is reported as allocated.
	weightBytes = uint64(bufSize)
	cacheBytes = 0 // Cache is included in the same buffer; no separate accounting.
	return weightBytes, cacheBytes
}

// layerHasAltFFN checks if the alt FFN weights actually exist in the GGUF for this layer.
// Skips alt weights whose GGUF tensor name also appears in standard FFN, block, or common
// weights — those exist in both dense and MoE models and can't distinguish them.
func layerHasAltFFN(store *WeightStore, lw ResolvedLayerWeights) bool {
	// Collect GGUF tensor names used by non-alt weight sources.
	shared := make(map[string]struct{})
	for _, tn := range lw.Common {
		shared[tn] = struct{}{}
	}
	for _, tn := range lw.Block {
		shared[tn] = struct{}{}
	}
	for _, tn := range lw.FFN {
		shared[tn] = struct{}{}
	}

	lt := store.layers[lw.Index]
	for logicalName, tensorName := range lw.FFNAlt {
		if _, ok := shared[tensorName]; ok {
			continue
		}
		if t := lt["ffn_"+logicalName]; !t.IsNil() {
			return true
		}
	}
	return false
}

// Close releases all GPU resources.
func (m *GenericModel) Close() {
	if m.cachedSched != nil {
		m.cachedSched.Free()
		m.cachedSched = nil
	}
	if m.cachedCtx != nil {
		m.cachedCtx.Free()
		m.cachedCtx = nil
	}
	m.rlb.Free()
	if m.Store != nil {
		m.Store.Close()
		m.Store = nil
	}
}

// --- Pure-Go GGUF reader types ---

// tensorSpec holds tensor metadata extracted from gguf-parser-go,
// replacing the C shape context. Dimensions are padded to 4 elements (unused = 1).
type tensorSpec struct {
	Type ggml.GGMLType // ggml type (matches ggml.NewTensor*D)
	Ne   [4]int64      // dims; unused = 1
	Size int           // total bytes
}

// goGGUFReader implements GGUFReader using gguf-parser-go metadata.
type goGGUFReader struct {
	kvs         ggufparser.GGUFMetadataKVs
	tensorSpecs map[string]tensorSpec
}

func (r *goGGUFReader) GetU32(key string) (uint32, bool) {
	kv, ok := r.kvs.Get(key)
	if !ok {
		return 0, false
	}
	if kv.ValueType != ggufparser.GGUFMetadataValueTypeUint32 {
		return 0, false
	}
	return kv.ValueUint32(), true
}

func (r *goGGUFReader) GetF32(key string) (float32, bool) {
	kv, ok := r.kvs.Get(key)
	if !ok {
		return 0, false
	}
	if kv.ValueType != ggufparser.GGUFMetadataValueTypeFloat32 {
		return 0, false
	}
	return kv.ValueFloat32(), true
}

func (r *goGGUFReader) GetArrInts(key string) ([]int, bool) {
	kv, ok := r.kvs.Get(key)
	if !ok {
		return nil, false
	}
	if kv.ValueType != ggufparser.GGUFMetadataValueTypeArray {
		return nil, false
	}
	av := kv.ValueArray()
	switch av.Type {
	case ggufparser.GGUFMetadataValueTypeInt32:
		raw := av.ValuesInt32()
		arr := make([]int, len(raw))
		for i, v := range raw {
			arr[i] = int(v)
		}
		return arr, len(arr) > 0
	case ggufparser.GGUFMetadataValueTypeUint32:
		raw := av.ValuesUint32()
		arr := make([]int, len(raw))
		for i, v := range raw {
			arr[i] = int(v)
		}
		return arr, len(arr) > 0
	default:
		return nil, false
	}
}

func (r *goGGUFReader) GetArrBools(key string) ([]bool, bool) {
	kv, ok := r.kvs.Get(key)
	if !ok {
		return nil, false
	}
	if kv.ValueType != ggufparser.GGUFMetadataValueTypeArray {
		return nil, false
	}
	av := kv.ValueArray()
	if av.Type != ggufparser.GGUFMetadataValueTypeBool {
		return nil, false
	}
	arr := av.ValuesBool()
	return arr, len(arr) > 0
}

func (r *goGGUFReader) GetTensorDim(name string, dim int) (int64, bool) {
	spec, ok := r.tensorSpecs[name+".weight"]
	if !ok {
		spec, ok = r.tensorSpecs[name]
	}
	if !ok {
		return 0, false
	}
	if dim < 0 || dim > 3 {
		return 0, false
	}
	return spec.Ne[dim], true
}

// buildTensorSpecs constructs a map of tensor name → tensorSpec from a parsed GGUF file.
// Dimensions are padded to 4 elements; missing trailing dimensions default to 1.
func buildTensorSpecs(f *ggufparser.GGUFFile) map[string]tensorSpec {
	specs := make(map[string]tensorSpec, len(f.TensorInfos))
	for _, ti := range f.TensorInfos {
		var ne [4]int64
		ne[0], ne[1], ne[2], ne[3] = 1, 1, 1, 1
		for d := 0; d < len(ti.Dimensions) && d < 4; d++ {
			ne[d] = int64(ti.Dimensions[d])
		}
		specs[ti.Name] = tensorSpec{
			Type: ggml.GGMLType(ti.Type),
			Ne:   ne,
			Size: int(ti.Bytes()),
		}
	}
	return specs
}
