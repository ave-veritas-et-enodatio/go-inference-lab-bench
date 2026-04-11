package arch

import (
	"fmt"
	"os"

	ggufparser "github.com/gpustack/gguf-parser-go"

	"inference-lab-bench/internal/ggml"
	log "inference-lab-bench/internal/log"
)

// GenericModel is a model loaded from a GGUF file using an architecture definition.
type GenericModel struct {
	Def             *ArchDef
	Params          *ResolvedParams
	Weights         *ResolvedWeights
	Store           *WeightStore               // immutable weight storage (read-only after load)
	LayerBlockNames []string                    // which block each layer uses
	BlockBuilders   []BlockBuilder              // per-layer block builder
	FFNBuilders     []FFNBuilder                // per-layer FFN builder
	FFNConfigs      []map[string]any            // per-layer FFN config (from [ffn.config] / [ffn_alt.config])

	// Canonical module map and supporting data for per-query culling.
	CanonicalModuleMap *ModuleMap
	HeadDim            int           // elements per attention head (for flash attention geometry check)
	TensorDims         TensorDimsMap
	ModelPath          string

	// Persistent compute resources for ForwardCached. Created at model load, reused across tokens.
	cachedCtx   *ggml.GraphContext
	cachedSched *ggml.Sched

	// Pre-allocated scratch buffers for the hot decode path (ForwardCached).
	ffnScratch map[string]ggml.Tensor // reused by buildFFNBlock each layer
	logitBuf   []float32              // reused by readLogitsInto each token
}

// NewGenericModel loads a GGUF model using the named architecture definition.
// archDir is the directory containing architecture definition TOML files.
// If gf is non-nil it is used directly; otherwise the GGUF is parsed from modelPath.
func NewGenericModel(archName, modelPath, archDir string, gf *ggufparser.GGUFFile) (*GenericModel, error) {
	def, err := Load(archDir, archName)
	if err != nil {
		return nil, fmt.Errorf("loading arch def %q: %w", archName, err)
	}

	// 1. Parse GGUF with pure-Go parser — provides KV metadata and tensor specs.
	if gf == nil {
		var parseErr error
		gf, parseErr = ggufparser.ParseGGUFFile(modelPath)
		if parseErr != nil {
			return nil, fmt.Errorf("failed to parse GGUF: %w", parseErr)
		}
	}
	if gf.TensorDataStartOffset == 0 && len(gf.TensorInfos) > 0 {
		return nil, fmt.Errorf("split GGUF files are not supported (use a single-file model)")
	}
	tensorSpecs := buildTensorSpecs(gf)
	reader := &goGGUFReader{kvs: gf.Header.MetadataKV, tensorSpecs: tensorSpecs}

	// 2. Resolve params
	params, err := ResolveParams(def, reader)
	if err != nil {
		return nil, fmt.Errorf("resolving params: %w", err)
	}

	// 3. Resolve weights
	weights, err := ResolveWeights(def, params)
	if err != nil {
		return nil, fmt.Errorf("resolving weights: %w", err)
	}

	// 4. Build canonical module map and supporting data for per-query culling.
	headDim := params.Ints["head_dim"]
	canonicalMM := BuildModuleMap(weights)
	tensorDims := BuildTensorDimsMap(weights, func(name string) (int64, int64, int64, bool) {
		if s, ok := tensorSpecs[name]; ok {
			return s.Ne[0], s.Ne[1], int64(s.Size), true
		}
		return 0, 0, 0, false
	}, params.Ints["head_dim"])

	// 5. Init backends
	metal := ggml.MetalInit()
	if metal == nil {
		return nil, fmt.Errorf("failed to init GPU backend")
	}
	cpu := ggml.CPUInit()

	// 6. Build tensor context: all tensors loaded into VRAM (culling is per-query, not load-time).
	nTensors := int64(len(gf.TensorInfos))

	ctxSize := int64(ggml.TensorOverhead())*nTensors + 1*1024*1024
	weightCtx := ggml.NewGraphContext(int(ctxSize))
	if weightCtx == nil {
		metal.Free()
		cpu.Free()
		return nil, fmt.Errorf("failed to create weight context")
	}

	weightTensorIndex := make(map[string]ggml.Tensor, nTensors)
	for _, ti := range gf.TensorInfos {
		spec := tensorSpecs[ti.Name]
		t := makeTensorFromSpec(weightCtx, spec.Type, spec.Ne[0], spec.Ne[1], spec.Ne[2], spec.Ne[3])
		ggml.SetName(t, ti.Name)
		weightTensorIndex[ti.Name] = t
	}

	// 7. Allocate all tensors on GPU
	weightBuf := ggml.AllocCtxTensors(weightCtx, metal)
	if weightBuf == nil {
		weightCtx.Free()
		metal.Free()
		cpu.Free()
		return nil, fmt.Errorf("failed to alloc GPU VRAM")
	}

	// 8. Load tensor data from file
	f, ferr := os.Open(modelPath)
	if ferr != nil {
		weightBuf.Free()
		weightCtx.Free()
		metal.Free()
		cpu.Free()
		return nil, fmt.Errorf("failed to open model file: %w", ferr)
	}
	defer f.Close()

	dataOffset := gf.TensorDataStartOffset
	for _, ti := range gf.TensorInfos {
		t, ok := weightTensorIndex[ti.Name]
		if !ok {
			continue
		}
		size := int(ti.Bytes())
		buf := make([]byte, size)
		if _, err := f.ReadAt(buf, dataOffset+int64(ti.Offset)); err != nil {
			return nil, fmt.Errorf("read tensor %s: %w", ti.Name, err)
		}
		ggml.TensorSetBytes(t, buf, 0)
	}

	// 9. Build weight store — all tensors present (culling is per-query via MaskedLayer).
	store := &WeightStore{
		global: make(map[string]ggml.Tensor),
		ctx:    weightCtx,
		Buffer: weightBuf,
		GPU:    metal,
		CPU:    cpu,
	}

	// Resolve global tensors
	for logicalName, tensorName := range weights.Global {
		t, ok := weightTensorIndex[tensorName]
		if !ok {
			t = ggml.NilTensor()
		}
		store.global[logicalName] = t
	}

	// Handle tied embeddings
	if def.Architecture.TiedEmbeddings && store.Global("output").IsNil() {
		store.global["output"] = store.global["token_embd"]
	}

	if store.Global("token_embd").IsNil() || store.Global("output_norm").IsNil() {
		store.Close()
		return nil, fmt.Errorf("missing global tensors")
	}

	// Resolve per-layer tensors
	nLayers := params.Ints["n_layers"]
	store.layers = make([]map[string]ggml.Tensor, nLayers)

	for i, lw := range weights.Layers {
		lt := make(map[string]ggml.Tensor)

		for logicalName, tensorName := range lw.Common {
			t, ok := weightTensorIndex[tensorName]
			if !ok {
				t = ggml.NilTensor()
			}
			lt[logicalName] = t
		}

		blockDef := def.Blocks[lw.BlockName]
		for logicalName, tensorName := range lw.Block {
			t, ok := weightTensorIndex[tensorName]
			if !ok {
				// Fallback: try the raw suffix as a global tensor name (e.g. rope_freqs.weight)
				if suffix, has := blockDef.Weights[logicalName]; has {
					t, ok = weightTensorIndex[suffix]
				}
			}
			if !ok {
				t = ggml.NilTensor()
			}
			lt[logicalName] = t
		}

		// n_kv_shared_layers: non-KV layers still have weights in the GGUF but
		// must NOT compute their own K/V or write to cache. Null out K/V weights
		// so the builder's hasKV check correctly identifies them as non-KV.
		if nKVShared, ok := params.Ints["n_kv_shared_layers"]; ok && nKVShared > 0 {
			nKVFromStart := nLayers - nKVShared
			if i >= nKVFromStart {
				lt["attn_k"] = ggml.NilTensor()
				lt["attn_v"] = ggml.NilTensor()
			}
		}

		for logicalName, tensorName := range lw.FFN {
			t, ok := weightTensorIndex[tensorName]
			if !ok {
				t = ggml.NilTensor()
			}
			lt["ffn_"+logicalName] = t
		}

		// FFN alt weights (also prefixed with ffn_)
		for logicalName, tensorName := range lw.FFNAlt {
			t, ok := weightTensorIndex[tensorName]
			if !ok {
				t = ggml.NilTensor()
			}
			lt["ffn_"+logicalName] = t
		}

		store.layers[i] = lt
	}

	// 10. Build model struct
	m := &GenericModel{
		Def:     def,
		Params:  params,
		Weights: weights,
		Store:   store,
	}

	m.LayerBlockNames = make([]string, nLayers)
	m.BlockBuilders = make([]BlockBuilder, nLayers)

	for i, lw := range weights.Layers {
		m.LayerBlockNames[i] = lw.BlockName
		bb, ok := GetBlockBuilder(def.Blocks[lw.BlockName].Builder)
		if !ok {
			store.Close()
			return nil, fmt.Errorf("layer %d: unknown block builder %q", i, def.Blocks[lw.BlockName].Builder)
		}
		m.BlockBuilders[i] = bb
	}

	// Resolve per-layer FFN builders
	fb, ok := GetFFNBuilder(def.FFN.Builder)
	if !ok {
		store.Close()
		return nil, fmt.Errorf("unknown FFN builder %q", def.FFN.Builder)
	}
	var fbAlt FFNBuilder
	if def.FFNAlt != nil {
		fbAlt, ok = GetFFNBuilder(def.FFNAlt.Builder)
		if !ok {
			store.Close()
			return nil, fmt.Errorf("unknown FFN alt builder %q", def.FFNAlt.Builder)
		}
	}
	m.FFNBuilders = make([]FFNBuilder, nLayers)
	m.FFNConfigs = make([]map[string]any, nLayers)
	for i, lw := range weights.Layers {
		// Use alt FFN if this layer has alt weights in the GGUF
		if fbAlt != nil && len(lw.FFNAlt) > 0 && layerHasAltFFN(store, lw) {
			m.FFNBuilders[i] = fbAlt
			m.FFNConfigs[i] = def.FFNAlt.Config
		} else {
			m.FFNBuilders[i] = fb
			m.FFNConfigs[i] = def.FFN.Config
		}
	}

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
		def.Architecture.Name, nLayers, blockSummary,
		params.Ints["n_embd"], params.Ints["n_heads"], params.Ints["n_kv_heads"], params.Ints["head_dim"])

	// Store canonical module map and supporting data for per-query culling.
	m.CanonicalModuleMap = canonicalMM
	m.HeadDim = headDim
	m.TensorDims = tensorDims
	m.ModelPath = modelPath

	m.cachedCtx = ggml.NewGraphContext(graphCtxSize(nLayers))
	if m.cachedCtx == nil {
		store.Close()
		return nil, fmt.Errorf("failed to create cached graph context")
	}
	m.cachedSched = ggml.NewSched(store.GPU, store.CPU, 16384)
	if m.cachedSched == nil {
		m.cachedCtx.Free()
		store.Close()
		return nil, fmt.Errorf("failed to create cached scheduler")
	}

	m.ffnScratch = make(map[string]ggml.Tensor)
	m.logitBuf = make([]float32, params.Ints["n_vocab"])

	return m, nil
}

// ResolveWeightLayout loads an architecture definition and resolves the complete weight
// name mapping for a model file. Reads GGUF metadata but does not allocate GPU memory.
// Useful for tools (e.g. gen-modulemap) that need the weight structure without loading the model.
func ResolveWeightLayout(archName, modelPath, archDir string) (*ResolvedWeights, error) {
	def, err := Load(archDir, archName)
	if err != nil {
		return nil, fmt.Errorf("loading arch def %q: %w", archName, err)
	}

	gf, err := ggufparser.ParseGGUFFile(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to parse GGUF: %w", err)
	}
	tensorSpecs := buildTensorSpecs(gf)
	reader := &goGGUFReader{kvs: gf.Header.MetadataKV, tensorSpecs: tensorSpecs}

	params, err := ResolveParams(def, reader)
	if err != nil {
		return nil, fmt.Errorf("resolving params: %w", err)
	}

	weights, err := ResolveWeights(def, params)
	if err != nil {
		return nil, fmt.Errorf("resolving weights: %w", err)
	}

	return weights, nil
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
	if m.Store != nil {
		m.Store.Close()
		m.Store = nil
	}
}

// --- Pure-Go GGUF reader types ---

// tensorSpec holds tensor metadata extracted from gguf-parser-go,
// replacing the C shape context. Dimensions are padded to 4 elements (unused = 1).
type tensorSpec struct {
	Type int      // ggml type int (matches ggml.NewTensor*D)
	Ne   [4]int64 // dims; unused = 1
	Size int      // total bytes
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
			Type: int(ti.Type),
			Ne:   ne,
			Size: int(ti.Bytes()),
		}
	}
	return specs
}

