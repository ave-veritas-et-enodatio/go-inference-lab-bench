package arch

/*
#cgo CFLAGS: -std=c17 -I${SRCDIR}/../../../ggml_lib/src
#include "ggml_ops.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"os"
	"unsafe"

	"inference-lab-bench/internal/inference/ggml"
)

// GenericModel is a model loaded from a GGUF file using an architecture definition.
type GenericModel struct {
	Def             *ArchDef
	Params          *ResolvedParams
	Weights         *ResolvedWeights
	Store           *WeightStore   // immutable weight storage (read-only after load)
	LayerBlockNames []string       // which block each layer uses
	BlockBuilders   []BlockBuilder // per-layer block builder
	FFNBuilders     []FFNBuilder   // per-layer FFN builder

	// Canonical module map and tensor dims for visualization.
	CanonicalModuleMap *ModuleMap
	TensorDims         TensorDimsMap
	ModelPath          string
}

// ggufReader wraps C GGUF access to implement the GGUFReader interface.
type ggufReader struct {
	gf          C.ggml_go_gguf
	tensorIndex map[string]ggml.Tensor
}

func (r *ggufReader) findKey(key string) (C.int64_t, int, bool) {
	ck := C.CString(key)
	defer C.free(unsafe.Pointer(ck))
	idx := C.ggml_go_gguf_find_key(r.gf, ck)
	if idx < 0 {
		return 0, 0, false
	}
	kvType := int(C.ggml_go_gguf_get_kv_type(r.gf, idx))
	return idx, kvType, true
}

func (r *ggufReader) GetU32(key string) (uint32, bool) {
	idx, kvType, ok := r.findKey(key)
	if !ok || kvType != int(C.GGML_GO_GGUF_TYPE_UINT32) {
		return 0, false
	}
	return uint32(C.ggml_go_gguf_get_u32(r.gf, idx)), true
}

func (r *ggufReader) GetF32(key string) (float32, bool) {
	idx, kvType, ok := r.findKey(key)
	if !ok || kvType != int(C.GGML_GO_GGUF_TYPE_FLOAT32) {
		return 0, false
	}
	return float32(C.ggml_go_gguf_get_f32(r.gf, idx)), true
}

func (r *ggufReader) GetArrInts(key string) ([]int, bool) {
	idx, kvType, ok := r.findKey(key)
	if !ok || kvType != int(C.GGML_GO_GGUF_TYPE_ARRAY) {
		return nil, false
	}
	n := int(C.ggml_go_gguf_get_arr_n(r.gf, idx))
	if n == 0 {
		return nil, false
	}
	data := C.ggml_go_gguf_get_arr_data(r.gf, idx)
	arr := make([]int, n)
	raw := unsafe.Slice((*int32)(data), n)
	for i := range n {
		arr[i] = int(raw[i])
	}
	return arr, true
}

func (r *ggufReader) GetTensorDim(name string, dim int) (int64, bool) {
	t, ok := r.tensorIndex[name+".weight"]
	if !ok {
		t, ok = r.tensorIndex[name]
	}
	if !ok {
		return 0, false
	}
	return t.Ne(dim), true
}

// NewGenericModel loads a GGUF model using the named architecture definition.
// archDir is the directory containing architecture definition TOML files.
func NewGenericModel(archName, modelPath, archDir string) (*GenericModel, error) {
	def, err := Load(archDir, archName)
	if err != nil {
		return nil, fmt.Errorf("loading arch def %q: %w", archName, err)
	}

	cpath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cpath))

	// 1. Open GGUF — single handle provides both KV metadata and tensor shapes.
	var shapeCtxPtr C.ggml_go_context
	gf := C.ggml_go_gguf_init(cpath, &shapeCtxPtr)
	if gf == nil || shapeCtxPtr == nil {
		return nil, fmt.Errorf("failed to open GGUF: %s", modelPath)
	}

	shapeCtx := &ggml.GraphContext{}
	shapeCtx.SetPtr(unsafe.Pointer(shapeCtxPtr))

	tensorIndex := buildTensorIndex(shapeCtx)

	reader := &ggufReader{gf: gf, tensorIndex: tensorIndex}

	// 2. Resolve params
	params, err := ResolveParams(def, reader)
	if err != nil {
		C.ggml_go_gguf_free(gf)
		shapeCtx.Free()
		return nil, fmt.Errorf("resolving params: %w", err)
	}

	// 3. Resolve weights
	weights, err := ResolveWeights(def, params)
	if err != nil {
		C.ggml_go_gguf_free(gf)
		shapeCtx.Free()
		return nil, fmt.Errorf("resolving weights: %w", err)
	}

	// 4. Build canonical module map and tensor dims for visualization.
	canonicalMM := BuildModuleMap(weights)
	tensorDims := BuildTensorDimsMap(weights, func(name string) (int64, int64, int64, bool) {
		if t, ok := tensorIndex[name]; ok {
			return t.Ne(0), t.Ne(1), int64(t.Nbytes()), true
		}
		return 0, 0, 0, false
	}, params.Ints["head_dim"])

	// 5. Init backends
	metal := ggml.MetalInit()
	if metal == nil {
		C.ggml_go_gguf_free(gf)
		shapeCtx.Free()
		return nil, fmt.Errorf("failed to init GPU backend")
	}
	cpu := ggml.CPUInit()

	// 6. Build tensor context: all tensors loaded into VRAM.
	nTensors := int64(C.ggml_go_gguf_n_tensors(gf))

	ctxSize := int64(ggml.TensorOverhead())*nTensors + 1*1024*1024
	weightCtx := ggml.NewGraphContext(int(ctxSize))
	if weightCtx == nil {
		C.ggml_go_gguf_free(gf)
		shapeCtx.Free()
		metal.Free()
		cpu.Free()
		return nil, fmt.Errorf("failed to create weight context")
	}

	weightTensorIndex := make(map[string]ggml.Tensor, nTensors)
	for ti := int64(0); ti < nTensors; ti++ {
		tname := C.GoString(C.ggml_go_gguf_tensor_name(gf, C.int64_t(ti)))
		src, ok := tensorIndex[tname]
		if !ok {
			continue
		}
		t := makeSameTypeTensor(weightCtx, src)
		ggml.SetName(t, tname)
		weightTensorIndex[tname] = t
	}

	// shapeCtx only held shapes for weightTensorIndex construction; no longer needed.
	shapeCtx.Free()

	// 7. Allocate all tensors on GPU
	weightBuf := ggml.AllocCtxTensors(weightCtx, metal)
	if weightBuf == nil {
		C.ggml_go_gguf_free(gf)
		weightCtx.Free()
		metal.Free()
		cpu.Free()
		return nil, fmt.Errorf("failed to alloc GPU VRAM")
	}

	// 8. Load tensor data from file
	f, ferr := os.Open(modelPath)
	if ferr != nil {
		C.ggml_go_gguf_free(gf)
		weightBuf.Free()
		weightCtx.Free()
		metal.Free()
		cpu.Free()
		return nil, fmt.Errorf("failed to open model file: %w", ferr)
	}
	defer f.Close()

	dataOffset := int64(C.ggml_go_gguf_data_offset(gf))
	for ti := int64(0); ti < nTensors; ti++ {
		tname := C.GoString(C.ggml_go_gguf_tensor_name(gf, C.int64_t(ti)))
		toff := int64(C.ggml_go_gguf_tensor_offset(gf, C.int64_t(ti)))
		tsz := int(C.ggml_go_gguf_tensor_size(gf, C.int64_t(ti)))

		t, ok := weightTensorIndex[tname]
		if !ok {
			continue
		}
		buf := make([]byte, tsz)
		if _, err := f.ReadAt(buf, dataOffset+toff); err != nil {
			C.ggml_go_gguf_free(gf)
			return nil, fmt.Errorf("read tensor %s: %w", tname, err)
		}
		ggml.TensorSetBytes(t, buf, 0)
	}
	C.ggml_go_gguf_free(gf)

	// 9. Build weight store.
	store := &WeightStore{
		global: make(map[string]ggml.Tensor),
		ctx:    weightCtx,
		buf:    weightBuf,
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

		for logicalName, tensorName := range lw.Block {
			t, ok := weightTensorIndex[tensorName]
			if !ok {
				t = ggml.NilTensor()
			}
			lt[logicalName] = t
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
	for i, lw := range weights.Layers {
		// Use alt FFN if this layer has alt weights in the GGUF
		if fbAlt != nil && len(lw.FFNAlt) > 0 && layerHasAltFFN(store, lw) {
			m.FFNBuilders[i] = fbAlt
		} else {
			m.FFNBuilders[i] = fb
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
	fmt.Fprintf(os.Stderr, "[INF] %s: %d layers (%s), n_embd=%d, n_heads=%d/%d, head_dim=%d\n",
		def.Architecture.Name, nLayers, blockSummary,
		params.Ints["n_embd"], params.Ints["n_heads"], params.Ints["n_kv_heads"], params.Ints["head_dim"])

	// Store canonical module map and tensor dims for visualization.
	m.CanonicalModuleMap = canonicalMM
	m.TensorDims = tensorDims
	m.ModelPath = modelPath

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

	cpath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cpath))

	var shapeCtxPtr C.ggml_go_context
	gf := C.ggml_go_gguf_init(cpath, &shapeCtxPtr)
	if gf == nil || shapeCtxPtr == nil {
		return nil, fmt.Errorf("failed to open GGUF: %s", modelPath)
	}
	defer C.ggml_go_gguf_free(gf)
	shapeCtx := &ggml.GraphContext{}
	shapeCtx.SetPtr(unsafe.Pointer(shapeCtxPtr))
	defer shapeCtx.Free()

	tensorIndex := buildTensorIndex(shapeCtx)
	reader := &ggufReader{gf: gf, tensorIndex: tensorIndex}

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
func layerHasAltFFN(store *WeightStore, lw ResolvedLayerWeights) bool {
	lt := store.layers[lw.Index]
	for logicalName := range lw.FFNAlt {
		if t := lt["ffn_"+logicalName]; !t.IsNil() {
			return true
		}
	}
	return false
}

// Close releases all GPU resources.
func (m *GenericModel) Close() {
	if m.Store != nil {
		m.Store.Close()
		m.Store = nil
	}
}

// makeSameTypeTensor creates a new graph-context tensor with the same quantized type and shape
// as src.
func makeSameTypeTensor(gctx *ggml.GraphContext, src ggml.Tensor) ggml.Tensor {
	typ := ggml.TensorType(src)
	ne0, ne1, ne2, ne3 := src.Ne(0), src.Ne(1), src.Ne(2), src.Ne(3)
	if ne3 > 1 {
		return ggml.NewTensor4D(gctx, typ, ne0, ne1, ne2, ne3)
	} else if ne2 > 1 {
		return ggml.NewTensor3D(gctx, typ, ne0, ne1, ne2)
	} else if ne1 > 1 {
		return ggml.NewTensor2D(gctx, typ, ne0, ne1)
	}
	return ggml.NewTensor1D(gctx, typ, ne0)
}

func buildTensorIndex(ctx *ggml.GraphContext) map[string]ggml.Tensor {
	idx := make(map[string]ggml.Tensor)
	for t := ggml.GetFirstTensor(ctx); !t.IsNil(); t = ggml.GetNextTensor(ctx, t) {
		name := ggml.TensorName(t)
		idx[name] = t
	}
	return idx
}
