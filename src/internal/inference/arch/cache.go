package arch

import (
	"fmt"
	"log"
	"strconv"
	"strings"

	ggml "inference-lab-bench/internal/inference/ggml"
)

// GenericCache holds per-layer cache tensors allocated on Metal.
type GenericCache struct {
	Model     *GenericModel
	SeqPos    int
	MaxSeqLen int
	Layers    []LayerCache // one per model layer
	cacheCtx  *ggml.GraphContext
	cacheBuf  *ggml.Buffer
}

// NewGenericCache creates a cache from the architecture definition.
func (m *GenericModel) NewCache(maxSeqLen int) (*GenericCache, error) {
	nLayers := m.Params.Ints["n_layers"]

	// Count tensors needed
	nTensors := 0
	for i := range nLayers {
		blockName := m.LayerBlockNames[i]
		block := m.Def.Blocks[blockName]
		nTensors += len(block.Cache)
	}

	ctxSize := nTensors*ggml.TensorOverhead() + 1024
	cacheCtx := ggml.NewGraphContext(ctxSize)
	if cacheCtx == nil {
		return nil, fmt.Errorf("failed to create cache context")
	}

	gc := &GenericCache{
		Model:     m,
		MaxSeqLen: maxSeqLen,
		Layers:    make([]LayerCache, nLayers),
	}

	for i := range nLayers {
		blockName := m.LayerBlockNames[i]
		block := m.Def.Blocks[blockName]
		lc := LayerCache{
			Tensors:   make(map[string]ggml.Tensor, len(block.Cache)),
			MaxSeqLen: maxSeqLen,
		}
		for cacheName, spec := range block.Cache {
			dims, err := resolveCacheDims(spec.Dims, m.Params, maxSeqLen)
			if err != nil {
				cacheCtx.Free()
				return nil, fmt.Errorf("layer %d cache %q: %w", i, cacheName, err)
			}
			dtype := ggml.TypeF32 // only F32 cache supported for now
			_ = spec.Dtype

			var t ggml.Tensor
			switch len(dims) {
			case 2:
				t = ggml.NewTensor2D(cacheCtx, dtype, dims[0], dims[1])
			case 3:
				t = ggml.NewTensor3D(cacheCtx, dtype, dims[0], dims[1], dims[2])
			case 4:
				t = ggml.NewTensor4D(cacheCtx, dtype, dims[0], dims[1], dims[2], dims[3])
			default:
				cacheCtx.Free()
				return nil, fmt.Errorf("layer %d cache %q: unsupported %d dims", i, cacheName, len(dims))
			}
			lc.Tensors[cacheName] = t
		}
		gc.Layers[i] = lc
	}

	cacheBuf := ggml.AllocCtxTensors(cacheCtx, m.Store.GPU)
	if cacheBuf == nil {
		cacheCtx.Free()
		return nil, fmt.Errorf("failed to alloc cache GPU VRAM")
	}

	gc.cacheCtx = cacheCtx
	gc.cacheBuf = cacheBuf
	gc.Clear()

	log.Printf("[DBG] cache: %d seq max, %.1f MB GPU VRAM",
		maxSeqLen, float64(cacheBuf.Size())/(1024*1024))

	return gc, nil
}

// Clear resets the cache to empty.
func (gc *GenericCache) Clear() {
	gc.SeqPos = 0
	for _, lc := range gc.Layers {
		for _, t := range lc.Tensors {
			if !t.IsNil() {
				zeros := make([]byte, t.Nbytes())
				ggml.TensorSetBytes(t, zeros, 0)
			}
		}
	}
}

// Free releases cache GPU VRAM.
func (gc *GenericCache) Free() {
	if gc.cacheBuf != nil {
		gc.cacheBuf.Free()
		gc.cacheBuf = nil
	}
	if gc.cacheCtx != nil {
		gc.cacheCtx.Free()
		gc.cacheCtx = nil
	}
}

// resolveCacheDims evaluates cache dimension expressions.
// Supports: param names, "max_seq_len", integer literals, simple arithmetic (e.g. "ssm_d_conv - 1").
func resolveCacheDims(dimExprs []string, params *ResolvedParams, maxSeqLen int) ([]int64, error) {
	dims := make([]int64, len(dimExprs))
	for i, expr := range dimExprs {
		v, err := evalCacheDim(expr, params, maxSeqLen)
		if err != nil {
			return nil, fmt.Errorf("dim %d (%q): %w", i, expr, err)
		}
		dims[i] = int64(v)
	}
	return dims, nil
}

func evalCacheDim(expr string, params *ResolvedParams, maxSeqLen int) (int, error) {
	expr = strings.TrimSpace(expr)

	// Special token
	if expr == "max_seq_len" {
		return maxSeqLen, nil
	}

	// Direct param lookup
	if v, ok := params.Ints[expr]; ok {
		return v, nil
	}

	// Integer literal
	if v, err := strconv.Atoi(expr); err == nil {
		return v, nil
	}

	// Simple arithmetic expression (reuse the expression evaluator).
	// Use a shallow copy so we don't mutate the shared ResolvedParams map.
	pCopy := *params
	pCopy.Ints = make(map[string]int, len(params.Ints)+1)
	for k, v := range params.Ints {
		pCopy.Ints[k] = v
	}
	pCopy.Ints["max_seq_len"] = maxSeqLen
	v, err := evalExpr(expr, &pCopy)
	if err != nil {
		return 0, err
	}
	return v, nil
}
