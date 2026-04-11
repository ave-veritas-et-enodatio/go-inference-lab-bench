package arch

import (
	ggml "inference-lab-bench/internal/ggml"
)

// WeightStore holds immutable model weights in GPU VRAM.
// After construction, weight tensors are never written — all access is read-only.
type WeightStore struct {
	global map[string]ggml.Tensor   // logical name → tensor (token_embd, output_norm, output)
	layers []map[string]ggml.Tensor // per-layer: all weights (common + block + ffn)
	ctx    *ggml.GraphContext       // allocation context (owns tensor metadata)
	Buffer *ggml.Buffer             // GPU VRAM buffer (owns tensor data)
	GPU    *ggml.Backend
	CPU    *ggml.Backend
}

// Global returns a global weight tensor by logical name.
func (ws *WeightStore) Global(name string) ggml.Tensor {
	if t, ok := ws.global[name]; ok {
		return t
	}
	return ggml.NilTensor()
}

// Layer returns the weight tensor map for a given layer index.
func (ws *WeightStore) Layer(idx int) map[string]ggml.Tensor {
	return ws.layers[idx]
}

// NLayers returns the number of layers.
func (ws *WeightStore) NLayers() int {
	return len(ws.layers)
}

// Close releases all GPU VRAM and metadata.
func (ws *WeightStore) Close() {
	if ws.Buffer != nil {
		ws.Buffer.Free()
		ws.Buffer = nil
	}
	if ws.ctx != nil {
		ws.ctx.Free()
		ws.ctx = nil
	}
	// Backends must be freed last — the buffer depends on them.
	if ws.GPU != nil {
		ws.GPU.Free()
		ws.GPU = nil
	}
	if ws.CPU != nil {
		ws.CPU.Free()
		ws.CPU = nil
	}
}
