package arch

import (
	ggml "inference-lab-bench/internal/ggml"
)

// LayerTensors holds per-layer weight tensors separated by role.
// Common weights (norms, scaling) are shared across block types; Block weights are
// attention/SSM-specific; FFN weights are feed-forward-specific (dense or MoE).
type LayerTensors struct {
	Common map[string]ggml.Tensor // attn_norm, ffn_norm, attn_post_norm, etc.
	Block  map[string]ggml.Tensor // attn_q, attn_k, attn_v, attn_output, etc.
	FFN    map[string]ggml.Tensor // gate, up, down (dense) or gate_inp, gate_exps, etc. (MoE)
}

// WeightStore holds immutable model weights in GPU VRAM.
// After construction, weight tensors are never written — all access is read-only.
type WeightStore struct {
	global map[string]ggml.Tensor // logical name → tensor (token_embd, output_norm, output)
	layers []LayerTensors         // per-layer weights separated by role
	ctx    *ggml.GraphContext     // allocation context (owns tensor metadata)
	Buffer *ggml.Buffer           // GPU VRAM buffer (owns tensor data)
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

// Layer returns the per-layer weight tensors for a given layer index.
// Returns a pointer to the layer's tensors, or nil for out-of-bounds indices.
func (ws *WeightStore) Layer(idx int) *LayerTensors {
	if idx < 0 || idx >= len(ws.layers) {
		return nil
	}
	return &ws.layers[idx]
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
