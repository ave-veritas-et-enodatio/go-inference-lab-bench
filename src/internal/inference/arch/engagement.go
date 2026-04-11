package arch

import (
	"math"
	"unsafe"

	ggml "inference-lab-bench/internal/ggml"
)

// EngagementData holds per-layer cosine similarity measurements between the
// residual stream before and after each major operation. Unconditionally
// populated by ForwardStateless — the overhead is ~10 reduction ops per layer
// on tensors already in the graph, with ~200 scalar readbacks total.
type EngagementData struct {
	// BlockCosSim[il] is the cosine similarity of the residual stream before
	// and after the attention/SSM block (including residual add) at layer il.
	// Range [-1, 1]: 1 = no directional change (transparent), 0 = orthogonal.
	// NaN if the block was culled/skipped.
	BlockCosSim []float32

	// FFNCosSim[il] is the cosine similarity for the FFN block at layer il.
	// NaN if the FFN was culled/skipped.
	FFNCosSim []float32

	// Internal: graph tensors to read back after compute.
	blockTensors []ggml.Tensor
	ffnTensors   []ggml.Tensor
}

// newEngagementData allocates engagement tracking for nLayers.
func newEngagementData(nLayers int) *EngagementData {
	return &EngagementData{
		blockTensors: make([]ggml.Tensor, nLayers),
		ffnTensors:   make([]ggml.Tensor, nLayers),
	}
}

// buildCosineSim builds ggml graph ops computing cosine similarity between two
// same-shaped tensors. Returns a scalar output tensor (already marked as output).
//
//	cos(a, b) = sum(a*b) / (sqrt(sum(a*a)) * sqrt(sum(b*b)))
func buildCosineSim(gctx *ggml.GraphContext, a, b ggml.Tensor) ggml.Tensor {
	dot := ggml.Sum(gctx, ggml.Mul(gctx, a, b))
	normA := ggml.Sqrt(gctx, ggml.Sum(gctx, ggml.Mul(gctx, a, a)))
	normB := ggml.Sqrt(gctx, ggml.Sum(gctx, ggml.Mul(gctx, b, b)))
	denom := ggml.Mul(gctx, normA, normB)
	cs := ggml.Div(gctx, dot, denom)
	ggml.SetOutput(cs)
	return cs
}

// readResults copies the GPU scalar results into the exported slices.
func (ed *EngagementData) readResults() {
	ed.BlockCosSim = make([]float32, len(ed.blockTensors))
	ed.FFNCosSim = make([]float32, len(ed.ffnTensors))
	for il, tn := range ed.blockTensors {
		if tn.IsNil() {
			ed.BlockCosSim[il] = float32(math.NaN())
			continue
		}
		var v float32
		ggml.TensorGet(tn, unsafe.Pointer(&v), 0, 4)
		ed.BlockCosSim[il] = v
	}
	for il, tn := range ed.ffnTensors {
		if tn.IsNil() {
			ed.FFNCosSim[il] = float32(math.NaN())
			continue
		}
		var v float32
		ggml.TensorGet(tn, unsafe.Pointer(&v), 0, 4)
		ed.FFNCosSim[il] = v
	}
}
