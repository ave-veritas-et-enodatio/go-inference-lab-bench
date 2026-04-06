package arch

import (
	ggml "inference-lab-bench/internal/inference/ggml"
)

// SwiGLUBuilder implements the SwiGLU feed-forward network.
// Matches the FFN in qwen35.go: silu(gate * x) * up * x → down.
type SwiGLUBuilder struct{}

func (b *SwiGLUBuilder) Contract() BuilderContract {
	return BuilderContract{
		RequiredWeights: []string{"gate", "up", "down"},
	}
}

func (b *SwiGLUBuilder) BuildFFN(ctx *ggml.GraphContext, input ggml.Tensor,
	weights map[string]ggml.Tensor, params *ResolvedParams, config map[string]any) ggml.Tensor {

	gateFfn := ggml.Silu(ctx, ggml.MulMat(ctx, weights["gate"], input))
	up := ggml.MulMat(ctx, weights["up"], input)
	return ggml.MulMat(ctx, weights["down"], ggml.Mul(ctx, gateFfn, up))
}
