package arch

import (
	ggml "inference-lab-bench/internal/inference/ggml"
)

// GeGLUBuilder implements the GeGLU feed-forward network.
// Same structure as SwiGLU but uses GELU activation: gelu(gate * x) * up * x → down.
type GeGLUBuilder struct{}

func (b *GeGLUBuilder) Contract() BuilderContract {
	return BuilderContract{
		RequiredWeights: []string{"gate", "up", "down"},
	}
}

func (b *GeGLUBuilder) BuildFFN(ctx *ggml.GraphContext, input ggml.Tensor,
	weights map[string]ggml.Tensor, params *ResolvedParams) ggml.Tensor {

	gateFfn := ggml.Gelu(ctx, ggml.MulMat(ctx, weights["gate"], input))
	up := ggml.MulMat(ctx, weights["up"], input)
	return ggml.MulMat(ctx, weights["down"], ggml.Mul(ctx, gateFfn, up))
}
