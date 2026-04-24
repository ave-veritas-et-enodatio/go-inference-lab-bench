package arch

import (
	ggml "inference-lab-bench/internal/ggml"
)

// gluBuilder implements a gated-linear-unit feed-forward network:
// activation(gate * x) * (up * x) → down.
//
// The activation function is selected at registration time and is the only
// difference between the "swiglu" (SiLU) and "geglu" (GELU) variants. The
// activation is dispatched via applyActivation (shared with the MoE builder)
// using the ActivationSiLU / ActivationGELU string constants.
type gluBuilder struct {
	activation string
}

func (b *gluBuilder) Contract() BuilderContract {
	return BuilderContract{
		Kind:            KindFFN,
		RequiredWeights: []string{MoEGate, MoEUp, MoEDown},
	}
}

func (b *gluBuilder) BuildFFN(ctx *ggml.GraphContext, input ggml.Tensor,
	weights map[string]ggml.Tensor, params *ResolvedParams, config map[string]any) ggml.Tensor {

	gateFfn := applyActivation(ctx, ggml.MulMat(ctx, weights[MoEGate], input), b.activation)
	up := ggml.MulMat(ctx, weights[MoEUp], input)
	return ggml.MulMat(ctx, weights[MoEDown], ggml.Mul(ctx, gateFfn, up))
}
