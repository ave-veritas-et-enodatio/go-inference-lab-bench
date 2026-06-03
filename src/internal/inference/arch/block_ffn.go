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
	weights map[string]ggml.Tensor, params *ResolvedParams, config map[string]any,
	inputs *GraphInputs) ggml.Tensor {

	// Clamps resolve to inactive (bare MulMat) for every decoder path, where
	// GraphInputs.LinearClamps is nil. The Gemma-4 vision tower supplies a
	// per-layer clamp map keyed by ffn_gate / ffn_up / ffn_down.
	gateClamp := clampFor(inputs.LinearClamps, WeightFFNGate)
	upClamp := clampFor(inputs.LinearClamps, WeightFFNUp)
	downClamp := clampFor(inputs.LinearClamps, WeightFFNDown)

	gateFfn := applyActivation(ctx, mulMatClamped(ctx, weights[MoEGate], input, gateClamp), b.activation)
	up := mulMatClamped(ctx, weights[MoEUp], input, upClamp)
	return mulMatClamped(ctx, weights[MoEDown], ggml.Mul(ctx, gateFfn, up), downClamp)
}
