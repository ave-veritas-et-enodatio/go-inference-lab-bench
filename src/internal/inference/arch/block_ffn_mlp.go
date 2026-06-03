package arch

import (
	ggml "inference-lab-bench/internal/ggml"
)

// mlpBuilder implements a plain (non-gated) MLP feed-forward network:
//
//	down · act(up · x)
//
// No gate — this is the difference from gluBuilder. The activation is selected
// per-layer via config[ConfigActivation] (default GELU, matching the Qwen3-VL
// vision tower which uses tanh-approx GELU). Optional per-projection biases on
// up/down are added after each matmul (no-op when absent, mirroring the
// attention builder's addProjBias discipline). Both matmuls route through
// mulMatClamped: every decoder path supplies a nil LinearClamps map, so the
// clamp resolves inactive and the matmul degrades to a bare ggml.MulMat.
//
// Dormant: no arch.toml selects "mlp"; it goes live with the Qwen3-VL vision
// tower in P5.
type mlpBuilder struct{}

func (b *mlpBuilder) Contract() BuilderContract {
	return BuilderContract{
		Kind:            KindFFN,
		RequiredWeights: []string{MoEUp, MoEDown},
		OptionalWeights: []string{MoEUpBias, MoEDownBias},
		ConfigSchema: map[string][]string{
			ConfigActivation: {ActivationSiLU, ActivationGELU, ""},
		},
	}
}

func (b *mlpBuilder) BuildFFN(ctx *ggml.GraphContext, input ggml.Tensor,
	weights map[string]ggml.Tensor, params *ResolvedParams, config map[string]any,
	inputs *GraphInputs) ggml.Tensor {

	upClamp := clampFor(inputs.LinearClamps, WeightFFNUp)
	downClamp := clampFor(inputs.LinearClamps, WeightFFNDown)
	activation := configStrOr(config, ConfigActivation, ActivationGELU)

	up := mulMatClamped(ctx, weights[MoEUp], input, upClamp)
	up = addFFNBias(ctx, up, weights[MoEUpBias])
	up = applyActivation(ctx, up, activation)

	down := mulMatClamped(ctx, weights[MoEDown], up, downClamp)
	down = addFFNBias(ctx, down, weights[MoEDownBias])
	return down
}

// addFFNBias adds an optional 1D FFN projection bias, broadcast over the token
// dimension. bias is [ff] (after up) or [n_embd] (after down); ggml broadcasts
// it across the [*, n_tokens] activation. Nil bias (every decoder/Gemma path)
// is a no-op, leaving the graph byte-identical.
func addFFNBias(ctx *ggml.GraphContext, t, bias ggml.Tensor) ggml.Tensor {
	if bias.IsNil() {
		return t
	}
	return ggml.Add(ctx, t, bias)
}
