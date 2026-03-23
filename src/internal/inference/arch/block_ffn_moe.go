package arch

import (
	"math"

	ggml "inference-lab-bench/internal/inference/ggml"
)

// MoEWithSharedBuilder implements a Mixture of Experts FFN with an optional
// sigmoid-gated shared expert (as used by Qwen3.5 MoE).
//
// Expected weights:
//   Routed:  gate_inp, gate_exps, up_exps, down_exps
//   Shared:  gate_inp_shexp, gate_shexp, up_shexp, down_shexp (all optional)
//
// Expected params:
//   n_expert, n_expert_used (ints)
//   expert_weights_scale (float, optional — 0.0 means no scaling)
type MoEWithSharedBuilder struct{}

func (b *MoEWithSharedBuilder) Contract() BuilderContract {
	return BuilderContract{
		RequiredWeights: []string{"gate_inp", "gate_exps", "up_exps", "down_exps"},
		OptionalWeights: []string{"gate_inp_shexp", "gate_shexp", "up_shexp", "down_shexp", "exp_probs_b"},
		RequiredParams:  []string{"n_expert", "n_expert_used"},
	}
}

func (b *MoEWithSharedBuilder) BuildFFN(ctx *ggml.GraphContext, input ggml.Tensor,
	weights map[string]ggml.Tensor, params *ResolvedParams) ggml.Tensor {

	nExpert := int64(params.Ints["n_expert"])
	nExpertUsed := int64(params.Ints["n_expert_used"])
	nEmbd := input.Ne(0)
	nTokens := input.Ne(1)

	// 1. Router logits: [n_expert, n_tokens]
	logits := ggml.MulMat(ctx, weights["gate_inp"], input)

	// 2. Gating probabilities (softmax): [n_expert, n_tokens]
	probs := ggml.SoftMaxExt(ctx, logits, ggml.NilTensor(), 1.0, 0.0)

	// 3. Expert selection — optionally biased for top-k (DeepSeek V2/V3 style)
	//    Bias affects selection ranking but NOT the final aggregation weights.
	selectionProbs := probs
	if !weights["exp_probs_b"].IsNil() {
		selectionProbs = ggml.Add(ctx, probs, weights["exp_probs_b"])
	}
	selected := ggml.ArgsortTopK(ctx, selectionProbs, int(nExpertUsed))

	// 4. Extract routing weights for selected experts
	//    reshape probs to [1, n_expert, n_tokens], gather → [1, n_expert_used, n_tokens]
	probs3d := ggml.Reshape3D(ctx, probs, 1, nExpert, nTokens)
	routeWeights := ggml.GetRows(ctx, probs3d, selected)

	// 4a. Normalize routing weights so the selected subset sums to 1 per token
	rw2d := ggml.Reshape2D(ctx, routeWeights, nExpertUsed, nTokens)
	rwSum := ggml.SumRows(ctx, rw2d) // [1, n_tokens]
	rwSum = ggml.Clamp(ctx, rwSum, 6.103515625e-5, float32(math.Inf(1)))
	rw2d = ggml.Div(ctx, rw2d, rwSum)
	routeWeights = ggml.Reshape3D(ctx, rw2d, 1, nExpertUsed, nTokens)

	// Optional weight scaling
	wScale := params.Floats["expert_weights_scale"]
	if wScale != 0.0 && wScale != 1.0 {
		routeWeights = ggml.Scale(ctx, routeWeights, wScale)
	}

	// 5. Reshape input to 3D for mul_mat_id: [n_embd, 1, n_tokens]
	cur := ggml.Reshape3D(ctx, input, nEmbd, 1, nTokens)

	// 6. Expert projections via indexed matmul
	gate := ggml.MulMatId(ctx, weights["gate_exps"], cur, selected) // [n_ff, n_expert_used, n_tokens]
	up := ggml.MulMatId(ctx, weights["up_exps"], cur, selected)     // [n_ff, n_expert_used, n_tokens]

	// 7. SwiGLU activation: silu(gate) * up
	activated := ggml.Mul(ctx, ggml.Silu(ctx, gate), up)

	// 8. Down projection: [n_embd, n_expert_used, n_tokens]
	experts := ggml.MulMatId(ctx, weights["down_exps"], activated, selected)

	// 9. Apply routing weights
	experts = ggml.Mul(ctx, experts, routeWeights)

	// 10. Aggregate expert outputs by summing across the expert dimension
	var moeOut ggml.Tensor
	for e := range nExpertUsed {
		slice := ggml.View2D(ctx, experts, nEmbd, nTokens,
			experts.Nb(2), int(e)*experts.Nb(1))
		if e == 0 {
			moeOut = slice
		} else {
			moeOut = ggml.Add(ctx, moeOut, slice)
		}
	}
	if nExpertUsed == 1 {
		moeOut = ggml.Cont(ctx, moeOut)
	}

	// 11. Shared expert (optional)
	if !weights["up_shexp"].IsNil() {
		// Standard SwiGLU for shared expert
		shGate := ggml.Silu(ctx, ggml.MulMat(ctx, weights["gate_shexp"], input))
		shUp := ggml.MulMat(ctx, weights["up_shexp"], input)
		shOut := ggml.MulMat(ctx, weights["down_shexp"], ggml.Mul(ctx, shGate, shUp))

		// Optional sigmoid gate per token (Qwen3.5 MoE has this, DeepSeek2 does not)
		if !weights["gate_inp_shexp"].IsNil() {
			shGateVal := ggml.Sigmoid(ctx, ggml.MulMat(ctx, weights["gate_inp_shexp"], input))
			shOut = ggml.Mul(ctx, shOut, shGateVal)
		}

		moeOut = ggml.Add(ctx, moeOut, shOut)
	}

	return moeOut
}

