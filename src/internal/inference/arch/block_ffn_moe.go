package arch

import (
	"math"

	ggml "inference-lab-bench/internal/ggml"
)

// MoEBuilder implements a unified Mixture of Experts FFN that handles multiple
// MoE architectures through weight detection and config:
//
// Core MoE routing (shared by all variants):
//   softmax → top-k → normalize weights → MulMatId → weight → aggregate
//
// Variant features (auto-detected from weights):
//   - Router: direct matmul (Qwen) or rms_norm + learned scale (Gemma4, via gate_inp_s)
//   - Expert proj: separate gate_exps/up_exps (Qwen) or fused gate_up_exps (Gemma4)
//   - Per-expert output scaling: down_exps_s (Gemma4)
//   - Expert selection bias: exp_probs_b (DeepSeek)
//   - Shared expert: SwiGLU with sigmoid gate (*_shexp, Qwen) or plain FFN (gate/up/down, Gemma4)
//
// Config-driven features:
//   activation  — "silu" (default) or "gelu"
//   self_normed — "true" if builder manages its own pre/post norms (Gemma4 parallel paths)
type MoEBuilder struct{}

func (b *MoEBuilder) Contract() BuilderContract {
	return BuilderContract{
		Kind:            KindFFN,
		ExpertRouted:    true,
		RequiredWeights: []string{MoEGateInp, MoEDownExps},
		OptionalWeights: []string{
			// Router
			MoEGateInpS,
			// Separate expert projections (Qwen-style)
			MoEGateExps, MoEUpExps,
			// Fused expert projection (Gemma4-style)
			MoEGateUpExps,
			// Per-expert scaling
			MoEDownExpsS,
			// Expert bias
			MoEExpProbsB,
			// Shared expert — Qwen-style (SwiGLU + optional sigmoid gate)
			MoEGateInpShexp, MoEGateShexp, MoEUpShexp, MoEDownShexp,
			// Shared expert — Gemma4-style (plain FFN using same activation)
			MoEGate, MoEUp, MoEDown,
			// Self-normed weights (parallel shared+expert paths)
			MoENorm, MoEPreNorm2, MoEPostNorm1, MoEPostNorm2,
		},
		RequiredParams: []string{ParamNExpert, ParamNExpertUsed},
		ConfigSchema: map[string][]string{
			ConfigActivation: {ActivationSiLU, ActivationGELU, ""},
			ConfigSelfNormed: nil,
			ConfigNormW:      {"true", "false", ""},
			ConfigNormWParam: nil, // free-form string value; validated in arch.go, not here
		},
	}
}

func (b *MoEBuilder) BuildFFN(ctx *ggml.GraphContext, input ggml.Tensor,
	weights map[string]ggml.Tensor, params *ResolvedParams, config map[string]any) ggml.Tensor {

	nExpert := int64(params.Ints[ParamNExpert])
	nExpertUsed := int64(params.Ints[ParamNExpertUsed])
	nEmbd := input.Ne(0)
	nTokens := input.Ne(1)
	selfNormed := configBoolOr(config, ConfigSelfNormed, false)

	// --- Router logits ---
	var logits ggml.Tensor
	if !weights[MoEGateInpS].IsNil() {
		// Normalized router (Gemma4): rms_norm(input) / sqrt(n_embd) * learned_scale → matmul
		rmsEps := params.Floats[ParamRMSEps]
		tmp := ggml.RmsNorm(ctx, input, rmsEps)
		tmp = ggml.Scale(ctx, tmp, float32(1.0/math.Sqrt(float64(nEmbd))))
		tmp = ggml.Mul(ctx, tmp, weights[MoEGateInpS])
		logits = ggml.MulMat(ctx, weights[MoEGateInp], tmp)
	} else {
		// Direct router (Qwen): gate_inp @ input
		logits = ggml.MulMat(ctx, weights[MoEGateInp], input)
	}

	// --- Gating probabilities (softmax) ---
	probs := ggml.SoftMaxExt(ctx, logits, ggml.NilTensor(), 1.0, 0.0)

	// --- Expert selection (optionally biased for ranking only) ---
	selectionProbs := probs
	if !weights[MoEExpProbsB].IsNil() {
		selectionProbs = ggml.Add(ctx, probs, weights[MoEExpProbsB])
	}
	selected := ggml.ArgsortTopK(ctx, selectionProbs, int(nExpertUsed))

	// --- Extract routing weights ---
	probs3d := ggml.Reshape3D(ctx, probs, 1, nExpert, nTokens)
	routeWeights := ggml.GetRows(ctx, probs3d, selected)

	// Optional weight normalization: re-normalize selected weights to sum to 1.0.
	// norm_w = "true" (Qwen3.5-MoE requires this); default = no normalization (LLaDA-MoE, etc.).
	// norm_w_param overrides norm_w when set: looks up the named integer param (GGUF booleans
	// resolve to Ints as 0/1); absent param → 0 → false is the correct default.
	normW := configBoolOr(config, ConfigNormW, false)
	if normWParam := configStrOr(config, ConfigNormWParam, ""); normWParam != "" {
		normW = params.Ints[normWParam] != 0
	}
	if normW {
		rw2d := ggml.Reshape2D(ctx, routeWeights, nExpertUsed, nTokens)
		rwSum := ggml.SumRows(ctx, rw2d)
		rwSum = ggml.Clamp(ctx, rwSum, 6.103515625e-5, float32(math.Inf(1)))
		rw2d = ggml.Div(ctx, rw2d, rwSum)
		routeWeights = ggml.Reshape3D(ctx, rw2d, 1, nExpertUsed, nTokens)
	}

	// Optional weight scaling
	wScale := params.Floats[ParamExpertWeightScale]
	if wScale != 0.0 && wScale != 1.0 {
		routeWeights = ggml.Scale(ctx, routeWeights, wScale)
	}

	// --- Expert input ---
	// For self-normed builders, apply the expert pre-norm; otherwise input is already normed.
	expertInput := input
	if selfNormed && !weights[MoEPreNorm2].IsNil() {
		rmsEps := params.Floats[ParamRMSEps]
		expertInput = rmsNormApply(ctx, input, weights[MoEPreNorm2], rmsEps)
	}
	cur := ggml.Reshape3D(ctx, expertInput, nEmbd, 1, nTokens)

	// --- Expert projections ---
	activation := configStrOr(config, ConfigActivation, "")

	var activated ggml.Tensor
	if !weights[MoEGateUpExps].IsNil() {
		// Fused gate+up path (Gemma4): single MulMatId → View3D split
		gateUp := ggml.MulMatId(ctx, weights[MoEGateUpExps], cur, selected)
		nFF := gateUp.Ne(0) / 2
		gate := ggml.View3D(ctx, gateUp, nFF, gateUp.Ne(1), gateUp.Ne(2),
			gateUp.Nb(1), gateUp.Nb(2), 0)
		up := ggml.View3D(ctx, gateUp, nFF, gateUp.Ne(1), gateUp.Ne(2),
			gateUp.Nb(1), gateUp.Nb(2), int(nFF)*gateUp.Nb(0))
		activated = ggml.Mul(ctx, applyActivation(ctx, gate, activation), up)
	} else {
		// Separate gate/up path (Qwen): two MulMatId ops
		gate := ggml.MulMatId(ctx, weights[MoEGateExps], cur, selected)
		up := ggml.MulMatId(ctx, weights[MoEUpExps], cur, selected)
		activated = ggml.Mul(ctx, applyActivation(ctx, gate, activation), up)
	}

	// --- Down projection ---
	experts := ggml.MulMatId(ctx, weights[MoEDownExps], activated, selected)

	// --- Per-expert output scaling ---
	// down_exps_s is [n_expert] — one scale per expert. Gather scales for each
	// selected expert per token. selected is a non-contiguous view [n_expert_used, n_tokens],
	// so we flatten via Cont2D, gather, then reshape to [1, n_expert_used, n_tokens].
	if !weights[MoEDownExpsS].IsNil() {
		s := ggml.Reshape2D(ctx, weights[MoEDownExpsS], 1, nExpert)
		selFlat := ggml.Reshape2D(ctx, ggml.Cont(ctx, selected), nExpertUsed*nTokens, 1)
		sGather := ggml.GetRows(ctx, s, selFlat) // [1, n_expert_used * n_tokens]
		sGather = ggml.Reshape3D(ctx, sGather, 1, nExpertUsed, nTokens)
		experts = ggml.Mul(ctx, experts, sGather)
	}

	// --- Apply routing weights ---
	experts = ggml.Mul(ctx, experts, routeWeights)

	// --- Aggregate expert outputs ---
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

	// --- Post-norm for expert path (self-normed only) ---
	if selfNormed && !weights[MoEPostNorm2].IsNil() {
		rmsEps := params.Floats[ParamRMSEps]
		moeOut = rmsNormApply(ctx, moeOut, weights[MoEPostNorm2], rmsEps)
	}

	// --- Shared expert ---
	moeOut = b.addSharedExpert(ctx, moeOut, input, weights, params, config)

	return moeOut
}

// addSharedExpert adds the shared expert output to the MoE output.
// Detects Qwen-style (*_shexp weights, SwiGLU + optional sigmoid gate) or
// Gemma4-style (gate/up/down weights, same activation as experts).
func (b *MoEBuilder) addSharedExpert(ctx *ggml.GraphContext, moeOut, input ggml.Tensor,
	weights map[string]ggml.Tensor, params *ResolvedParams, config map[string]any) ggml.Tensor {

	selfNormed := configBoolOr(config, ConfigSelfNormed, false)
	activation := configStrOr(config, ConfigActivation, "")

	// Qwen-style shared expert: separate *_shexp weights with SwiGLU + optional sigmoid gate
	if !weights[MoEUpShexp].IsNil() {
		shGate := ggml.Silu(ctx, ggml.MulMat(ctx, weights[MoEGateShexp], input))
		shUp := ggml.MulMat(ctx, weights[MoEUpShexp], input)
		shOut := ggml.MulMat(ctx, weights[MoEDownShexp], ggml.Mul(ctx, shGate, shUp))
		if !weights[MoEGateInpShexp].IsNil() {
			shGateVal := ggml.Sigmoid(ctx, ggml.MulMat(ctx, weights[MoEGateInpShexp], input))
			shOut = ggml.Mul(ctx, shOut, shGateVal)
		}
		return ggml.Add(ctx, moeOut, shOut)
	}

	// Gemma4-style shared expert: plain FFN using gate/up/down with same activation
	if !weights[MoEUp].IsNil() {
		sharedInput := input
		if selfNormed && !weights[MoENorm].IsNil() {
			rmsEps := params.Floats[ParamRMSEps]
			sharedInput = rmsNormApply(ctx, input, weights[MoENorm], rmsEps)
		}
		gate := applyActivation(ctx, ggml.MulMat(ctx, weights[MoEGate], sharedInput), activation)
		up := ggml.MulMat(ctx, weights[MoEUp], sharedInput)
		shOut := ggml.MulMat(ctx, weights[MoEDown], ggml.Mul(ctx, gate, up))
		if selfNormed && !weights[MoEPostNorm1].IsNil() {
			rmsEps := params.Floats[ParamRMSEps]
			shOut = rmsNormApply(ctx, shOut, weights[MoEPostNorm1], rmsEps)
		}
		return ggml.Add(ctx, moeOut, shOut)
	}

	return moeOut
}

// applyActivation applies the configured activation function to a tensor.
func applyActivation(ctx *ggml.GraphContext, x ggml.Tensor, activation string) ggml.Tensor {
	if activation == ActivationGELU {
		return ggml.Gelu(ctx, x)
	}
	return ggml.Silu(ctx, x) // default
}
