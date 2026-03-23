package arch

import (
	ggml "inference-lab-bench/internal/inference/ggml"
)

// newZeroFilledInput creates a tensor, marks it as input, and appends it to the zero-fill list.
func newZeroFilledInput(ctx *ggml.GraphContext, typ int, zeroFill *[]ggml.Tensor, dims ...int64) ggml.Tensor {
	var t ggml.Tensor
	switch len(dims) {
	case 3:
		t = ggml.NewTensor3D(ctx, typ, dims[0], dims[1], dims[2])
	case 4:
		t = ggml.NewTensor4D(ctx, typ, dims[0], dims[1], dims[2], dims[3])
	}
	ggml.SetInput(t)
	*zeroFill = append(*zeroFill, t)
	return t
}

// GatedDeltaNetBuilder implements the gated delta-net SSM block.
// Direct translation of buildSSMLayer / buildSSMLayerCached in qwen35.go.
type GatedDeltaNetBuilder struct{}

func (b *GatedDeltaNetBuilder) Contract() BuilderContract {
	return BuilderContract{
		RequiredWeights: []string{"attn_qkv", "attn_gate", "ssm_a", "ssm_alpha", "ssm_beta",
			"ssm_conv1d", "ssm_dt_bias", "ssm_out"},
		OptionalWeights: []string{"ssm_norm"},
		RequiredParams: []string{"conv_channels", "ssm_dt_rank", "ssm_d_state", "ssm_n_group",
			"head_v_dim", "ssm_d_conv", "ssm_d_inner", "n_embd"},
		ConfigSchema: map[string][]string{
			"conv_activation": {"silu", ""},
			"qk_norm":         {"l2", "rms", ""},
			"gate_norm":       {"rms", ""},
			"gate_activation": {"silu", ""},
		},
	}
}

func (b *GatedDeltaNetBuilder) BuildStateless(
	ctx *ggml.GraphContext, cur ggml.Tensor, weights map[string]ggml.Tensor,
	params *ResolvedParams, config map[string]any, inputs *GraphInputs,
	zeroFill *[]ggml.Tensor) ggml.Tensor {

	nTokens := inputs.NTokens
	rmsEps := params.Floats["rms_eps"]
	nSeqs := int64(1)
	cc := int64(params.Ints["conv_channels"])
	dtRank := int64(params.Ints["ssm_dt_rank"])
	dState := int64(params.Ints["ssm_d_state"])
	nGroup := int64(params.Ints["ssm_n_group"])
	hvDim := int64(params.Ints["head_v_dim"])
	dConv := int64(params.Ints["ssm_d_conv"])
	dInner := int64(params.Ints["ssm_d_inner"])
	nEmbd := int64(params.Ints["n_embd"])

	qkvMixed := ggml.Reshape3D(ctx, ggml.MulMat(ctx, weights["attn_qkv"], cur), cc, nTokens, nSeqs)
	z := ggml.MulMat(ctx, weights["attn_gate"], cur)

	beta := ggml.Sigmoid(ctx, ggml.Reshape4D(ctx, ggml.MulMat(ctx, weights["ssm_beta"], cur), 1, dtRank, nTokens, nSeqs))

	alpha := ggml.Reshape3D(ctx, ggml.MulMat(ctx, weights["ssm_alpha"], cur), dtRank, nTokens, nSeqs)
	alpha = ggml.Softplus(ctx, ggml.Add(ctx, alpha, weights["ssm_dt_bias"]))
	g := ggml.Reshape4D(ctx, ggml.Mul(ctx, alpha, weights["ssm_a"]), 1, dtRank, nTokens, nSeqs)

	// Conv1d with zero initial state
	convStates := newZeroFilledInput(ctx, ggml.TypeF32, zeroFill, dConv-1, cc, nSeqs)

	qkvT := ggml.Transpose(ctx, qkvMixed)
	convInput := ggml.Concat(ctx, convStates, qkvT, 0)
	convOut := ggml.Silu(ctx, ggml.SSMConv(ctx, convInput, weights["ssm_conv1d"]))

	// Split Q, K, V
	qkvDim := dState*nGroup*2 + hvDim*dtRank
	nb1QKV := ggml.RowSize(ggml.TypeF32, qkvDim)

	qSSM := ggml.View4D(ctx, convOut, dState, nGroup, nTokens, nSeqs,
		ggml.RowSize(ggml.TypeF32, dState), nb1QKV, nb1QKV*int(nTokens), 0)
	kSSM := ggml.View4D(ctx, convOut, dState, nGroup, nTokens, nSeqs,
		ggml.RowSize(ggml.TypeF32, dState), nb1QKV, nb1QKV*int(nTokens),
		int(dState*nGroup)*convOut.ElementSize())
	vSSM := ggml.View4D(ctx, convOut, hvDim, dtRank, nTokens, nSeqs,
		ggml.RowSize(ggml.TypeF32, hvDim), nb1QKV, nb1QKV*int(nTokens),
		ggml.RowSize(ggml.TypeF32, 2*dState*nGroup))

	qSSM = ggml.L2Norm(ctx, qSSM, rmsEps)
	kSSM = ggml.L2Norm(ctx, kSSM, rmsEps)

	if nGroup != dtRank {
		qSSM = ggml.Repeat4D(ctx, qSSM, dState, dtRank, nTokens, nSeqs)
		kSSM = ggml.Repeat4D(ctx, kSSM, dState, dtRank, nTokens, nSeqs)
	}

	// Delta-net fused op with zero initial state
	ssmState := newZeroFilledInput(ctx, ggml.TypeF32, zeroFill, hvDim, hvDim, dtRank, nSeqs)

	deltaOut := ggml.GatedDeltaNet(ctx, qSSM, kSSM, vSSM, g, beta, ssmState)

	ssmOutput := ggml.View4D(ctx, deltaOut, hvDim, dtRank, nTokens, nSeqs,
		ggml.RowSize(ggml.TypeF32, hvDim),
		ggml.RowSize(ggml.TypeF32, hvDim*dtRank),
		ggml.RowSize(ggml.TypeF32, hvDim*dtRank*nTokens), 0)

	// Gated normalization
	z4d := ggml.Reshape4D(ctx, z, hvDim, dtRank, nTokens, nSeqs)
	cur = ggml.Mul(ctx, rmsNormApply(ctx, ssmOutput, weights["ssm_norm"], rmsEps), ggml.Silu(ctx, z4d))

	cur = ggml.Reshape2D(ctx, cur, dInner, nTokens*nSeqs)
	cur = ggml.MulMat(ctx, weights["ssm_out"], cur)
	cur = ggml.Reshape2D(ctx, cur, nEmbd, nTokens*nSeqs)
	return cur
}

func (b *GatedDeltaNetBuilder) BuildCached(
	ctx *ggml.GraphContext, cur ggml.Tensor, weights map[string]ggml.Tensor,
	params *ResolvedParams, config map[string]any, inputs *GraphInputs,
	cache *LayerCache, writebacks *[]CacheWriteback) ggml.Tensor {

	nNew := inputs.NTokens
	rmsEps := params.Floats["rms_eps"]
	nSeqs := int64(1)
	cc := int64(params.Ints["conv_channels"])
	dtRank := int64(params.Ints["ssm_dt_rank"])
	dState := int64(params.Ints["ssm_d_state"])
	nGroup := int64(params.Ints["ssm_n_group"])
	hvDim := int64(params.Ints["head_v_dim"])
	dConv := int64(params.Ints["ssm_d_conv"])
	dInner := int64(params.Ints["ssm_d_inner"])
	nEmbd := int64(params.Ints["n_embd"])

	qkvMixed := ggml.Reshape3D(ctx, ggml.MulMat(ctx, weights["attn_qkv"], cur), cc, nNew, nSeqs)
	z := ggml.MulMat(ctx, weights["attn_gate"], cur)

	beta := ggml.Sigmoid(ctx, ggml.Reshape4D(ctx, ggml.MulMat(ctx, weights["ssm_beta"], cur), 1, dtRank, nNew, nSeqs))

	alpha := ggml.Reshape3D(ctx, ggml.MulMat(ctx, weights["ssm_alpha"], cur), dtRank, nNew, nSeqs)
	alpha = ggml.Softplus(ctx, ggml.Add(ctx, alpha, weights["ssm_dt_bias"]))
	g := ggml.Reshape4D(ctx, ggml.Mul(ctx, alpha, weights["ssm_a"]), 1, dtRank, nNew, nSeqs)

	// Conv1d with cached conv state
	convSt3d := ggml.Reshape3D(ctx, cache.Tensors[CacheConvState], dConv-1, cc, nSeqs)
	qkvT := ggml.Transpose(ctx, qkvMixed)
	convInput := ggml.Concat(ctx, convSt3d, qkvT, 0)

	// Conv state writeback
	newConvSt := ggml.View3D(ctx, convInput, dConv-1, cc, nSeqs,
		convInput.Nb(1), convInput.Nb(2),
		int(convInput.Ne(0)-(dConv-1))*convInput.ElementSize())
	newConvStCont := ggml.Cont(ctx, newConvSt)
	ggml.SetOutput(newConvStCont)
	convBytes := int((dConv - 1) * cc * nSeqs * 4)
	*writebacks = append(*writebacks, CacheWriteback{
		Src: newConvStCont, Dst: cache.Tensors[CacheConvState],
		NHeads: 1, HeadSrc: convBytes, HeadDst: convBytes, HeadOffset: 0, HeadBytes: convBytes,
	})

	convOut := ggml.Silu(ctx, ggml.SSMConv(ctx, convInput, weights["ssm_conv1d"]))

	// Split Q, K, V
	qkvDim := dState*nGroup*2 + hvDim*dtRank
	nb1QKV := ggml.RowSize(ggml.TypeF32, qkvDim)

	qSSM := ggml.View4D(ctx, convOut, dState, nGroup, nNew, nSeqs,
		ggml.RowSize(ggml.TypeF32, dState), nb1QKV, nb1QKV*int(nNew), 0)
	kSSM := ggml.View4D(ctx, convOut, dState, nGroup, nNew, nSeqs,
		ggml.RowSize(ggml.TypeF32, dState), nb1QKV, nb1QKV*int(nNew),
		int(dState*nGroup)*convOut.ElementSize())
	vSSM := ggml.View4D(ctx, convOut, hvDim, dtRank, nNew, nSeqs,
		ggml.RowSize(ggml.TypeF32, hvDim), nb1QKV, nb1QKV*int(nNew),
		ggml.RowSize(ggml.TypeF32, 2*dState*nGroup))

	qSSM = ggml.L2Norm(ctx, qSSM, rmsEps)
	kSSM = ggml.L2Norm(ctx, kSSM, rmsEps)

	if nGroup != dtRank {
		qSSM = ggml.Repeat4D(ctx, qSSM, dState, dtRank, nNew, nSeqs)
		kSSM = ggml.Repeat4D(ctx, kSSM, dState, dtRank, nNew, nSeqs)
	}

	// Delta-net with cached SSM state
	ssmSt4d := ggml.Reshape4D(ctx, cache.Tensors[CacheSSMState], hvDim, hvDim, dtRank, nSeqs)
	deltaOut := ggml.GatedDeltaNet(ctx, qSSM, kSSM, vSSM, g, beta, ssmSt4d)

	ssmOutput := ggml.View4D(ctx, deltaOut, hvDim, dtRank, nNew, nSeqs,
		ggml.RowSize(ggml.TypeF32, hvDim), ggml.RowSize(ggml.TypeF32, hvDim*dtRank),
		ggml.RowSize(ggml.TypeF32, hvDim*dtRank*nNew), 0)

	// SSM state writeback
	newSSMSt := ggml.View4D(ctx, deltaOut, hvDim, hvDim, dtRank, nSeqs,
		ggml.RowSize(ggml.TypeF32, hvDim), ggml.RowSize(ggml.TypeF32, hvDim*hvDim),
		ggml.RowSize(ggml.TypeF32, hvDim*hvDim*dtRank),
		ggml.RowSize(ggml.TypeF32, hvDim*dtRank*nNew*nSeqs))
	newSSMCont := ggml.Cont(ctx, newSSMSt)
	ggml.SetOutput(newSSMCont)
	ssmBytes := int(hvDim * hvDim * dtRank * nSeqs * 4)
	*writebacks = append(*writebacks, CacheWriteback{
		Src: newSSMCont, Dst: cache.Tensors[CacheSSMState],
		NHeads: 1, HeadSrc: ssmBytes, HeadDst: ssmBytes, HeadOffset: 0, HeadBytes: ssmBytes,
	})

	// Gated normalization
	z4d := ggml.Reshape4D(ctx, z, hvDim, dtRank, nNew, nSeqs)
	cur = ggml.Mul(ctx, rmsNormApply(ctx, ssmOutput, weights["ssm_norm"], rmsEps), ggml.Silu(ctx, z4d))

	cur = ggml.Reshape2D(ctx, cur, dInner, nNew*nSeqs)
	cur = ggml.MulMat(ctx, weights["ssm_out"], cur)
	cur = ggml.Reshape2D(ctx, cur, nEmbd, nNew*nSeqs)
	return cur
}
