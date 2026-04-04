package arch

import (
	ggml "inference-lab-bench/internal/inference/ggml"
)

// FullAttentionGatedBuilder implements gated multi-head attention with QK-norm and MRoPE.
// Direct translation of buildFullAttnLayer / buildFullAttnLayerCached in qwen35.go.
type FullAttentionGatedBuilder struct{}

func (b *FullAttentionGatedBuilder) Contract() BuilderContract {
	return BuilderContract{
		RequiredWeights: []string{"attn_q", "attn_k", "attn_v", "attn_output"},
		OptionalWeights: []string{"attn_q_norm", "attn_k_norm"},
		RequiredParams:  []string{"head_dim", "n_heads", "n_kv_heads", "rope_n_rot", "rope_freq_base"},
		ConfigSchema: map[string][]string{
			"q_has_gate":  nil, // bool, any value
			"qk_norm":     {"rms", "l2", ""},
			"rope":        {"multi", "standard", ""},
			"output_gate": {"sigmoid", ""},
		},
	}
}

// splitGatedProjection splits a joint Q+gate projection into separate Q and gate tensors.
func splitGatedProjection(ctx *ggml.GraphContext, qg ggml.Tensor, headDim, nHeads, nTokens int64) (ggml.Tensor, ggml.Tensor) {
	elemSz := qg.ElementSize()
	q := ggml.View3D(ctx, qg, headDim, nHeads, nTokens,
		elemSz*int(headDim)*2, elemSz*int(headDim)*2*int(nHeads), 0)
	gate := ggml.View3D(ctx, qg, headDim, nHeads, nTokens,
		elemSz*int(headDim)*2, elemSz*int(headDim)*2*int(nHeads), elemSz*int(headDim))
	gate = ggml.Cont2D(ctx, gate, headDim*nHeads, nTokens)
	return q, gate
}

func (b *FullAttentionGatedBuilder) BuildStateless(
	ctx *ggml.GraphContext, cur ggml.Tensor, weights map[string]ggml.Tensor,
	params *ResolvedParams, config map[string]any, inputs *GraphInputs,
	zeroFill *[]ggml.Tensor) ggml.Tensor {

	headDim := int64(params.Ints["head_dim"])
	nHeads := int64(params.Ints["n_heads"])
	nKVHeads := int64(params.Ints["n_kv_heads"])
	rmsEps := params.Floats["rms_eps"]
	nTokens := inputs.NTokens

	// Q+gate joint projection
	qg := ggml.MulMat(ctx, weights["attn_q"], cur)
	q, gate := splitGatedProjection(ctx, qg, headDim, nHeads, nTokens)

	// QK norm
	q = rmsNormApply(ctx, q, weights["attn_q_norm"], rmsEps)

	k := projectReshape3D(ctx, weights["attn_k"], cur, headDim, nKVHeads, nTokens)
	k = rmsNormApply(ctx, k, weights["attn_k_norm"], rmsEps)

	v := projectReshape3D(ctx, weights["attn_v"], cur, headDim, nKVHeads, nTokens)

	// MRoPE
	nRot := params.Ints["rope_n_rot"]
	sections := ropeSections(params)
	freqBase := params.Floats["rope_freq_base"]
	q, k = applyRoPEMultiPair(ctx, q, k, inputs.InpPos, nRot, sections, freqBase)

	// Permute for attention
	q = ggml.Permute(ctx, q, 0, 2, 1, 3)
	k = ggml.Permute(ctx, k, 0, 2, 1, 3)
	v = ggml.Permute(ctx, v, 0, 2, 1, 3)

	// Scaled dot-product attention with GQA
	cur = scaledDotProductAttention(ctx, q, k, v, inputs.InpMask, attentionScale(headDim), nHeads, nTokens)

	// Gate + output projection
	cur = ggml.Mul(ctx, cur, ggml.Sigmoid(ctx, gate))
	cur = ggml.MulMat(ctx, weights["attn_output"], cur)
	return cur
}

func (b *FullAttentionGatedBuilder) BuildCached(
	ctx *ggml.GraphContext, cur ggml.Tensor, weights map[string]ggml.Tensor,
	params *ResolvedParams, config map[string]any, inputs *GraphInputs,
	cache *LayerCache, writebacks *[]CacheWriteback) ggml.Tensor {

	headDim := int64(params.Ints["head_dim"])
	nHeads := int64(params.Ints["n_heads"])
	nKVHeads := int64(params.Ints["n_kv_heads"])
	rmsEps := params.Floats["rms_eps"]
	nNew := inputs.NTokens
	nKV := inputs.NKV
	seqPos := inputs.SeqPos

	// Q+gate joint projection
	qg := ggml.MulMat(ctx, weights["attn_q"], cur)
	q, gate := splitGatedProjection(ctx, qg, headDim, nHeads, nNew)

	q = rmsNormApply(ctx, q, weights["attn_q_norm"], rmsEps)

	kNew := projectReshape3D(ctx, weights["attn_k"], cur, headDim, nKVHeads, nNew)
	kNew = rmsNormApply(ctx, kNew, weights["attn_k_norm"], rmsEps)
	vNew := projectReshape3D(ctx, weights["attn_v"], cur, headDim, nKVHeads, nNew)

	// RoPE
	nRot := params.Ints["rope_n_rot"]
	sections := ropeSections(params)
	freqBase := params.Floats["rope_freq_base"]
	q, kNew = applyRoPEMultiPair(ctx, q, kNew, inputs.InpPos, nRot, sections, freqBase)

	// KV cache writeback
	writeCacheKV(ctx, kNew, vNew, cache, seqPos, nKVHeads, writebacks)

	// For attention: prefill uses inline K/V, decode reads from cache
	kAttn, vAttn := selectCachedKV(ctx, cache, seqPos, kNew, vNew, headDim, nKV, nKVHeads)

	q = ggml.Permute(ctx, q, 0, 2, 1, 3)

	cur = scaledDotProductAttention(ctx, q, kAttn, vAttn, inputs.InpMask, attentionScale(headDim), nHeads, nNew)
	cur = ggml.Mul(ctx, cur, ggml.Sigmoid(ctx, gate))
	cur = ggml.MulMat(ctx, weights["attn_output"], cur)
	return cur
}
