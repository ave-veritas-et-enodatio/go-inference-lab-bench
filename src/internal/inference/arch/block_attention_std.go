package arch

import (
	ggml "inference-lab-bench/internal/inference/ggml"
)

// AttentionBuilder implements standard multi-head attention with GQA and RoPE.
// No gating on output, no joint Q+gate projection — standard transformer attention.
type AttentionBuilder struct{}

func (b *AttentionBuilder) Contract() BuilderContract {
	return BuilderContract{
		RequiredWeights: []string{"attn_q", "attn_k", "attn_v", "attn_output"},
		OptionalWeights: []string{"attn_q_norm", "attn_k_norm"},
		RequiredParams:  []string{"head_dim", "n_heads", "n_kv_heads", "rope_n_rot", "rope_freq_base"},
		ConfigSchema: map[string][]string{
			"rope": {"standard", ""},
		},
	}
}

func (b *AttentionBuilder) BuildStateless(
	ctx *ggml.GraphContext, cur ggml.Tensor, weights map[string]ggml.Tensor,
	params *ResolvedParams, config map[string]any, inputs *GraphInputs,
	zeroFill *[]ggml.Tensor) ggml.Tensor {

	headDim := int64(params.Ints["head_dim"])
	nHeads := int64(params.Ints["n_heads"])
	nKVHeads := int64(params.Ints["n_kv_heads"])
	nTokens := inputs.NTokens

	q := projectReshape3D(ctx, weights["attn_q"], cur, headDim, nHeads, nTokens)
	k := projectReshape3D(ctx, weights["attn_k"], cur, headDim, nKVHeads, nTokens)
	v := projectReshape3D(ctx, weights["attn_v"], cur, headDim, nKVHeads, nTokens)

	// Optional QK-norm
	if !weights["attn_q_norm"].IsNil() {
		rmsEps := params.Floats["rms_eps"]
		q = rmsNormApply(ctx, q, weights["attn_q_norm"], rmsEps)
		k = rmsNormApply(ctx, k, weights["attn_k_norm"], rmsEps)
	}

	// Standard RoPE
	nRot := params.Ints["rope_n_rot"]
	freqBase := params.Floats["rope_freq_base"]
	q, k = applyRoPEPair(ctx, q, k, inputs.InpPos, nRot, freqBase)

	// Permute for attention
	q = ggml.Permute(ctx, q, 0, 2, 1, 3)
	k = ggml.Permute(ctx, k, 0, 2, 1, 3)
	v = ggml.Permute(ctx, v, 0, 2, 1, 3)

	// Scaled dot-product attention with GQA
	cur = scaledDotProductAttention(ctx, q, k, v, inputs.InpMask, headDim, nHeads, nTokens)
	cur = ggml.MulMat(ctx, weights["attn_output"], cur)
	return cur
}

func (b *AttentionBuilder) BuildCached(
	ctx *ggml.GraphContext, cur ggml.Tensor, weights map[string]ggml.Tensor,
	params *ResolvedParams, config map[string]any, inputs *GraphInputs,
	cache *LayerCache, writebacks *[]CacheWriteback) ggml.Tensor {

	headDim := int64(params.Ints["head_dim"])
	nHeads := int64(params.Ints["n_heads"])
	nKVHeads := int64(params.Ints["n_kv_heads"])
	nNew := inputs.NTokens
	nKV := inputs.NKV
	seqPos := inputs.SeqPos

	q := projectReshape3D(ctx, weights["attn_q"], cur, headDim, nHeads, nNew)
	kNew := projectReshape3D(ctx, weights["attn_k"], cur, headDim, nKVHeads, nNew)
	vNew := projectReshape3D(ctx, weights["attn_v"], cur, headDim, nKVHeads, nNew)

	// Optional QK-norm
	if !weights["attn_q_norm"].IsNil() {
		rmsEps := params.Floats["rms_eps"]
		q = rmsNormApply(ctx, q, weights["attn_q_norm"], rmsEps)
		kNew = rmsNormApply(ctx, kNew, weights["attn_k_norm"], rmsEps)
	}

	// Standard RoPE
	nRot := params.Ints["rope_n_rot"]
	freqBase := params.Floats["rope_freq_base"]
	q, kNew = applyRoPEPair(ctx, q, kNew, inputs.InpPos, nRot, freqBase)

	// KV cache writeback
	writeCacheKV(ctx, kNew, vNew, cache, seqPos, nKVHeads, writebacks)

	// Prefill uses inline K/V, decode reads from cache
	kAttn, vAttn := selectCachedKV(ctx, cache, seqPos, kNew, vNew, headDim, nKV, nKVHeads)

	q = ggml.Permute(ctx, q, 0, 2, 1, 3)

	cur = scaledDotProductAttention(ctx, q, kAttn, vAttn, inputs.InpMask, headDim, nHeads, nNew)
	cur = ggml.MulMat(ctx, weights["attn_output"], cur)
	return cur
}
