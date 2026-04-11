package arch

import (
	ggml "inference-lab-bench/internal/ggml"
)

// MLAAttentionBuilder implements Multi-head Latent Attention (DeepSeek V2 / GLM-4).
// Low-rank Q and KV compression with Q-nope absorption and post-attention V decompression.
type MLAAttentionBuilder struct{}

// buildMLAQueryPath computes the MLA Q path: low-rank compress → norm → expand → split →
// RoPE on positional component → Q-nope absorption → concat absorbed + positional.
// Returns qFinal: [kvLoraRank+ropeDim, nHeads, seqLen].
func buildMLAQueryPath(ctx *ggml.GraphContext, cur ggml.Tensor, weights map[string]ggml.Tensor,
	pos ggml.Tensor, nHeads, headKDim, ropeDim, seqLen int64,
	rmsEps, freqBase float32) ggml.Tensor {
	nopeDim := headKDim - ropeDim

	qCompressed := ggml.MulMat(ctx, weights["attn_q_a"], cur)
	qCompressed = rmsNormApply(ctx, qCompressed, weights["attn_q_a_norm"], rmsEps)
	qExpanded := ggml.MulMat(ctx, weights["attn_q_b"], qCompressed)
	q := ggml.Reshape3D(ctx, qExpanded, headKDim, nHeads, seqLen)

	qNopeNb1 := q.Nb(1)
	qNope := ggml.View3D(ctx, q, nopeDim, nHeads, seqLen,
		qNopeNb1, q.Nb(2), 0)
	qPe := ggml.View3D(ctx, q, ropeDim, nHeads, seqLen,
		qNopeNb1, q.Nb(2), int(nopeDim)*q.ElementSize())

	qPe = defaultRopeExt(ctx, qPe, pos, int(ropeDim), freqBase)

	qNopePerm := ggml.Permute(ctx, qNope, 0, 2, 1, 3)
	qAbsorbed := ggml.MulMat(ctx, weights["attn_k_b"], qNopePerm)
	qAbsorbed = ggml.Permute(ctx, qAbsorbed, 0, 2, 1, 3)
	return ggml.Concat(ctx, qAbsorbed, qPe, 0)
}

func (b *MLAAttentionBuilder) Contract() BuilderContract {
	return BuilderContract{
		RequiredWeights: []string{
			"attn_q_a", "attn_q_a_norm", "attn_q_b",
			"attn_kv_a_mqa", "attn_kv_a_norm",
			"attn_k_b", "attn_v_b",
			"attn_output",
		},
		RequiredParams: []string{
			"n_heads", "rms_eps", "rope_n_rot", "rope_freq_base",
			"kv_lora_rank", "head_k_dim_mla",
		},
	}
}

func (b *MLAAttentionBuilder) BuildStateless(
	ctx *ggml.GraphContext, cur ggml.Tensor, weights map[string]ggml.Tensor,
	params *ResolvedParams, config map[string]any, inputs *GraphInputs,
	zeroFill *[]ggml.Tensor) ggml.Tensor {

	nHeads := int64(params.Ints["n_heads"])
	rmsEps := params.Floats["rms_eps"]
	nTokens := inputs.NTokens
	kvLoraRank := int64(params.Ints["kv_lora_rank"])
	headKDim := int64(params.Ints["head_k_dim_mla"])
	ropeDim := int64(params.Ints["rope_n_rot"])
	freqBase := params.Floats["rope_freq_base"]

	qFinal := buildMLAQueryPath(ctx, cur, weights, inputs.InpPos,
		nHeads, headKDim, ropeDim, nTokens, rmsEps, freqBase)

	// KV path: compress → split → norm + RoPE
	kvFull := ggml.MulMat(ctx, weights["attn_kv_a_mqa"], cur) // [kvLoraRank+ropeDim, nTokens]

	// Split into compressed KV and positional K
	kvCompressed := ggml.View2D(ctx, kvFull, kvLoraRank, nTokens,
		kvFull.Nb(1), 0)
	kPe := ggml.View3D(ctx, kvFull, ropeDim, int64(1), nTokens,
		kvFull.Nb(1), kvFull.Nb(1),
		int(kvLoraRank)*kvFull.ElementSize())

	// Norm the compressed KV
	kvCompressed = rmsNormApply(ctx, kvCompressed, weights["attn_kv_a_norm"], rmsEps)

	// RoPE on positional K only
	kPe = defaultRopeExt(ctx, kPe, inputs.InpPos, int(ropeDim), freqBase)

	// K: concat compressed + positional (MQA: 1 KV head)
	kvCompressed3d := ggml.Reshape3D(ctx, kvCompressed, kvLoraRank, int64(1), nTokens)
	kFinal := ggml.Concat(ctx, kvCompressed3d, kPe, 0) // [kvLoraRank+ropeDim, 1, nTokens]

	// V: just the compressed representation
	vFinal := ggml.Reshape3D(ctx, kvCompressed, kvLoraRank, int64(1), nTokens)

	// Attention (MQA: 1 KV head, n_heads Q heads — GQA broadcasting)
	qPerm := ggml.Permute(ctx, qFinal, 0, 2, 1, 3) // [kvLoraRank+ropeDim, nTokens, nHeads]
	kPerm := ggml.Permute(ctx, kFinal, 0, 2, 1, 3) // [kvLoraRank+ropeDim, nTokens, 1]
	vPerm := ggml.Permute(ctx, vFinal, 0, 2, 1, 3) // [kvLoraRank, nTokens, 1]

	kqScale := attentionScale(kvLoraRank + ropeDim)
	kq := ggml.MulMat(ctx, kPerm, qPerm) // [nTokens, nTokens, nHeads] with GQA broadcast
	kq = ggml.SoftMaxExt(ctx, kq, inputs.InpMask, kqScale, 0.0)

	vT := ggml.Cont(ctx, ggml.Transpose(ctx, vPerm))
	kqv := ggml.MulMat(ctx, vT, kq) // [kvLoraRank, nTokens, nHeads]

	// V decompression: attn_v_b [kvLoraRank, headVDim, nHeads] @ kqv [kvLoraRank, nTokens, nHeads]
	// Batched matmul on dim 2 (nHeads), contracts on dim 0 (kvLoraRank) → [headVDim, nTokens, nHeads]
	decompressed := ggml.MulMat(ctx, weights["attn_v_b"], kqv)

	// Merge heads: permute to [headVDim, nHeads, nTokens] then flatten
	headVDim := decompressed.Ne(0)
	merged := ggml.Permute(ctx, decompressed, 0, 2, 1, 3) // [headVDim, nHeads, nTokens]
	cur = ggml.Cont2D(ctx, merged, headVDim*nHeads, nTokens)
	cur = ggml.MulMat(ctx, weights["attn_output"], cur)
	return cur
}

func (b *MLAAttentionBuilder) BuildCached(
	ctx *ggml.GraphContext, gf *ggml.Graph, cur ggml.Tensor, weights map[string]ggml.Tensor,
	params *ResolvedParams, config map[string]any, inputs *GraphInputs,
	cache *LayerCache) ggml.Tensor {

	nHeads := int64(params.Ints["n_heads"])
	rmsEps := params.Floats["rms_eps"]
	nNew := inputs.NTokens
	nKV := inputs.NKV
	seqPos := inputs.SeqPos
	kvLoraRank := int64(params.Ints["kv_lora_rank"])
	headKDim := int64(params.Ints["head_k_dim_mla"])
	ropeDim := int64(params.Ints["rope_n_rot"])
	freqBase := params.Floats["rope_freq_base"]
	kDim := kvLoraRank + ropeDim // total K cache dim per entry

	qFinal := buildMLAQueryPath(ctx, cur, weights, inputs.InpPos,
		nHeads, headKDim, ropeDim, nNew, rmsEps, freqBase)

	// KV path for new tokens
	kvFull := ggml.MulMat(ctx, weights["attn_kv_a_mqa"], cur)
	kvCompressed := ggml.View2D(ctx, kvFull, kvLoraRank, nNew, kvFull.Nb(1), 0)
	kPeNew := ggml.View3D(ctx, kvFull, ropeDim, int64(1), nNew,
		kvFull.Nb(1), kvFull.Nb(1),
		int(kvLoraRank)*kvFull.ElementSize())

	kvCompressed = rmsNormApply(ctx, kvCompressed, weights["attn_kv_a_norm"], rmsEps)
	kPeNew = defaultRopeExt(ctx, kPeNew, inputs.InpPos, int(ropeDim), freqBase)

	kvCompressed3d := ggml.Reshape3D(ctx, kvCompressed, kvLoraRank, int64(1), nNew)
	kNew := ggml.Concat(ctx, kvCompressed3d, kPeNew, 0) // [kDim, 1, nNew]

	// Cache writeback: K only (MLA: V derived from K's compressed portion).
	// Emit in-graph cpy into the cache buffer at seqPos.
	kForCache := ggml.Cont(ctx, ggml.Permute(ctx, kNew, 0, 2, 1, 3)) // [kDim, nNew, 1]
	kc := cache.Tensors[CacheK]
	const float32Size = 4
	kView := ggml.View3D(ctx, kc, kDim, nNew, int64(1), kc.Nb(1), kc.Nb(2), seqPos*int(kDim)*float32Size)
	gf.BuildForwardExpand(ggml.Cpy(ctx, kForCache, kView))

	// For attention: build K and V from cache or inline
	var kAttn, vAttn ggml.Tensor
	if seqPos == 0 {
		// Prefill: use inline
		kAttn = ggml.Cont(ctx, ggml.Permute(ctx, kNew, 0, 2, 1, 3))
		// V = compressed portion of K (first kvLoraRank elements)
		vNew := ggml.Reshape3D(ctx, kvCompressed, kvLoraRank, int64(1), nNew)
		vAttn = ggml.Cont(ctx, ggml.Permute(ctx, vNew, 0, 2, 1, 3))
	} else {
		// Decode: read K from cache, derive V from K's compressed portion
		kAttn = ggml.View3D(ctx, kc, kDim, nKV, int64(1), kc.Nb(1), kc.Nb(2), 0)
		// V = first kvLoraRank elements of each K entry
		vAttn = ggml.View3D(ctx, kc, kvLoraRank, nKV, int64(1), kc.Nb(1), kc.Nb(2), 0)
	}

	// Attention
	qPerm := ggml.Permute(ctx, qFinal, 0, 2, 1, 3)
	kqScale := attentionScale(kDim)
	kq := ggml.MulMat(ctx, kAttn, qPerm)
	kq = ggml.SoftMaxExt(ctx, kq, inputs.InpMask, kqScale, 0.0)

	vT := ggml.Cont(ctx, ggml.Transpose(ctx, vAttn))
	kqv := ggml.MulMat(ctx, vT, kq)

	// V decompression: attn_v_b [kvLoraRank, headVDim, nHeads] @ kqv [kvLoraRank, nNew, nHeads]
	// Batched matmul on dim 2 (nHeads), contracts on dim 0 (kvLoraRank) → [headVDim, nNew, nHeads]
	decompressed := ggml.MulMat(ctx, weights["attn_v_b"], kqv)

	// Merge heads: permute to [headVDim, nHeads, nNew] then flatten
	headVDim := decompressed.Ne(0)
	merged := ggml.Permute(ctx, decompressed, 0, 2, 1, 3) // [headVDim, nHeads, nNew]
	cur = ggml.Cont2D(ctx, merged, headVDim*nHeads, nNew)
	cur = ggml.MulMat(ctx, weights["attn_output"], cur)
	return cur
}
