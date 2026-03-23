package arch

import (
	ggml "inference-lab-bench/internal/inference/ggml"
)

// scaledDotProductAttention computes scaled dot-product attention on pre-permuted Q, K, V.
// Q, K, V must already be in [dim, nTokens/nKV, nHeads/nKVHeads] layout (post-permute).
// Returns merged heads: [headDim*nHeads, nTokens].
func scaledDotProductAttention(ctx *ggml.GraphContext, q, k, v, mask ggml.Tensor,
	headDim, nHeads, nTokens int64) ggml.Tensor {
	kq := ggml.MulMat(ctx, k, q)
	kq = ggml.SoftMaxExt(ctx, kq, mask, attentionScale(headDim), 0.0)
	vT := ggml.Cont(ctx, ggml.Transpose(ctx, v))
	kqv := ggml.MulMat(ctx, vT, kq)
	return ggml.Cont2D(ctx, ggml.Permute(ctx, kqv, 0, 2, 1, 3), headDim*nHeads, nTokens)
}

// writeCacheKV prepares K and V for cache writeback and appends CacheWriteback entries.
func writeCacheKV(ctx *ggml.GraphContext, kNew, vNew ggml.Tensor,
	cache *LayerCache, seqPos int, nKVHeads int64,
	writebacks *[]CacheWriteback) {

	headDim := kNew.Ne(0)
	nNew := kNew.Ne(2)

	kForCache := ggml.Cont(ctx, ggml.Permute(ctx, kNew, 0, 2, 1, 3))
	ggml.SetOutput(kForCache)
	vForCache := ggml.Cont(ctx, ggml.Permute(ctx, vNew, 0, 2, 1, 3))
	ggml.SetOutput(vForCache)

	// F32 is the one and only supported cache element type.
	const float32Size = 4
	perHeadSrc := int(headDim) * int(nNew) * float32Size
	perHeadDst := int(headDim) * cache.MaxSeqLen * float32Size
	posOffset := seqPos * int(headDim) * float32Size
	*writebacks = append(*writebacks, CacheWriteback{
		Src: kForCache, Dst: cache.Tensors[CacheK],
		NHeads: int(nKVHeads), HeadSrc: perHeadSrc, HeadDst: perHeadDst, HeadOffset: posOffset, HeadBytes: perHeadSrc,
	})
	*writebacks = append(*writebacks, CacheWriteback{
		Src: vForCache, Dst: cache.Tensors[CacheV],
		NHeads: int(nKVHeads), HeadSrc: perHeadSrc, HeadDst: perHeadDst, HeadOffset: posOffset, HeadBytes: perHeadSrc,
	})
}

// selectCachedKV returns K and V tensors for attention — inline for prefill, from cache for decode.
func selectCachedKV(ctx *ggml.GraphContext, cache *LayerCache, seqPos int,
	kNew, vNew ggml.Tensor, headDim, nKV, nKVHeads int64) (ggml.Tensor, ggml.Tensor) {
	if seqPos == 0 {
		kAttn := ggml.Cont(ctx, ggml.Permute(ctx, kNew, 0, 2, 1, 3))
		vAttn := ggml.Cont(ctx, ggml.Permute(ctx, vNew, 0, 2, 1, 3))
		return kAttn, vAttn
	}
	kc := cache.Tensors[CacheK]
	vc := cache.Tensors[CacheV]
	kAttn := ggml.View3D(ctx, kc, headDim, nKV, nKVHeads, kc.Nb(1), kc.Nb(2), 0)
	vAttn := ggml.View3D(ctx, vc, headDim, nKV, nKVHeads, vc.Nb(1), vc.Nb(2), 0)
	return kAttn, vAttn
}

// defaultRopeExt applies RoPE with standard YaRN scaling defaults.
func defaultRopeExt(ctx *ggml.GraphContext, a, pos ggml.Tensor, nRot int, freqBase float32) ggml.Tensor {
	return ggml.RopeExt(ctx, a, pos, ggml.NilTensor(), nRot, 0, 0,
		freqBase, 1.0, 0.0, 1.0, 32.0, 1.0)
}

// defaultRopeMulti applies multi-section RoPE with standard YaRN scaling defaults.
func defaultRopeMulti(ctx *ggml.GraphContext, a, pos ggml.Tensor,
	nRot int, sections [4]int, freqBase float32) ggml.Tensor {
	return ggml.RopeMulti(ctx, a, pos, ggml.NilTensor(), nRot, sections, ggml.RopeNeoX, 0,
		freqBase, 1.0, 0.0, 1.0, 32.0, 1.0)
}

// applyRoPEPair applies standard RoPE to both Q and K tensors.
func applyRoPEPair(ctx *ggml.GraphContext, q, k, pos ggml.Tensor,
	nRot int, freqBase float32) (ggml.Tensor, ggml.Tensor) {
	return defaultRopeExt(ctx, q, pos, nRot, freqBase),
		defaultRopeExt(ctx, k, pos, nRot, freqBase)
}

// applyRoPEMultiPair applies multi-section RoPE to both Q and K tensors.
func applyRoPEMultiPair(ctx *ggml.GraphContext, q, k, pos ggml.Tensor,
	nRot int, sections [4]int, freqBase float32) (ggml.Tensor, ggml.Tensor) {
	return defaultRopeMulti(ctx, q, pos, nRot, sections, freqBase),
		defaultRopeMulti(ctx, k, pos, nRot, sections, freqBase)
}
