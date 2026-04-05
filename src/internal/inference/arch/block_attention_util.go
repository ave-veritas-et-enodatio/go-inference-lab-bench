package arch

import (
	ggml "inference-lab-bench/internal/inference/ggml"
)

// scaledDotProductAttention computes scaled dot-product attention on pre-permuted Q, K, V.
// Q, K, V must already be in [dim, nTokens/nKV, nHeads/nKVHeads] layout (post-permute).
// Returns merged heads: [headDim*nHeads, nTokens].
func scaledDotProductAttention(ctx *ggml.GraphContext, q, k, v, mask ggml.Tensor,
	kqScale float32, nHeads, nTokens int64) ggml.Tensor {
	kq := ggml.MulMat(ctx, k, q)
	ggml.MulMatSetPrecF32(kq)
	kq = ggml.SoftMaxExt(ctx, kq, mask, kqScale, 0.0)
	vT := ggml.Cont(ctx, ggml.Transpose(ctx, v))
	kqv := ggml.MulMat(ctx, vT, kq)
	return ggml.Cont2D(ctx, ggml.Permute(ctx, kqv, 0, 2, 1, 3), kqv.Ne(0)*nHeads, nTokens)
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

// selectSharedKV returns K and V for a non-KV layer using the shared cache.
// Prefill: uses SharedKV directly (all tokens computed by the KV layer in-graph).
// Decode: concatenates cached history with the current-token SharedKV tensor.
func selectSharedKV(ctx *ggml.GraphContext, cache *LayerCache, seqPos int,
	shared *SharedKVState, group string, headDim, nKV, nKVHeads int64) (ggml.Tensor, ggml.Tensor) {
	kShared := shared.K[group]
	vShared := shared.V[group]

	if seqPos == 0 {
		// Prefill: SharedKV has all tokens — permute to [headDim, nTokens, nKVHeads]
		kAttn := ggml.Cont(ctx, ggml.Permute(ctx, kShared, 0, 2, 1, 3))
		vAttn := ggml.Cont(ctx, ggml.Permute(ctx, vShared, 0, 2, 1, 3))
		return kAttn, vAttn
	}

	// Decode: concat cache[0:seqPos] with current-token shared K/V
	kc := cache.Tensors[CacheK]
	vc := cache.Tensors[CacheV]

	// Cache view: [headDim, seqPos, nKVHeads]
	kHist := ggml.View3D(ctx, kc, headDim, int64(seqPos), nKVHeads, kc.Nb(1), kc.Nb(2), 0)
	vHist := ggml.View3D(ctx, vc, headDim, int64(seqPos), nKVHeads, vc.Nb(1), vc.Nb(2), 0)

	// Shared current token: permute [headDim, nKVHeads, 1] → [headDim, 1, nKVHeads]
	kCur := ggml.Cont(ctx, ggml.Permute(ctx, kShared, 0, 2, 1, 3))
	vCur := ggml.Cont(ctx, ggml.Permute(ctx, vShared, 0, 2, 1, 3))

	// Concat along dim 1 (sequence): [headDim, seqPos+1, nKVHeads]
	kAttn := ggml.Concat(ctx, kHist, kCur, 1)
	vAttn := ggml.Concat(ctx, vHist, vCur, 1)

	return kAttn, vAttn
}
