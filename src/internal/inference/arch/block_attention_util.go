package arch

import (
	ggml "inference-lab-bench/internal/ggml"
	"inference-lab-bench/internal/log"
)

// flashAttnHeadDims is the set of head dimensions supported by the GPU FA2 kernel.
var flashAttnHeadDims = map[int64]bool{
	32: true, 40: true, 48: true, 64: true, 72: true, 80: true,
	96: true, 112: true, 128: true, 192: true, 256: true,
	320: true, 512: true, 576: true,
}

// FlashAttnSupported reports whether headDim is in the GPU FA2 supported set.
func FlashAttnSupported(headDim int) bool {
	return flashAttnHeadDims[int64(headDim)]
}

// scaledDotProductAttention computes scaled dot-product attention on pre-permuted Q, K, V.
// Q, K, V must already be in [dim, nTokens/nKV, nHeads/nKVHeads] layout (post-permute).
// Returns merged heads: [headDim*nHeads, nTokens].
//
// When flashAttn is true and the head dimension is in the GPU FA2 supported set,
// uses ggml_flash_attn_ext (fused online softmax). Otherwise falls back to the
// standard explicit KQ→softmax→V matmul path.
//
// Capture (caps) is skipped when FA2 is active — the fused op has no intermediate
// kq tensor to extract.
func scaledDotProductAttention(ctx *ggml.GraphContext, q, k, v, mask ggml.Tensor,
	kqScale float32, nHeads, nTokens int64, caps *ForwardCaptures, flashAttn bool) ggml.Tensor {
	if flashAttn && flashAttnHeadDims[q.Ne(0)] {
		// FA2: V is already in [headDim, nKV, nKVHeads] — no transpose needed.
		// Cast mask to F16 as required by ggml_flash_attn_ext.
		// Output shape: [headDim, nHeads, nTokens] — merged by Cont2D.
		maskF16 := ggml.Cast(ctx, mask, ggml.TypeF16)
		cur := ggml.FlashAttnExt(ctx, q, k, v, maskF16, kqScale, 0.0, 0.0)
		ggml.FlashAttnExtSetPrec(cur, ggml.PrecF32)
		return ggml.Cont2D(ctx, cur, cur.Ne(0)*nHeads, nTokens)
	}
	// Standard path.
	kq := ggml.MulMat(ctx, k, q)
	ggml.MulMatSetPrecF32(kq)
	kq = ggml.SoftMaxExt(ctx, kq, mask, kqScale, 0.0)
	if caps != nil && caps.Flags&CaptureAttnWeights != 0 {
		il := caps.currentLayer
		if il >= 0 && il < len(caps.attnTensors) {
			kqCap := ggml.Cont(ctx, kq)
			ggml.SetOutput(kqCap)
			caps.attnTensors[il] = kqCap
		}
	}
	vT := ggml.Cont(ctx, ggml.Transpose(ctx, v))
	kqv := ggml.MulMat(ctx, vT, kq)
	return ggml.Cont2D(ctx, ggml.Permute(ctx, kqv, 0, 2, 1, 3), kqv.Ne(0)*nHeads, nTokens)
}

// writeCacheKV emits in-graph ggml_cpy ops that write K and V directly into the
// persistent cache buffer on the GPU, with no CPU involvement.
// kNew and vNew are in [headDim, nKVHeads, nNew] layout (pre-permute).
func writeCacheKV(ctx *ggml.GraphContext, gf *ggml.Graph, kNew, vNew ggml.Tensor,
	cache *LayerCache, seqPos int, nKVHeads int64) {

	headDim := kNew.Ne(0)
	kc := cache.Tensors[CacheK]
	if kc.IsNil() {
		log.Error("writeCacheKV: nil K cache tensor — KV write skipped")
		return
	}
	nNew := kNew.Ne(2)

	// Permute new K/V to [headDim, nNew, nKVHeads] — contiguous for cpy source.
	kForCache := ggml.Cont(ctx, ggml.Permute(ctx, kNew, 0, 2, 1, 3))
	vForCache := ggml.Cont(ctx, ggml.Permute(ctx, vNew, 0, 2, 1, 3))

	// Build a strided view into the cache at seqPos.
	// Cache layout: [headDim, maxSeqLen, nKVHeads] (F32).
	// nb1 = headDim * 4 (stride between sequence positions within one head).
	// nb2 = kc.Nb(2) (stride between heads = headDim * maxSeqLen * 4).
	// offset = seqPos * headDim * 4 (skip past already-written positions).
	const float32Size = 4
	offset := seqPos * int(headDim) * float32Size

	kView := ggml.View3D(ctx, kc, headDim, nNew, nKVHeads, kc.Nb(1), kc.Nb(2), offset)
	gf.BuildForwardExpand(ggml.Cpy(ctx, kForCache, kView))

	vc := cache.Tensors[CacheV]
	vView := ggml.View3D(ctx, vc, headDim, nNew, nKVHeads, vc.Nb(1), vc.Nb(2), offset)
	gf.BuildForwardExpand(ggml.Cpy(ctx, vForCache, vView))
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
