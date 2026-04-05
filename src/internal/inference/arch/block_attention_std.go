package arch

import (
	ggml "inference-lab-bench/internal/inference/ggml"
)

// AttentionBuilder implements standard multi-head attention with GQA and RoPE.
// Supports config-based param overrides for per-block head dims/counts,
// optional K/V (shared KV for non-KV layers), V-norm, sliding window mask,
// and RoPE frequency factors.
type AttentionBuilder struct{}

func (b *AttentionBuilder) Contract() BuilderContract {
	return BuilderContract{
		RequiredWeights: []string{"attn_q", "attn_output"},
		OptionalWeights: []string{"attn_k", "attn_v", "attn_q_norm", "attn_k_norm", "rope_freqs"},
		RequiredParams:  []string{"head_dim", "n_heads", "n_kv_heads", "rope_n_rot", "rope_freq_base"},
		ConfigSchema: map[string][]string{
			"rope":            {"standard", "neox", ""},
			"v_norm":          {"rms", ""},
			"sliding_window":  nil,
			"shared_kv_group": nil,
			"head_dim":        nil,
			"n_heads":         nil,
			"n_kv_heads":      nil,
			"rope_n_rot":      nil,
			"rope_freq_base":  nil,
			"kq_scale":        nil,
		},
	}
}

// attnParams reads attention dimensions from config overrides or global params.
func attnParams(params *ResolvedParams, config map[string]any) (headDim, nHeads, nKVHeads int64, nRot, ropeMode int, freqBase, kqScale float32) {
	headDim = int64(configIntOr(config, "head_dim", params))
	nHeads = int64(configIntOr(config, "n_heads", params))
	nKVHeads = int64(configIntOr(config, "n_kv_heads", params))
	nRot = configIntOr(config, "rope_n_rot", params)
	freqBase = configFloatOr(config, "rope_freq_base", params)
	kqScale = configFloatOr(config, "kq_scale", params)
	if kqScale == 0 {
		kqScale = attentionScale(headDim)
	}
	if configStr(config, "rope") == "neox" {
		ropeMode = ggml.RopeNeoX
	}
	return
}

func (b *AttentionBuilder) BuildStateless(
	ctx *ggml.GraphContext, cur ggml.Tensor, weights map[string]ggml.Tensor,
	params *ResolvedParams, config map[string]any, inputs *GraphInputs,
	zeroFill *[]ggml.Tensor) ggml.Tensor {

	headDim, nHeads, nKVHeads, nRot, ropeMode, freqBase, kqScale := attnParams(params, config)
	nTokens := inputs.NTokens
	hasKV := !weights["attn_k"].IsNil()

	q := projectReshape3D(ctx, weights["attn_q"], cur, headDim, nHeads, nTokens)

	var k, v ggml.Tensor
	if hasKV {
		k = projectReshape3D(ctx, weights["attn_k"], cur, headDim, nKVHeads, nTokens)
		if !weights["attn_v"].IsNil() {
			v = projectReshape3D(ctx, weights["attn_v"], cur, headDim, nKVHeads, nTokens)
		} else {
			v = k // use K as V if attn_v absent
		}
	}

	// Optional QK-norm
	if !weights["attn_q_norm"].IsNil() {
		rmsEps := params.Floats["rms_eps"]
		q = rmsNormApply(ctx, q, weights["attn_q_norm"], rmsEps)
		if hasKV {
			k = rmsNormApply(ctx, k, weights["attn_k_norm"], rmsEps)
		}
	}

	// Optional V-norm (raw RMS norm, no learned weights)
	if hasKV && configStr(config, "v_norm") == "rms" {
		rmsEps := params.Floats["rms_eps"]
		v = ggml.RmsNorm(ctx, v, rmsEps)
	}

	// RoPE with optional frequency factors
	freqFactors := weights["rope_freqs"]
	if freqFactors.IsNil() {
		freqFactors = ggml.NilTensor()
	}
	q = ggml.RopeExt(ctx, q, inputs.InpPos, freqFactors, nRot, ropeMode, 0,
		freqBase, 1.0, 0.0, 1.0, 32.0, 1.0)
	if hasKV {
		k = ggml.RopeExt(ctx, k, inputs.InpPos, freqFactors, nRot, ropeMode, 0,
			freqBase, 1.0, 0.0, 1.0, 32.0, 1.0)
	}

	// Update SharedKV if this is a KV layer
	if hasKV && inputs.SharedKV != nil {
		if group := configStr(config, "shared_kv_group"); group != "" {
			inputs.SharedKV.K[group] = k
			inputs.SharedKV.V[group] = v
		}
	}

	// Non-KV layers get K/V from SharedKV
	if !hasKV && inputs.SharedKV != nil {
		group := configStr(config, "shared_kv_group")
		k = inputs.SharedKV.K[group]
		v = inputs.SharedKV.V[group]
	}

	// Select mask (SWA or standard)
	mask := inputs.InpMask
	if configStr(config, "sliding_window") == "true" && !inputs.InpMaskSWA.IsNil() {
		mask = inputs.InpMaskSWA
	}

	// Permute for attention
	q = ggml.Permute(ctx, q, 0, 2, 1, 3)
	k = ggml.Permute(ctx, k, 0, 2, 1, 3)
	v = ggml.Permute(ctx, v, 0, 2, 1, 3)

	cur = scaledDotProductAttention(ctx, q, k, v, mask, kqScale, nHeads, nTokens)
	cur = ggml.MulMat(ctx, weights["attn_output"], cur)
	return cur
}

func (b *AttentionBuilder) BuildCached(
	ctx *ggml.GraphContext, cur ggml.Tensor, weights map[string]ggml.Tensor,
	params *ResolvedParams, config map[string]any, inputs *GraphInputs,
	cache *LayerCache, writebacks *[]CacheWriteback) ggml.Tensor {

	headDim, nHeads, nKVHeads, nRot, ropeMode, freqBase, kqScale := attnParams(params, config)
	nNew := inputs.NTokens
	nKV := inputs.NKV
	seqPos := inputs.SeqPos
	hasKV := !weights["attn_k"].IsNil()

	q := projectReshape3D(ctx, weights["attn_q"], cur, headDim, nHeads, nNew)

	var kNew, vNew ggml.Tensor
	if hasKV {
		kNew = projectReshape3D(ctx, weights["attn_k"], cur, headDim, nKVHeads, nNew)
		if !weights["attn_v"].IsNil() {
			vNew = projectReshape3D(ctx, weights["attn_v"], cur, headDim, nKVHeads, nNew)
		} else {
			vNew = kNew
		}
	}

	// Optional QK-norm
	if !weights["attn_q_norm"].IsNil() {
		rmsEps := params.Floats["rms_eps"]
		q = rmsNormApply(ctx, q, weights["attn_q_norm"], rmsEps)
		if hasKV {
			kNew = rmsNormApply(ctx, kNew, weights["attn_k_norm"], rmsEps)
		}
	}

	// Optional V-norm (raw RMS norm, no learned weights)
	if hasKV && configStr(config, "v_norm") == "rms" {
		rmsEps := params.Floats["rms_eps"]
		vNew = ggml.RmsNorm(ctx, vNew, rmsEps)
	}

	// RoPE with optional frequency factors
	freqFactors := weights["rope_freqs"]
	if freqFactors.IsNil() {
		freqFactors = ggml.NilTensor()
	}
	q = ggml.RopeExt(ctx, q, inputs.InpPos, freqFactors, nRot, ropeMode, 0,
		freqBase, 1.0, 0.0, 1.0, 32.0, 1.0)
	if hasKV {
		kNew = ggml.RopeExt(ctx, kNew, inputs.InpPos, freqFactors, nRot, ropeMode, 0,
			freqBase, 1.0, 0.0, 1.0, 32.0, 1.0)
	}

	// Update SharedKV with in-graph K/V for downstream non-KV layers
	if hasKV && inputs.SharedKV != nil {
		if group := configStr(config, "shared_kv_group"); group != "" {
			inputs.SharedKV.K[group] = kNew
			inputs.SharedKV.V[group] = vNew
		}
	}

	// Select mask (SWA or standard)
	mask := inputs.InpMask
	if configStr(config, "sliding_window") == "true" && !inputs.InpMaskSWA.IsNil() {
		mask = inputs.InpMaskSWA
	}

	if hasKV {
		// KV layer: write to cache and use cache for attention
		writeCacheKV(ctx, kNew, vNew, cache, seqPos, nKVHeads, writebacks)
		kAttn, vAttn := selectCachedKV(ctx, cache, seqPos, kNew, vNew, headDim, nKV, nKVHeads)
		q = ggml.Permute(ctx, q, 0, 2, 1, 3)
		cur = scaledDotProductAttention(ctx, q, kAttn, vAttn, mask, kqScale, nHeads, nNew)
	} else {
		// Non-KV layer: get K/V from shared state + cache
		group := configStr(config, "shared_kv_group")
		kAttn, vAttn := selectSharedKV(ctx, cache, seqPos, inputs.SharedKV, group, headDim, nKV, nKVHeads)
		q = ggml.Permute(ctx, q, 0, 2, 1, 3)
		cur = scaledDotProductAttention(ctx, q, kAttn, vAttn, mask, kqScale, nHeads, nNew)
	}

	cur = ggml.MulMat(ctx, weights["attn_output"], cur)
	return cur
}

// configStr returns a string config value or "".
func configStr(config map[string]any, key string) string {
	if config == nil {
		return ""
	}
	if v, ok := config[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
		// TOML booleans
		if b, ok := v.(bool); ok && b {
			return "true"
		}
	}
	return ""
}
