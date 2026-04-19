package arch

import (
	ggml "inference-lab-bench/internal/ggml"
)

// AttentionBuilder implements standard multi-head attention with GQA and RoPE.
// Supports config-based param overrides for per-block head dims/counts,
// optional K/V (shared KV for non-KV layers), V-norm, sliding window mask,
// and RoPE frequency factors.
type AttentionBuilder struct{}

func (b *AttentionBuilder) Contract() BuilderContract {
	return BuilderContract{
		RequiredWeights: []string{WeightAttnQ, WeightAttnOutput},
		OptionalWeights: []string{WeightAttnK, WeightAttnV, WeightAttnQNorm, WeightAttnKNorm, WeightRoPEFreqs},
		RequiredParams:  []string{ParamHeadDim, ParamNHeads, ParamNKVHeads, ParamRoPENRot, ParamRoPEFreqBase},
		ConfigSchema: map[string][]string{
			ConfigRope:          {RopeStandard, RopeNeox, ""},
			ConfigVNorm:         {NormRMS, ""},
			ParamSlidingWindow:  nil,
			ConfigSharedKVGroup: nil,
			ParamHeadDim:        nil,
			ParamNHeads:         nil,
			ParamNKVHeads:       nil,
			ParamRoPENRot:       nil,
			ParamRoPEFreqBase:   nil,
			ConfigKQScale:       nil,
		},
	}
}

// attnParams reads attention dimensions from config overrides or global params.
func attnParams(params *ResolvedParams, config map[string]any) (headDim, nHeads, nKVHeads int64, nRot, ropeMode int, freqBase, kqScale float32) {
	headDim = int64(configIntOr(config, ParamHeadDim, params))
	nHeads = int64(configIntOr(config, ParamNHeads, params))
	nKVHeads = int64(configIntOr(config, ParamNKVHeads, params))
	nRot = configIntOr(config, ParamRoPENRot, params)
	freqBase = configFloatOr(config, ParamRoPEFreqBase, params)
	kqScale = configFloatOr(config, ConfigKQScale, params)
	if kqScale == 0 {
		kqScale = attentionScale(headDim)
	}
	if configStrOr(config, ConfigRope, "") == RopeNeox {
		ropeMode = ggml.RopeNeoX
	}
	return
}

// qkvResult holds the prepared Q, K, V tensors from the KV pipeline.
// HasKV indicates whether this layer projected its own K/V (vs reading from SharedKV).
type qkvResult struct {
	Q, K, V  ggml.Tensor
	HasKV    bool
	NKVHeads int64 // actual kv_heads (may be inferred from K weight shape)
}

// prepareQKV runs the full KV preparation pipeline: Q/K/V projection, optional QK-norm,
// optional V-norm, RoPE (with optional frequency factors), and SharedKV read/write.
func (b *AttentionBuilder) prepareQKV(
	ctx *ggml.GraphContext, cur ggml.Tensor, weights map[string]ggml.Tensor,
	params *ResolvedParams, config map[string]any, inputs *GraphInputs,
	headDim, nHeads, nKVHeads int64, nRot, ropeMode int, freqBase float32,
) qkvResult {
	nTokens := inputs.NTokens
	hasKV := !weights[WeightAttnK].IsNil()

	q := projectReshape3D(ctx, weights[WeightAttnQ], cur, headDim, nHeads, nTokens)

	var k, v ggml.Tensor
	if hasKV {
		// Infer actual n_kv_heads from K weight shape when it differs from the param.
		// Handles models where SWA and full attention have different kv_head counts
		// but only one n_kv_heads param exists in the GGUF.
		if actual := weights[WeightAttnK].Ne(1) / headDim; actual > 0 {
			nKVHeads = actual
		}
		k = projectReshape3D(ctx, weights[WeightAttnK], cur, headDim, nKVHeads, nTokens)
		if !weights[WeightAttnV].IsNil() {
			v = projectReshape3D(ctx, weights[WeightAttnV], cur, headDim, nKVHeads, nTokens)
		} else {
			v = k
		}
	}

	// Optional QK-norm
	if !weights[WeightAttnQNorm].IsNil() {
		rmsEps := params.Floats[ParamRMSEps]
		q = rmsNormApply(ctx, q, weights[WeightAttnQNorm], rmsEps)
		if hasKV {
			k = rmsNormApply(ctx, k, weights[WeightAttnKNorm], rmsEps)
		}
	}

	// Optional V-norm (raw RMS norm, no learned weights)
	if hasKV && configStrOr(config, ConfigVNorm, "") == NormRMS {
		v = ggml.RmsNorm(ctx, v, params.Floats[ParamRMSEps])
	}

	// RoPE with optional frequency factors
	freqFactors := weights[WeightRoPEFreqs]
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
		if group := configStrOr(config, ConfigSharedKVGroup, ""); group != "" {
			inputs.SharedKV.K[group] = k
			inputs.SharedKV.V[group] = v
		}
	}

	// Non-KV layers get K/V from SharedKV
	if !hasKV && inputs.SharedKV != nil {
		group := configStrOr(config, ConfigSharedKVGroup, "")
		k = inputs.SharedKV.K[group]
		v = inputs.SharedKV.V[group]
	}

	return qkvResult{Q: q, K: k, V: v, HasKV: hasKV, NKVHeads: nKVHeads}
}

// selectMask returns the SWA mask if sliding_window is configured, else the standard mask.
func selectMask(config map[string]any, inputs *GraphInputs) ggml.Tensor {
	if configBoolOr(config, ParamSlidingWindow, false) && !inputs.InpMaskSWA.IsNil() {
		return inputs.InpMaskSWA
	}
	return inputs.InpMask
}

func (b *AttentionBuilder) BuildStateless(
	ctx *ggml.GraphContext, cur ggml.Tensor, weights map[string]ggml.Tensor,
	params *ResolvedParams, config map[string]any, inputs *GraphInputs,
	zeroFill *[]ggml.Tensor) ggml.Tensor {

	headDim, nHeads, nKVHeads, nRot, ropeMode, freqBase, kqScale := attnParams(params, config)
	kv := b.prepareQKV(ctx, cur, weights, params, config, inputs,
		headDim, nHeads, nKVHeads, nRot, ropeMode, freqBase)
	mask := selectMask(config, inputs)

	q := ggml.Permute(ctx, kv.Q, 0, 2, 1, 3)
	k := ggml.Permute(ctx, kv.K, 0, 2, 1, 3)
	v := ggml.Permute(ctx, kv.V, 0, 2, 1, 3)

	cur = scaledDotProductAttention(ctx, q, k, v, mask, kqScale, nHeads, inputs.NTokens, inputs.Captures, inputs.FlashAttn)
	return ggml.MulMat(ctx, weights[WeightAttnOutput], cur)
}

func (b *AttentionBuilder) BuildCached(
	ctx *ggml.GraphContext, gf *ggml.Graph, cur ggml.Tensor, weights map[string]ggml.Tensor,
	params *ResolvedParams, config map[string]any, inputs *GraphInputs,
	cache *LayerCache) ggml.Tensor {

	headDim, nHeads, nKVHeads, nRot, ropeMode, freqBase, kqScale := attnParams(params, config)
	kv := b.prepareQKV(ctx, cur, weights, params, config, inputs,
		headDim, nHeads, nKVHeads, nRot, ropeMode, freqBase)
	mask := selectMask(config, inputs)

	q := ggml.Permute(ctx, kv.Q, 0, 2, 1, 3)
	nKVHeadsActual := kv.NKVHeads
	if kv.HasKV {
		writeCacheKV(ctx, gf, kv.K, kv.V, cache, inputs.SeqPos, nKVHeadsActual)
		kAttn, vAttn := selectCachedKV(ctx, cache, inputs.SeqPos, kv.K, kv.V, headDim, inputs.NKV, nKVHeadsActual)
		cur = scaledDotProductAttention(ctx, q, kAttn, vAttn, mask, kqScale, nHeads, inputs.NTokens, inputs.Captures, inputs.FlashAttn)
	} else {
		group := configStrOr(config, ConfigSharedKVGroup, "")
		kAttn, vAttn := selectSharedKV(ctx, cache, inputs.SeqPos, inputs.SharedKV, group, headDim, inputs.NKV, nKVHeadsActual)
		cur = scaledDotProductAttention(ctx, q, kAttn, vAttn, mask, kqScale, nHeads, inputs.NTokens, inputs.Captures, inputs.FlashAttn)
	}

	return ggml.MulMat(ctx, weights[WeightAttnOutput], cur)
}
