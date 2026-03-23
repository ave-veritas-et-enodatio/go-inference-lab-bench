package arch

import (
	"math"

	ggml "inference-lab-bench/internal/inference/ggml"
)

// Cache tensor key names. These must match the key names used in
// [blocks.*.cache] sections of the arch TOML definitions.
const (
	CacheK        = "k"
	CacheV        = "v"
	CacheConvState = "conv_state"
	CacheSSMState  = "ssm_state"
)

// rmsNormApply applies RMS normalization and optional weight scaling.
// If weight is nil, returns the plain normalized tensor.
func rmsNormApply(ctx *ggml.GraphContext, tensor, weight ggml.Tensor, eps float32) ggml.Tensor {
	normed := ggml.RmsNorm(ctx, tensor, eps)
	if !weight.IsNil() {
		normed = ggml.Mul(ctx, normed, weight)
	}
	return normed
}

// projectReshape3D projects input through weight and reshapes to 3D.
func projectReshape3D(ctx *ggml.GraphContext, weight, input ggml.Tensor, d0, d1, d2 int64) ggml.Tensor {
	return ggml.Reshape3D(ctx, ggml.MulMat(ctx, weight, input), d0, d1, d2)
}

// attentionScale computes 1/sqrt(dim) for attention score scaling.
func attentionScale(dim int64) float32 {
	return float32(1.0 / math.Sqrt(float64(dim)))
}
