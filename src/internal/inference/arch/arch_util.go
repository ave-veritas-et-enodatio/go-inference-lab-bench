package arch

import (
	"errors"
	"math"

	ggml "inference-lab-bench/internal/ggml"
)

// ErrComputeFailed is returned when the ggml GPU backend fails to execute a
// graph compute. The GPU context sets an internal has_error flag that makes
// every subsequent compute call fail immediately — the engine must be evicted
// and re-created to recover.
var ErrComputeFailed = errors.New("graph compute failed")

// Cache tensor key names. These must match the key names used in
// [blocks.*.cache] sections of the arch TOML definitions.
const (
	CacheK         = "k"
	CacheV         = "v"
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

// configIntOr reads an integer param, using a config override if present.
// Config values are param-name strings; the named param is looked up in params.Ints.
func configIntOr(config map[string]any, key string, params *ResolvedParams) int {
	if config != nil {
		if v, ok := config[key]; ok {
			if s, ok := v.(string); ok {
				if iv, ok := params.Ints[s]; ok {
					return iv
				}
			}
		}
	}
	return params.Ints[key]
}

// configFloatOr reads a float param, using a config override if present.
// Config values can be a TOML float literal or a string param name.
func configFloatOr(config map[string]any, key string, params *ResolvedParams) float32 {
	if config != nil {
		if v, ok := config[key]; ok {
			if f, ok := v.(float64); ok {
				return float32(f)
			}
			if i, ok := v.(int64); ok {
				return float32(i)
			}
			if s, ok := v.(string); ok {
				if fv, ok := params.Floats[s]; ok {
					return fv
				}
			}
		}
	}
	return params.Floats[key]
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
