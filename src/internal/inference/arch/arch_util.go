package arch

import (
	"fmt"
	"math"
	"os"
	"unsafe"

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

// debugCapture registers a tensor for post-compute dumping when instrumentation is active.
func debugCapture(inputs *GraphInputs, label string, t ggml.Tensor) {
	if inputs.DebugTensors != nil {
		ggml.SetOutput(t)
		*inputs.DebugTensors = append(*inputs.DebugTensors, DebugTensor{Label: label, Tensor: t})
	}
}

// DumpDebugTensors reads and prints the first N float32 values of each captured tensor to stderr.
func DumpDebugTensors(tensors []DebugTensor) {
	const nVals = 16
	for _, dt := range tensors {
		vals := make([]float32, nVals)
		nbytes := dt.Tensor.Nbytes()
		readSize := nVals * 4
		if readSize > nbytes {
			readSize = nbytes
		}
		ggml.TensorGet(dt.Tensor, unsafe.Pointer(&vals[0]), 0, readSize)
		fmt.Fprintf(os.Stderr, "[DUMP] %-24s ne=[%d,%d,%d]", dt.Label, dt.Tensor.Ne(0), dt.Tensor.Ne(1), dt.Tensor.Ne(2))
		for i := range readSize / 4 {
			fmt.Fprintf(os.Stderr, " %12.6f", vals[i])
		}
		fmt.Fprintln(os.Stderr)
	}
}

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
