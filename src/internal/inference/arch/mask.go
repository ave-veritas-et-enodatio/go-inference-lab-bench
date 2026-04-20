package arch

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/BurntSushi/toml"

	ggml "inference-lab-bench/internal/ggml"
	"inference-lab-bench/internal/log"
)

// CullingMask defines which weight tensors are culled (zeroed out).
type CullingMask struct {
	// ZeroTensors lists GGUF tensor names to zero out entirely.
	ZeroTensors []string `toml:"zero_tensors"`

	// Precomputed lookup set (built from ZeroTensors on load)
	zeroSet map[string]bool
}

// LoadCullingMask reads a mask from a ModuleMap file.
// Supports two formats:
//   - ModuleMap TOML (has a "modules" key): loads and compiles.
//   - Legacy CullingMask JSON (has a "zero_tensors" key): loads directly.
//
// Returns (nil, nil) if the file does not exist.
func LoadCullingMask(path string) (*CullingMask, error) {
	if path == "" {
		return nil, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	// Try TOML ModuleMap format first.
	var mm ModuleMap
	if _, err := toml.Decode(string(data), &mm); err == nil && len(mm.Modules) > 0 {
		return mm.Compile(), nil
	}

	// Legacy JSON CullingMask format.
	var cm CullingMask
	if err := json.Unmarshal(data, &cm); err != nil {
		return nil, fmt.Errorf("parsing module map: %w", err)
	}
	cm.zeroSet = make(map[string]bool, len(cm.ZeroTensors))
	for _, name := range cm.ZeroTensors {
		cm.zeroSet[name] = true
	}
	return &cm, nil
}

// IsZeroed returns true if the given GGUF tensor name should be zeroed out entirely.
func (cm *CullingMask) IsZeroed(ggufName string) bool {
	return cm.zeroSet[ggufName]
}

// NumZeroed returns the count of fully-zeroed tensors.
func (cm *CullingMask) NumZeroed() int {
	return len(cm.ZeroTensors)
}

// MaskedLayer returns per-layer weight tensors with culling applied.
// Culled tensors (in ZeroTensors) are omitted from the returned maps — the zero-value ggml.Tensor
// satisfies IsNil(), so the graph builder skips their ops unchanged.
// No mask: returns the raw LayerTensors from the store (zero overhead).
func (m *GenericModel) MaskedLayer(layerIdx int, mask *CullingMask) *LayerTensors {
	lt := m.Store.Layer(layerIdx)
	if lt == nil {
		log.Error("MaskedLayer: layer %d out of bounds (no layer data available)", layerIdx)
		return nil
	}
	if mask == nil || len(mask.zeroSet) == 0 {
		return lt
	}

	lw := m.Weights.Layers[layerIdx]
	masked := &LayerTensors{
		Common: maskSubmap(lt.Common, lw.Common, mask),
		Block:  maskSubmap(lt.Block, lw.Block, mask),
		FFN:    maskFFN(lt.FFN, lw.FFN, lw.FFNAlt, mask),
	}
	return masked
}

// maskSubmap filters a weight submap through the culling mask. resolved maps
// logical name → GGUF tensor name for the name lookup.
func maskSubmap(src map[string]ggml.Tensor, resolved map[string]string, mask *CullingMask) map[string]ggml.Tensor {
	out := make(map[string]ggml.Tensor, len(src))
	for name, tensor := range src {
		if ggufName, ok := resolved[name]; ok && mask.zeroSet[ggufName] {
			continue
		}
		out[name] = tensor
	}
	return out
}

// maskFFN filters an FFN weight submap through the culling mask. FFN and FFNAlt
// resolved maps are both checked since both contribute to the same FFN submap.
func maskFFN(src map[string]ggml.Tensor, ffn, ffnAlt map[string]string, mask *CullingMask) map[string]ggml.Tensor {
	out := make(map[string]ggml.Tensor, len(src))
	for name, tensor := range src {
		if ggufName, ok := ffn[name]; ok && mask.zeroSet[ggufName] {
			continue
		}
		if ggufName, ok := ffnAlt[name]; ok && mask.zeroSet[ggufName] {
			continue
		}
		out[name] = tensor
	}
	return out
}

// MaskedGlobal returns a global weight tensor from the store.
// Global tensors (token_embd, output_norm, output) are never culled — culling them
// would break inference entirely.
func (m *GenericModel) MaskedGlobal(_ *ggml.GraphContext, name string) ggml.Tensor {
	return m.Store.Global(name)
}


// makeTensorFromSpec creates a new graph-context tensor from explicit type and dimension parameters.
// Unused dimensions should be 1. Used for Go-parser tensor creation.
func makeTensorFromSpec(gctx *ggml.GraphContext, typ ggml.GGMLType, ne0, ne1, ne2, ne3 int64) ggml.Tensor {
	if ne3 > 1 {
		return ggml.NewTensor4D(gctx, typ, ne0, ne1, ne2, ne3)
	} else if ne2 > 1 {
		return ggml.NewTensor3D(gctx, typ, ne0, ne1, ne2)
	} else if ne1 > 1 {
		return ggml.NewTensor2D(gctx, typ, ne0, ne1)
	}
	return ggml.NewTensor1D(gctx, typ, ne0)
}

// makeSameTypeTensor creates a new graph-context tensor with the same quantized type and shape
// as src. Used for weight tensor allocation.
func makeSameTypeTensor(gctx *ggml.GraphContext, src ggml.Tensor) ggml.Tensor {
	return makeTensorFromSpec(gctx, ggml.TensorType(src), src.Ne(0), src.Ne(1), src.Ne(2), src.Ne(3))
}
