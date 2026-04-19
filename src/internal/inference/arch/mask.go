package arch

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

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

// MaskedLayer returns a weight map for the given layer with culling applied.
// Culled tensors (in ZeroTensors) are omitted from the returned map — the zero-value ggml.Tensor
// satisfies IsNil(), so the graph builder skips their ops unchanged.
// No mask: returns the raw weight map from the store (zero overhead).
func (m *GenericModel) MaskedLayer(layerIdx int, mask *CullingMask) map[string]ggml.Tensor {
	lt := m.Store.Layer(layerIdx)
	if lt == nil {
		log.Error("MaskedLayer: layer %d out of bounds (no layer data available)", layerIdx)
		return nil
	}
	if mask == nil || len(mask.zeroSet) == 0 {
		return lt
	}

	masked := make(map[string]ggml.Tensor, len(lt))
	lw := m.Weights.Layers[layerIdx]
	for name, tensor := range lt {
		ggufName := resolveGGUFName(lw, name)

		// Whole-tensor culling: omit from map (zero-value Tensor -> IsNil() -> skip).
		if mask.zeroSet[ggufName] {
			continue
		}

		masked[name] = tensor
	}
	return masked
}

// MaskedGlobal returns a global weight tensor from the store.
// Global tensors (token_embd, output_norm, output) are never culled — culling them
// would break inference entirely.
func (m *GenericModel) MaskedGlobal(_ *ggml.GraphContext, name string) ggml.Tensor {
	return m.Store.Global(name)
}

// resolveGGUFName maps a logical weight name (as stored in WeightStore) to its GGUF tensor name.
// FFN and FFNAlt tensors are stored with an "ffn_" prefix; both maps are checked.
func resolveGGUFName(lw ResolvedLayerWeights, logicalName string) string {
	if n, ok := lw.Common[logicalName]; ok {
		return n
	}
	if n, ok := lw.Block[logicalName]; ok {
		return n
	}
	if after, ok := strings.CutPrefix(logicalName, PrefixFFNWeight); ok {
		if n, ok := lw.FFN[after]; ok {
			return n
		}
		if n, ok := lw.FFNAlt[after]; ok {
			return n
		}
	}
	return ""
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
