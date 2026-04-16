package arch

import (
	ggml "inference-lab-bench/internal/ggml"
)

// ModelReader abstracts loading a model from GGUF or safetensors.
// Everything above this boundary (NewGenericModel, ResolveParams, ResolveWeights)
// works only through this interface.
type ModelReader interface {
	// GGUFReader interface (metadata access)
	GetU32(key string) (uint32, bool)
	GetF32(key string) (float32, bool)
	GetArrInts(key string) ([]int, bool)
	GetArrBools(key string) ([]bool, bool)
	GetTensorDim(tensorName string, dim int) (int64, bool)

	// Tensor enumeration
	TensorCount() int
	TensorNames() []string
	TensorSpec(name string) (TensorSpec, bool)

	// Tensor data loading: reads raw bytes of the tensor into the provided buffer.
	// The buffer must be at least spec.Size bytes.
	ReadTensor(name string, buf []byte) error

	// MinMemoryRequired returns conservative estimates of memory (bytes) needed to
	// load this model and run inference at maxSeqLen. Values are intentionally
	// high-side so callers can reject models before OOM crashes.
	MinMemoryRequired(maxSeqLen int) MemReq

	// Cleanup
	Close() error
}

// MemReq represents estimated minimum memory required to load and run a model.
// Values are in bytes and are conservative (slightly high-side estimates).
type MemReq struct {
	// VRAM for model weights (GPU-resident tensors).
	WeightVRAM uint64
	// VRAM for KV/SSM cache at max sequence length.
	CacheVRAM uint64
	// RAM for non-GPU-resident data (model reader state, indices, temp buffers).
	OverheadRAM uint64
}

// Total returns the sum of all memory requirements.
func (m MemReq) TotalVRAM() uint64 {
	return m.WeightVRAM + m.CacheVRAM
}

// Total returns the sum of all memory requirements.
func (m MemReq) UnifiedRAM() uint64 {
	return m.WeightVRAM + m.CacheVRAM + m.OverheadRAM
}

// TensorSpec holds tensor metadata in format-agnostic form.
type TensorSpec struct {
	Type ggml.GGMLType // ggml type constant (e.g., ggml.TypeF16)
	Ne   [4]int64      // dimensions, padded to 4 (unused = 1)
	Size int           // total bytes in ggml format
}
