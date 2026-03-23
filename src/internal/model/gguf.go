package model

import (
	"fmt"

	ggufparser "github.com/gpustack/gguf-parser-go"
)

// TensorInfo holds the location and shape of a single tensor in the GGUF file.
type TensorInfo struct {
	Name   string
	Offset uint64 // byte offset from start of tensor data section
	Shape  []uint64
	DType  string
}

// GGUFMetadata holds parsed model metadata and the tensor index.
// Supported architectures
// supportedArchitectures is populated at runtime by scanArchDefinitions.
var supportedArchitectures = map[string]bool{}

type GGUFMetadata struct {
	Architecture string // e.g. "qwen35", "qwen35moe"

	// Model architecture fields
	NLayers    int
	NHeads     int
	NKVHeads   int
	HiddenDim  int
	FFNDim     int
	VocabSize  int
	ContextLen int

	// Tensor index: name → info
	Tensors map[string]TensorInfo

	// Raw KV metadata for tokenizer extraction
	RawMeta map[string]any
}

func ParseGGUF(path string) (*GGUFMetadata, error) {
	f, err := ggufparser.ParseGGUFFile(path)
	if err != nil {
		return nil, fmt.Errorf("parsing gguf %s: %w", path, err)
	}

	meta := &GGUFMetadata{
		Tensors: make(map[string]TensorInfo),
		RawMeta: make(map[string]any),
	}

	// Extract architecture metadata
	arch := f.Architecture()
	meta.Architecture = arch.Architecture
	meta.NLayers = int(arch.BlockCount)
	meta.NHeads = int(arch.AttentionHeadCount)
	meta.NKVHeads = int(arch.AttentionHeadCountKV)
	meta.HiddenDim = int(arch.EmbeddingLength)
	if len(arch.FeedForwardLength) > 0 {
		meta.FFNDim = int(arch.FeedForwardLength[0])
	}
	meta.VocabSize = int(f.Tokenizer().TokensLength)
	meta.ContextLen = int(arch.MaximumContextLength)

	// Build tensor index
	for _, ti := range f.TensorInfos {
		shape := make([]uint64, len(ti.Dimensions))
		copy(shape, ti.Dimensions)
		meta.Tensors[ti.Name] = TensorInfo{
			Name:   ti.Name,
			Offset: ti.Offset,
			Shape:  shape,
			DType:  ti.Type.String(),
		}
	}

	return meta, nil
}
