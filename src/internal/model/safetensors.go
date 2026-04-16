package model

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"

	"inference-lab-bench/internal/inference/arch"
	log "inference-lab-bench/internal/log"
)

// ParseSafetensorsDir parses a safetensors model directory and returns
// GGUFMetadata-compatible metadata. The safetensors index supplies the
// tensor inventory; config.json (if present) supplies numeric model params
// and the HuggingFace architecture class name used for stmap lookup.
//
// Error semantics:
//   - No index.json or model.safetensors → error (cannot list without it)
//   - No config.json → partial metadata (Architecture="unknown", numeric fields=0)
//   - Corrupt config.json → log debug warning, continue with partial data
//   - Corrupt index.json → error (let the index parser handle this)
func ParseSafetensorsDir(stDir string, archDir string) (*GGUFMetadata, error) {
	// Try index.json first (sharded models)
	idxPath := filepath.Join(stDir, "model.safetensors.index.json")
	if _, err := os.Stat(idxPath); err != nil {
		// Fallback: single model.safetensors file
		singlePath := filepath.Join(stDir, "model.safetensors")
		if _, err := os.Stat(singlePath); err != nil {
			return nil, fmt.Errorf("no safetensors index or model file in %s", stDir)
		}
	}

	idx, err := arch.LoadSafetensorsIndex(stDir)
	if err != nil {
		return nil, fmt.Errorf("loading safetensors index: %w", err)
	}

	meta := &GGUFMetadata{
		Architecture: "unknown",
		Tensors:      make(map[string]TensorInfo),
		RawMeta:      make(map[string]any),
	}

	// Convert STTensorEntry → TensorInfo
	// safetensors has no unified offset concept; set offset=0 for all entries.
	for name, entry := range idx.Tensors {
		shape := make([]uint64, len(entry.Shape))
		for i, s := range entry.Shape {
			shape[i] = uint64(s)
		}
		meta.Tensors[name] = TensorInfo{
			Name:   name,
			Offset: 0,
			Shape:  shape,
			DType:  entry.Dtype,
		}
	}

	// Attempt to read config.json for architecture and numeric params.
	meta = parseConfigJSON(stDir, meta)

	// Resolve architecture via stmap lookup.
	if meta.Architecture != "unknown" {
		archName, _, err := arch.FindSTMapByHFClass(archDir, meta.Architecture)
		if err != nil {
			log.Debug("stmap lookup for safetensors dir %s: %v", stDir, err)
			// Keep the HF class name as-is; resolution will fail downstream.
		} else if archName != "" {
			meta.Architecture = archName
		}
	}

	return meta, nil
}

// parseConfigJSON reads config.json from the safetensors directory and
// extracts architecture info and numeric model parameters into meta.
func parseConfigJSON(stDir string, meta *GGUFMetadata) *GGUFMetadata {
	configPath := filepath.Join(stDir, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		if os.IsNotExist(err) {
			return meta
		}
		log.Debug("safetensors dir %s: read config.json: %v", stDir, err)
		return meta
	}

	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		log.Debug("safetensors dir %s: corrupt config.json, continuing with partial data: %v", stDir, err)
		return meta
	}

	// Extract HF class name from architectures[0].
	if arches, ok := raw["architectures"].([]any); ok && len(arches) > 0 {
		if hfClass, ok := arches[0].(string); ok && hfClass != "" {
			meta.Architecture = hfClass
		}
	}

	// Extract numeric fields from config.json.
	meta.NLayers = extractConfigInt(raw, "num_hidden_layers")
	meta.VocabSize = extractConfigInt(raw, "vocab_size")
	meta.NHeads = extractConfigInt(raw, "num_attention_heads")
	meta.NKVHeads = extractConfigInt(raw, "num_key_value_heads")
	meta.HiddenDim = extractConfigInt(raw, "hidden_size")
	meta.FFNDim = extractConfigInt(raw, "intermediate_size")
	meta.ContextLen = extractConfigInt(raw, "max_position_embeddings")

	return meta
}

func extractConfigInt(raw map[string]any, key string) int {
	v, ok := raw[key]
	if !ok {
		return 0
	}
	switch n := v.(type) {
	case float64: // JSON numbers parse as float64
		return int(n)
	case int:
		return n
	case int64:
		return int(n)
	}
	// Also handle values serialized as strings (e.g. "128").
	s, ok := v.(string)
	if ok {
		if n, err := strconv.Atoi(s); err == nil {
			return n
		}
	}
	return 0
}
