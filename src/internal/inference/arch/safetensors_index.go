package arch

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

const maxIndexFileSize = 500 * 1024 // 500KB guard

// SafetensorsIndex holds the parsed index from model.safetensors.index.json.
type SafetensorsIndex struct {
	// Metadata is the "metadata" field from the index JSON, if present.
	Metadata map[string]any
	// Tensors maps tensor name -> entry.
	Tensors map[string]*STTensorEntry
	// Shards is an ordered list of unique shard filenames referenced by tensors.
	Shards []string
	// stDir is the directory containing the safetensors files (for later loading).
	stDir string
}

// STTensorEntry is a single tensor entry from the index.
type STTensorEntry struct {
	Dtype      string  // "F16", "BF16", "F32", "I32", etc.
	DataOffset int64   // byte offset within the shard file (absolute file position)
	DataSize   int64   // byte length of the tensor data
	Shard      string  // shard filename (e.g., "model-00001-of-00002.safetensors")
	Shape      []int64 // tensor shape
}

// LoadSafetensorsIndex loads and parses a safetensors model directory.
//
// It handles both sharded models (model.safetensors.index.json + shard files)
// and non-sharded models (single model.safetensors file with no index).
func LoadSafetensorsIndex(stDir string) (*SafetensorsIndex, error) {
	idxPath := filepath.Join(stDir, "model.safetensors.index.json")

	_, statErr := os.Stat(idxPath)
	if statErr == nil {
		return loadSharded(stDir, idxPath)
	}

	// No index file — check for a single non-sharded model.safetensors file.
	singlePath := filepath.Join(stDir, "model.safetensors")
	if _, err := os.Stat(singlePath); err == nil {
		return loadSingleShard(stDir, singlePath)
	}

	return nil, fmt.Errorf("safetensors index: neither %s nor %s found", idxPath, singlePath)
}

// loadSharded handles the index.json + multi-shard case.
func loadSharded(stDir, idxPath string) (*SafetensorsIndex, error) {
	// Guard against unreasonably large index files.
	info, err := os.Stat(idxPath)
	if err != nil {
		return nil, fmt.Errorf("safetensors index: stat %s: %w", idxPath, err)
	}
	if info.Size() > maxIndexFileSize {
		return nil, fmt.Errorf("safetensors index: file %s too large (%d bytes, limit %d)", idxPath, info.Size(), maxIndexFileSize)
	}

	data, err := os.ReadFile(idxPath)
	if err != nil {
		return nil, fmt.Errorf("safetensors index: read %s: %w", idxPath, err)
	}

	var rawIndex struct {
		Metadata  map[string]any    `json:"metadata"`
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(data, &rawIndex); err != nil {
		return nil, fmt.Errorf("safetensors index: parse %s: %w", idxPath, err)
	}

	idx := &SafetensorsIndex{
		Metadata: rawIndex.Metadata,
		Tensors:  make(map[string]*STTensorEntry),
		stDir:    stDir,
	}

	// Build ordered unique shard list from weight_map values.
	seen := make(map[string]bool)
	for _, shard := range rawIndex.WeightMap {
		if !seen[shard] {
			seen[shard] = true
			idx.Shards = append(idx.Shards, shard)
		}
	}

	// Parse each shard's JSON header and populate tensor entries.
	for _, shard := range idx.Shards {
		if err := parseShardHeader(idx, shard); err != nil {
			return nil, err
		}
	}

	return idx, nil
}

// loadSingleShard handles a non-sharded model with just model.safetensors.
func loadSingleShard(stDir, shardPath string) (*SafetensorsIndex, error) {
	idx := &SafetensorsIndex{
		Tensors: make(map[string]*STTensorEntry),
		Shards:  []string{"model.safetensors"},
		stDir:   stDir,
	}
	if err := parseShardHeader(idx, "model.safetensors"); err != nil {
		return nil, err
	}
	return idx, nil
}

// parseShardHeader reads and parses the JSON header from a single shard file,
// populating idx.Tensors with entries for that shard.
func parseShardHeader(idx *SafetensorsIndex, shardName string) error {
	shardPath := filepath.Join(idx.stDir, shardName)

	f, err := os.Open(shardPath)
	if err != nil {
		return fmt.Errorf("safetensors index: open shard %s: %w", shardPath, err)
	}
	defer f.Close()

	// Read 8-byte header length (uint64 LE).
	var jsonLenBuf [8]byte
	if _, err := f.Read(jsonLenBuf[:]); err != nil {
		return fmt.Errorf("safetensors index: read header length from %s: %w", shardName, err)
	}
	jsonHeaderLen := int64(binary.LittleEndian.Uint64(jsonLenBuf[:]))

	// Read the JSON header.
	jsonData := make([]byte, jsonHeaderLen)
	if _, err := f.Read(jsonData); err != nil {
		return fmt.Errorf("safetensors index: read JSON header from %s: %w", shardName, err)
	}

	// Parse JSON header: map of tensor_name -> {dtype, shape, data_offsets}
	var header map[string]json.RawMessage
	if err := json.Unmarshal(jsonData, &header); err != nil {
		return fmt.Errorf("safetensors index: parse shard header %s: %w", shardName, err)
	}

	for name, raw := range header {
		// Skip __metadata__ entries.
		if name == "__metadata__" {
			continue
		}

		var entry struct {
			Dtype       string  `json:"dtype"`
			Shape       []int64 `json:"shape"`
			DataOffsets []int64 `json:"data_offsets"`
		}
		if err := json.Unmarshal(raw, &entry); err != nil {
			return fmt.Errorf("safetensors index: parse tensor %q in %s: %w", name, shardName, err)
		}

		if entry.Dtype == "" {
			return fmt.Errorf("safetensors index: tensor %q in %s: missing dtype", name, shardName)
		}
		// shape may be nil for 0-size tensors, which is fine.
		if entry.Shape == nil {
			entry.Shape = []int64{}
		}

		var dataOffset, dataSize int64
		if len(entry.DataOffsets) == 2 {
			dataOffset = 8 + jsonHeaderLen + entry.DataOffsets[0]
			dataSize = entry.DataOffsets[1] - entry.DataOffsets[0]
		}
		// If data_offsets is [0,0] or missing, dataSize remains 0 (metadata-only tensor).

		// Check for duplicate tensor names across shards.
		if _, exists := idx.Tensors[name]; exists {
			return fmt.Errorf("safetensors index: duplicate tensor %q (previously seen, also in %s)", name, shardName)
		}

		idx.Tensors[name] = &STTensorEntry{
			Dtype:      entry.Dtype,
			DataOffset: dataOffset,
			DataSize:   dataSize,
			Shard:      shardName,
			Shape:      entry.Shape,
		}
	}

	return nil
}
