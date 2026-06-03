package arch

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
)

// SafetensorsIndex is the parsed view of a safetensors model directory:
// per-tensor location (dtype, shape, shard, byte range) plus the ordered
// shard list that the reader opens for ReadAt access.
type SafetensorsIndex struct {
	// Tensors maps tensor name -> entry.
	Tensors map[string]*STTensorEntry
	// Shards is the ordered list of shard filenames, sorted by shard index.
	Shards []string
	// stDir is the directory containing the safetensors files (for later loading).
	stDir string
}

// STTensorEntry is a single tensor entry from a shard header.
type STTensorEntry struct {
	Dtype      string  // "F16", "BF16", "F32", "I32", etc.
	DataOffset int64   // byte offset within the shard file (absolute file position)
	DataSize   int64   // byte length of the tensor data
	Shard      string  // shard filename (e.g., "model-00001-of-00002.safetensors")
	Shape      []int64 // tensor shape
}

// shardNamePattern matches the sharded safetensors filename conventions seen
// in the wild:
//   - `model-NNNNN-of-MMMMM.safetensors`             (standard HF)
//   - `model.safetensors-NNNNN-of-MMMMM.safetensors` (some uploaders/tools keep
//     the full `model.safetensors` base and append the shard suffix)
// N and M are the 1-based shard index and total shard count. Digit width is not
// fixed (some uploaders zero-pad to 5; accept any width to stay robust). The
// `model.safetensors` base variant is matched via an optional `.safetensors`
// segment before the `-N-of-M` suffix; the single-file `model.safetensors`
// (no `-N-of-M`) does not match and is handled separately by exact-name check.
var shardNamePattern = regexp.MustCompile(`^model(?:\.safetensors)?-(\d+)-of-(\d+)\.safetensors$`)

// LoadSafetensorsIndex discovers and parses the safetensors files in a
// model directory. Two layouts are accepted:
//
//  1. A single `model.safetensors` file (unsharded HF upload).
//  2. A set of sharded files matching shardNamePattern — both the standard
//     `model-N-of-M.safetensors` and the `model.safetensors-N-of-M.safetensors`
//     variant some uploaders produce.
//
// Shard discovery is done by directory glob — the historical
// `model.safetensors.index.json` is intentionally not consulted. The index
// JSON is a redundant routing table whose contents are derivable from the
// shard headers themselves, and HF uploads ship it inconsistently. The glob
// path also enforces a contiguous 1..M shard set from the filename pattern
// alone, which catches a missing tail shard at the same layer the index
// would have.
//
// A missing required weight (truncated upload, wrong arch) is still caught
// downstream by per-layer required-weight resolution against the arch
// definition; this function only enforces shard-set contiguity.
func LoadSafetensorsIndex(stDir string) (*SafetensorsIndex, error) {
	shards, err := discoverShards(stDir)
	if err != nil {
		return nil, err
	}

	idx := &SafetensorsIndex{
		Tensors: make(map[string]*STTensorEntry),
		Shards:  shards,
		stDir:   stDir,
	}
	for _, shard := range shards {
		if err := parseShardHeader(idx, shard); err != nil {
			return nil, err
		}
	}
	return idx, nil
}

// discoverShards inspects stDir and returns the ordered shard filename list.
// Returns a single-element list for an unsharded model.safetensors, or the
// 1..M ordered shard list parsed out of `model-N-of-M.safetensors` names.
// A partial shard set (gaps, mismatched M, single shard "1-of-1" without
// peers) is rejected with a named error.
func discoverShards(stDir string) ([]string, error) {
	entries, err := os.ReadDir(stDir)
	if err != nil {
		return nil, fmt.Errorf("safetensors: read dir %s: %w", stDir, err)
	}

	var (
		hasSingle bool
		shardByN  = map[int]string{}
		shardM    = -1 // expected total shard count; -1 == not yet set
	)
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if name == "model.safetensors" {
			hasSingle = true
			continue
		}
		m := shardNamePattern.FindStringSubmatch(name)
		if m == nil {
			continue
		}
		n, _ := strconv.Atoi(m[1]) // regex guarantees digits
		total, _ := strconv.Atoi(m[2])
		if shardM == -1 {
			shardM = total
		} else if total != shardM {
			return nil, fmt.Errorf("safetensors: %s declares %d-of-%d but earlier shard declared M=%d (mixed shard sets)", name, n, total, shardM)
		}
		if prev, dup := shardByN[n]; dup {
			return nil, fmt.Errorf("safetensors: duplicate shard index %d in %s (%s and %s)", n, stDir, prev, name)
		}
		shardByN[n] = name
	}

	hasSharded := shardM > 0
	switch {
	case hasSingle && hasSharded:
		return nil, fmt.Errorf("safetensors: %s contains both model.safetensors and model-N-of-M.safetensors shards (ambiguous layout)", stDir)
	case hasSingle:
		return []string{"model.safetensors"}, nil
	case hasSharded:
		if len(shardByN) != shardM {
			missing := make([]int, 0, shardM-len(shardByN))
			for n := 1; n <= shardM; n++ {
				if _, ok := shardByN[n]; !ok {
					missing = append(missing, n)
				}
			}
			return nil, fmt.Errorf("safetensors: incomplete shard set in %s (have %d, expected %d-of-%d; missing %v)", stDir, len(shardByN), shardM, shardM, missing)
		}
		ns := make([]int, 0, shardM)
		for n := range shardByN {
			ns = append(ns, n)
		}
		sort.Ints(ns)
		shards := make([]string, len(ns))
		for i, n := range ns {
			shards[i] = shardByN[n]
		}
		return shards, nil
	default:
		return nil, fmt.Errorf("safetensors: no model.safetensors or model-N-of-M.safetensors files in %s", stDir)
	}
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
