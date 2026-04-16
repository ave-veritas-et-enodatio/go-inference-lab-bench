package arch

import (
	"fmt"
	"os"
	"sort"

	ggufparser "github.com/gpustack/gguf-parser-go"

	"inference-lab-bench/internal/log"
)

// ---------------------------------------------------------------------------
// GGUF adapter
// ---------------------------------------------------------------------------

// ggufModelReader adapts a GGUF file to the ModelReader interface.
type ggufModelReader struct {
	*goGGUFReader   // metadata access (inlines GGUFReader methods)
	f               *os.File
	specs           map[string]tensorSpec // name → spec (Ne, Size, Type)
	tensorOffsets   map[string]uint64     // name → offset relative to TensorDataStartOffset
	tensorDataStart int64                 // absolute offset of tensor data region
	closed          bool
	archDef         *ArchDef
}

// NewModelReaderGGUF creates a ModelReader for a GGUF file.
// If gf is nil, parses the file. If gf is provided, uses it directly.
func NewModelReaderGGUF(archDef *ArchDef, modelPath string, gf *ggufparser.GGUFFile) (ModelReader, error) {
	if gf == nil {
		var err error
		gf, err = ggufparser.ParseGGUFFile(modelPath)
		if err != nil {
			return nil, fmt.Errorf("failed to parse GGUF: %w", err)
		}
	}
	if gf.TensorDataStartOffset == 0 && len(gf.TensorInfos) > 0 {
		return nil, fmt.Errorf("split GGUF files are not supported (use a single-file model)")
	}

	specs := buildTensorSpecs(gf)
	reader := &goGGUFReader{kvs: gf.Header.MetadataKV, tensorSpecs: specs}

	// Dump all GGUF metadata at debug level. Mirrors the paramValues dump in
	// NewModelReaderSafetensors so that loads of the same logical model in
	// each format can be diff'd key-for-key. Long arrays (vocab, merges,
	// token_type) are summarized by type+length to keep log lines bounded;
	// scalar values are printed inline.
	dumpGGUFMetadata(gf.Header.MetadataKV)

	offsets := make(map[string]uint64, len(gf.TensorInfos))
	for _, ti := range gf.TensorInfos {
		offsets[ti.Name] = ti.Offset
	}

	// Open the model file once for ReadTensor via ReadAt.
	f, err := os.Open(modelPath)
	if err != nil {
		return nil, fmt.Errorf("opening model file: %w", err)
	}

	return &ggufModelReader{
		goGGUFReader:    reader,
		f:               f,
		specs:           specs,
		tensorOffsets:   offsets,
		tensorDataStart: gf.TensorDataStartOffset,
		archDef:				 archDef,
	}, nil
}

func (r *ggufModelReader) TensorCount() int {
	return len(r.specs)
}

func (r *ggufModelReader) TensorNames() []string {
	names := make([]string, 0, len(r.specs))
	for n := range r.specs {
		names = append(names, n)
	}
	return names
}

func (r *ggufModelReader) TensorSpec(name string) (TensorSpec, bool) {
	s, ok := r.specs[name]
	if !ok {
		return TensorSpec{}, false
	}
	return TensorSpec{
		Type: s.Type,
		Ne:   s.Ne,
		Size: s.Size,
	}, true
}

func (r *ggufModelReader) ReadTensor(name string, buf []byte) error {
	if r.closed {
		return fmt.Errorf("reader closed")
	}
	offset, ok := r.tensorOffsets[name]
	if !ok {
		return fmt.Errorf("tensor %q not found", name)
	}
	spec, ok := r.specs[name]
	if !ok {
		return fmt.Errorf("tensor %q spec not found", name)
	}
	if len(buf) < spec.Size {
		return fmt.Errorf("buffer too small for tensor %q: need %d, have %d", name, spec.Size, len(buf))
	}
	absOff := r.tensorDataStart + int64(offset)
	_, err := r.f.ReadAt(buf[:spec.Size], absOff)
	if err != nil {
		return fmt.Errorf("reading tensor %q: %w", name, err)
	}
	return nil
}

func (r *ggufModelReader) MinMemoryRequired(maxSeqLen int) MemReq {
	// --- Weight VRAM ---
	var weightVRAM uint64
	for _, spec := range r.specs {
		weightVRAM += uint64(spec.Size)
	}

	// --- Cache VRAM ---
	cacheVRAM := r.estimateCacheVRAM(maxSeqLen)

	// --- Non-GPU RAM ---
	// Conservative: 64MB for GGUF file metadata, GGML context overhead, temp buffers.
	nonGPU := uint64(64 * 1024 * 1024)

	return MemReq{WeightVRAM: weightVRAM, CacheVRAM: cacheVRAM, OverheadRAM: nonGPU}
}

func (r *ggufModelReader) estimateCacheVRAM(maxSeqLen int) uint64 {
	arch := r.extractGGUFString("general.architecture")
	nLayers := int(r.extractGGUFUint64(arch + ".block_count"))
	nKVHeads := int(r.extractGGUFUint64(arch + ".attention.head_count_kv"))
	headDimK := int(r.extractGGUFUint64(arch + ".attention.key_length"))
	headDimV := int(r.extractGGUFUint64(arch + ".attention.value_length"))

	if nLayers == 0 || nKVHeads == 0 || headDimK == 0 || headDimV == 0 {
		// Can't parse metadata — use conservative estimate: ~20% of weight VRAM.
		totalWeight := uint64(0)
		for _, spec := range r.specs {
			totalWeight += uint64(spec.Size)
		}
		if totalWeight == 0 {
			return 0
		}
		return totalWeight / 5
	}

	// Per-layer KV cache: K + V tensors, each [headDim, nKVHeads, maxSeqLen] in F32 (4 bytes).
	perTokenPerLayer := 2 * (headDimK + headDimV) * nKVHeads * 4
	cacheVRAM := uint64(maxSeqLen * nLayers * perTokenPerLayer)

	_, hasSSM := r.archDef.Blocks["recurrent_ssm"]

	// SSM state: models using SSM have per-layer conv state — estimate as similar footprint to KV.
	// if arch == "qwen35" || arch == "llada" || arch == "llada-moe" {
	if hasSSM {
		cacheVRAM *= 2
	}

	// 20% overhead for alignment, scratch buffers, and graph construction.
	cacheVRAM = cacheVRAM * 6 / 5
	return cacheVRAM
}

func (r *ggufModelReader) extractGGUFString(key string) string {
	kv, ok := r.kvs.Get(key)
	if !ok {
		return ""
	}
	if s, ok := kv.Value.(string); ok {
		return s
	}
	return ""
}

func (r *ggufModelReader) extractGGUFUint64(key string) uint64 {
	kv, ok := r.kvs.Get(key)
	if !ok {
		return 0
	}
	switch v := kv.Value.(type) {
	case uint8:
		return uint64(v)
	case uint16:
		return uint64(v)
	case uint32:
		return uint64(v)
	case uint64:
		return v
	case int8:
		return uint64(v)
	case int16:
		return uint64(v)
	case int32:
		return uint64(v)
	case int64:
		return uint64(v)
	case float32:
		return uint64(v)
	case float64:
		return uint64(v)
	}
	return 0
}

// dumpGGUFMetadata logs every GGUF metadata KV at debug level, sorted by key.
// Long arrays (vocab, merges, token types) are summarized by element type and
// length rather than printed in full — a verbose dump would emit megabyte-sized
// log lines for 150k-entry token tables with no diagnostic value.
func dumpGGUFMetadata(kvs ggufparser.GGUFMetadataKVs) {
	keys := make([]string, len(kvs))
	byKey := make(map[string]ggufparser.GGUFMetadataKV, len(kvs))
	for i, kv := range kvs {
		keys[i] = kv.Key
		byKey[kv.Key] = kv
	}
	sort.Strings(keys)
	for _, k := range keys {
		kv := byKey[k]
		if kv.ValueType == ggufparser.GGUFMetadataValueTypeArray {
			av := kv.ValueArray()
			log.Debug("[param] %s = [%s × %d] (array)", k, av.Type.String(), av.Len)
			continue
		}
		log.Debug("[param] %s = %v (type %s)", k, kv.Value, kv.ValueType.String())
	}
}

func (r *ggufModelReader) Close() error {
	if r.closed {
		return nil
	}
	r.closed = true
	if r.f != nil {
		return r.f.Close()
	}
	return nil
}
