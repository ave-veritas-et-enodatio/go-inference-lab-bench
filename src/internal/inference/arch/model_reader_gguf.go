package arch

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	ggufparser "github.com/gpustack/gguf-parser-go"

	"inference-lab-bench/internal/log"
)

// ---------------------------------------------------------------------------
// GGUF adapter
// ---------------------------------------------------------------------------

// ggufModelReader adapts a GGUF file to the ModelReader interface.
//
// When `mmproj` is non-nil, this reader chains a paired vision/audio
// sidecar GGUF (created by `convert_hf_to_gguf.py --mmproj`). Lookups
// fall through to the mmproj when the primary decoder GGUF doesn't
// resolve a tensor or metadata key. Key aliasing translates our logical
// `vision.*` namespace into the mmproj's upstream `clip.vision.*` form.
type ggufModelReader struct {
	*goGGUFReader   // metadata access (inlines GetU32/F32/ArrInts/ArrBools/TensorDim)
	f               *os.File
	specs           map[string]tensorSpec // name → spec (Ne, Size, Type)
	tensorOffsets   map[string]uint64     // name → offset relative to TensorDataStartOffset
	tensorDataStart int64                 // absolute offset of tensor data region
	closed          bool
	archDef         *ArchDef

	// mmproj is the paired vision/audio sidecar reader. nil for
	// text-only decoder GGUFs. Tensor names in the mmproj follow the
	// `v.blk.*`, `v.patch_embd`, `v.position_embd`, `mm.*` upstream
	// convention — which happens to match our arch.toml's [vision] /
	// [projector] declared names exactly, so no name remapping needed.
	mmproj *ggufModelReader
	// mmprojPath is the absolute path; retained for clamp-scalar
	// loading via separate file I/O.
	mmprojPath string
}

// NewModelReaderGGUF creates a ModelReader for a GGUF file.
// If gf is nil, parses the file. If gf is provided, uses it directly.
// mmprojPath, when non-empty, points at a paired vision/audio sidecar
// GGUF; its tensors and metadata are made available through the same
// reader so the loader treats decoder + sidecar as one logical model.
func NewModelReaderGGUF(archDef *ArchDef, modelPath string, gf *ggufparser.GGUFFile, mmprojPath string) (ModelReader, error) {
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

	log.Info("ModelReader[gguf] created for %s", filepath.Base(modelPath))

	primary := &ggufModelReader{
		goGGUFReader:    reader,
		f:               f,
		specs:           specs,
		tensorOffsets:   offsets,
		tensorDataStart: gf.TensorDataStartOffset,
		archDef:         archDef,
	}

	// Load the paired mmproj sidecar (vision/audio tower + projector)
	// when present. A failure to parse the sidecar is logged and treated
	// as no-sidecar — the model still loads as decoder-only, the engine
	// gates vision setup on `vision.has_encoder` which won't resolve.
	if mmprojPath != "" {
		mmFile, err := ggufparser.ParseGGUFFile(mmprojPath)
		if err != nil {
			log.Warn("mmproj sidecar %s: parse failed, continuing decoder-only: %v",
				filepath.Base(mmprojPath), err)
		} else if mmFile.TensorDataStartOffset == 0 && len(mmFile.TensorInfos) > 0 {
			log.Warn("mmproj sidecar %s: split-file GGUF unsupported, continuing decoder-only",
				filepath.Base(mmprojPath))
		} else {
			mmSpecs := buildTensorSpecs(mmFile)
			mmOffsets := make(map[string]uint64, len(mmFile.TensorInfos))
			for _, ti := range mmFile.TensorInfos {
				mmOffsets[ti.Name] = ti.Offset
			}
			mmF, ferr := os.Open(mmprojPath)
			if ferr != nil {
				log.Warn("mmproj sidecar %s: open failed: %v", filepath.Base(mmprojPath), ferr)
			} else {
				dumpGGUFMetadata(mmFile.Header.MetadataKV)
				primary.mmproj = &ggufModelReader{
					goGGUFReader: &goGGUFReader{
						kvs:         mmFile.Header.MetadataKV,
						tensorSpecs: mmSpecs,
					},
					f:               mmF,
					specs:           mmSpecs,
					tensorOffsets:   mmOffsets,
					tensorDataStart: mmFile.TensorDataStartOffset,
				}
				primary.mmprojPath = mmprojPath
				log.Info("mmproj sidecar attached: %s (%d tensors)",
					filepath.Base(mmprojPath), len(mmSpecs))
			}
		}
	}

	return primary, nil
}

// ggufMmprojKeyAliases returns the mmproj-side GGUF key(s) to try when
// a primary lookup misses. mmproj sidecars use the upstream `clip.*`
// namespace while our arch.toml [vision.params] keys are `vision.*` —
// this bridges the gap without requiring per-format stmap variants.
//
//   - "vision.has_encoder"  → "clip.has_vision_encoder"
//   - "vision.<anything>"   → "clip.vision.<anything>"
//
// All other keys are not aliased (the alias path only fires when the
// primary lookup fails and a sidecar is attached, so this is a no-op
// for decoder-only metadata).
func ggufMmprojKeyAliases(key string) []string {
	if key == KeyVisionHasEncoder {
		return []string{GGUFKeyClipHasVisionEncoder}
	}
	if strings.HasPrefix(key, "vision.") {
		return []string{"clip." + key}
	}
	return nil
}

// GetU32 overrides the embedded goGGUFReader's GetU32 to add mmproj
// sidecar fallback with key aliasing. Same for GetF32, GetArrInts,
// GetArrBools, GetTensorDim below.
func (r *ggufModelReader) GetU32(key string) (uint32, bool) {
	if v, ok := r.goGGUFReader.GetU32(key); ok {
		return v, true
	}
	if r.mmproj != nil {
		if v, ok := r.mmproj.goGGUFReader.GetU32(key); ok {
			return v, true
		}
		for _, alias := range ggufMmprojKeyAliases(key) {
			if v, ok := r.mmproj.goGGUFReader.GetU32(alias); ok {
				return v, true
			}
			// `vision.has_encoder` is BOOL in mmproj — promote to U32.
			if alias == GGUFKeyClipHasVisionEncoder {
				if b, ok := r.mmproj.goGGUFReader.getBool(alias); ok {
					if b {
						return 1, true
					}
					return 0, true
				}
			}
		}
	}
	return 0, false
}

func (r *ggufModelReader) GetF32(key string) (float32, bool) {
	if v, ok := r.goGGUFReader.GetF32(key); ok {
		return v, true
	}
	if r.mmproj != nil {
		if v, ok := r.mmproj.goGGUFReader.GetF32(key); ok {
			return v, true
		}
		for _, alias := range ggufMmprojKeyAliases(key) {
			if v, ok := r.mmproj.goGGUFReader.GetF32(alias); ok {
				return v, true
			}
		}
	}
	return 0, false
}

func (r *ggufModelReader) GetArrInts(key string) ([]int, bool) {
	if v, ok := r.goGGUFReader.GetArrInts(key); ok {
		return v, true
	}
	if r.mmproj != nil {
		if v, ok := r.mmproj.goGGUFReader.GetArrInts(key); ok {
			return v, true
		}
		for _, alias := range ggufMmprojKeyAliases(key) {
			if v, ok := r.mmproj.goGGUFReader.GetArrInts(alias); ok {
				return v, true
			}
		}
	}
	return nil, false
}

func (r *ggufModelReader) GetArrBools(key string) ([]bool, bool) {
	if v, ok := r.goGGUFReader.GetArrBools(key); ok {
		return v, true
	}
	if r.mmproj != nil {
		if v, ok := r.mmproj.goGGUFReader.GetArrBools(key); ok {
			return v, true
		}
		for _, alias := range ggufMmprojKeyAliases(key) {
			if v, ok := r.mmproj.goGGUFReader.GetArrBools(alias); ok {
				return v, true
			}
		}
	}
	return nil, false
}

func (r *ggufModelReader) GetTensorDim(name string, dim int) (int64, bool) {
	if v, ok := r.goGGUFReader.GetTensorDim(name, dim); ok {
		return v, true
	}
	if r.mmproj != nil {
		if v, ok := r.mmproj.goGGUFReader.GetTensorDim(name, dim); ok {
			return v, true
		}
	}
	return 0, false
}

func (r *ggufModelReader) TensorCount() int {
	n := len(r.specs)
	if r.mmproj != nil {
		n += len(r.mmproj.specs)
	}
	return n
}

func (r *ggufModelReader) TensorNames() []string {
	cap := len(r.specs)
	if r.mmproj != nil {
		cap += len(r.mmproj.specs)
	}
	names := make([]string, 0, cap)
	for n := range r.specs {
		names = append(names, n)
	}
	if r.mmproj != nil {
		for n := range r.mmproj.specs {
			names = append(names, n)
		}
	}
	return names
}

func (r *ggufModelReader) TensorSpec(name string) (TensorSpec, bool) {
	if s, ok := r.specs[name]; ok {
		return TensorSpec{Type: s.Type, Ne: s.Ne, Size: s.Size}, true
	}
	if r.mmproj != nil {
		if s, ok := r.mmproj.specs[name]; ok {
			return TensorSpec{Type: s.Type, Ne: s.Ne, Size: s.Size}, true
		}
	}
	return TensorSpec{}, false
}

func (r *ggufModelReader) ReadTensor(name string, buf []byte) error {
	if r.closed {
		return fmt.Errorf("reader closed")
	}
	// Dispatch to whichever file holds this tensor's data.
	if _, ok := r.specs[name]; ok {
		return r.readFromBacking(r.f, r.tensorDataStart, r.tensorOffsets, r.specs, name, buf)
	}
	if r.mmproj != nil {
		if _, ok := r.mmproj.specs[name]; ok {
			return r.readFromBacking(r.mmproj.f, r.mmproj.tensorDataStart,
				r.mmproj.tensorOffsets, r.mmproj.specs, name, buf)
		}
	}
	return fmt.Errorf("tensor %q not found", name)
}

func (r *ggufModelReader) readFromBacking(f *os.File, dataStart int64,
	offsets map[string]uint64, specs map[string]tensorSpec,
	name string, buf []byte) error {
	offset, ok := offsets[name]
	if !ok {
		return fmt.Errorf("tensor %q offset missing", name)
	}
	spec, ok := specs[name]
	if !ok {
		return fmt.Errorf("tensor %q spec missing", name)
	}
	if len(buf) < spec.Size {
		return fmt.Errorf("buffer too small for tensor %q: need %d, have %d", name, spec.Size, len(buf))
	}
	absOff := dataStart + int64(offset)
	_, err := f.ReadAt(buf[:spec.Size], absOff)
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

	// SSM state: models with blocks declaring conv/SSM cache have per-layer
	// recurrent state — estimate as similar footprint to KV.
	if r.archDef.HasRecurrentCache() {
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
	var firstErr error
	if r.f != nil {
		if err := r.f.Close(); err != nil {
			firstErr = err
		}
	}
	if r.mmproj != nil && r.mmproj.f != nil {
		if err := r.mmproj.f.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}
