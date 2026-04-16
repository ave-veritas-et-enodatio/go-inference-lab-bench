package arch

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"unsafe"

	"inference-lab-bench/internal/ggml"
	"inference-lab-bench/internal/log"
)

// stDtypeToGGML maps safetensors dtype strings to the *target* ggml type used
// to store the tensor after loading. BF16 source data is converted to F16 at
// load time (both are 2 bytes but with different bit layouts, so they cannot
// be byte-copied between each other).
var stDtypeToGGML = map[string]ggml.GGMLType{
	"F16":  ggml.TypeF16,
	"BF16": ggml.TypeF16, // converted to F16 at load time
	"F32":  ggml.TypeF32,
	"I32":  ggml.TypeI32,
}

// stDtypeToGGMLRaw returns the actual ggml type matching the safetensors
// source dtype (no F16-substitution for BF16). Used by the slow-path Cast
// pipeline which needs the actual source representation to feed into
// ggml_cast — distinct from stDtypeToGGML which returns the *target* type.
func stDtypeToGGMLRaw(dtype string) (ggml.GGMLType, bool) {
	switch dtype {
	case "F16":
		return ggml.TypeF16, true
	case "BF16":
		return ggml.TypeBF16, true
	case "F32":
		return ggml.TypeF32, true
	case "I32":
		return ggml.TypeI32, true
	}
	return 0, false
}

// Slow-path scratch arena sizing. Sized so that one chunk's working set
// (input + F32 intermediate + transform intermediates + output cast) fits
// comfortably with headroom. 4M elements × 4 bytes × ~6 live tensors ≈ 96MB.
const (
	stScratchArenaSize     = 256 * 1024 * 1024
	stMaxConvertChunkElems = 4 * 1024 * 1024
)

// neElements returns the total element count of a tensor described by ne[].
// Unused dimensions are 1 in the convention used by buildSTTensorSpecs.
func neElements(ne [4]int64) int64 {
	n := int64(1)
	for _, d := range ne {
		if d > 0 {
			n *= d
		}
	}
	return n
}

// stElementSize returns the byte size per element for a ggml type.
// Only covers types we expect from safetensors (unquantized).
func stElementSize(ggmlType ggml.GGMLType) int {
	switch ggmlType {
	case ggml.TypeF16:
		return 2
	case ggml.TypeBF16:
		return 2
	case ggml.TypeF32:
		return 4
	case ggml.TypeI32:
		return 4
	default:
		return 0
	}
}

// tensorSizeForType computes the ggml byte size from shape and ggml type.
func tensorSizeForType(ggmlType ggml.GGMLType, ne [4]int64) int {
	elemSize := stElementSize(ggmlType)
	if elemSize == 0 {
		return 0
	}
	total := int64(elemSize)
	for _, d := range ne {
		total *= d
	}
	return int(total)
}

// stModelReader adapts a safetensors directory to the ModelReader interface.
type stModelReader struct {
	index       *SafetensorsIndex
	stmap       *ArchSTMap
	paramValues map[string]any        // GGUF key → value (from config.json + stmap)
	tensorSpecs map[string]TensorSpec // name → precomputed spec
	shardFiles  map[string]*os.File   // shard filename → open file handle
	ggufToHF    map[string]string     // GGUF tensor name → HF tensor name
	archDef     *ArchDef

	// scratchCtx is an AllocPermAllow ggml context used as a chunked working
	// arena for the slow-path tensor conversion pipeline (cast + element-wise
	// transforms). Tensors allocated here are backed by ggml-aligned arena
	// memory, eliminating the alignment hazards of TensorSetData pointer
	// aliasing. The arena is rewound between chunks; descriptors are short
	// lived. Allocated in NewModelReaderSafetensors, freed in Close.
	scratchCtx *ggml.GraphContext
}

// NewModelReaderSafetensors creates a ModelReader for a safetensors directory.
// archDir is the directory containing .arch.stmap.toml files.
func NewModelReaderSafetensors(archDef *ArchDef, stDir string, archDir string) (ModelReader, error) {
	// 1. Parse the index
	index, err := LoadSafetensorsIndex(stDir)
	if err != nil {
		return nil, fmt.Errorf("loading safetensors index: %w", err)
	}

	// 2. Read config.json
	cfgPath := filepath.Join(stDir, "config.json")
	configJSON, err := loadConfigJSON(cfgPath)
	if err != nil {
		return nil, fmt.Errorf("loading config.json: %w", err)
	}

	// 3. Extract HF class from config.json
	hfClass, err := extractHFClass(configJSON)
	if err != nil {
		return nil, fmt.Errorf("extracting HF class: %w", err)
	}

	// 4. Find stmap by HF class
	archName, stmap, err := FindSTMapByHFClass(archDir, hfClass)
	if err != nil {
		return nil, fmt.Errorf("finding stmap for HF class %q: %w", hfClass, err)
	}
	if stmap == nil {
		return nil, fmt.Errorf("no stmap found for HF class %q", hfClass)
	}

	// 5. Load the stmap file (redundant if FindSTMapByHFClass already did, but safe)
	stmap, err = LoadArchSTMap(archDir, archName)
	if err != nil {
		return nil, fmt.Errorf("loading stmap %q: %w", archName, err)
	}
	if stmap == nil {
		return nil, fmt.Errorf("stmap %q not found", archName)
	}

	// 6. Build param value map from stmap params + config.json
	paramValues := buildParamValues(archDef, stmap, configJSON)

	// Inject literal GGUF metadata values from the stmap.
	// These are architecture-level constants that only exist in the GGUF
	// metadata section (injected by llama.cpp) and have no config.json equivalent.
	for key, val := range stmap.Metadata {
		paramValues[key] = val
	}

	// Dump all resolved params at debug level. Mirrored by an equivalent dump
	// in NewModelReaderGGUF so that two model loads (same model, different
	// formats) can be diff'd key-for-key to catch param-coercion regressions
	// at a glance. Kept around permanently: the cost is a single sorted walk
	// at load time, and the rms_eps=0 / rope_freq_base=0 safetensors bug
	// would have been caught instantly by this dump.
	keys := make([]string, 0, len(paramValues))
	for k := range paramValues {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		log.Debug("[param] %s = %v (type %T)", k, paramValues[k], paramValues[k])
	}

	// 7. Build tensor specs (precompute ggml type and size)
	tensorSpecs := buildSTTensorSpecs(index)

	// 8. Build GGUF → HF tensor name translation map
	ggufToHF, err := buildGGUFToHFMap(stmap, index)
	if err != nil {
		return nil, fmt.Errorf("building GGUF→HF name map: %w", err)
	}
	// Size sanity check — if the stmap is incomplete the map will be
	// noticeably smaller than the index and downstream tensor-not-found errors
	// will cite this log line as the root cause.
	log.Debug("[st] ggufToHF has %d entries (safetensors index has %d tensors)", len(ggufToHF), len(index.Tensors))

	// 9. Allocate the slow-path scratch ggml context. Done before opening shard
	// files so that an allocation failure does not leak file handles.
	scratchCtx := ggml.NewGraphContext(stScratchArenaSize, ggml.AllocPermAllow)
	if scratchCtx == nil {
		return nil, fmt.Errorf("allocating safetensors scratch ggml context (%d bytes)", stScratchArenaSize)
	}

	// 10. Open all shard files
	shardFiles, err := openShardFiles(stDir, index.Shards)
	if err != nil {
		scratchCtx.Free()
		return nil, fmt.Errorf("opening shard files: %w", err)
	}

	return &stModelReader{
		index:       index,
		stmap:       stmap,
		paramValues: paramValues,
		tensorSpecs: tensorSpecs,
		shardFiles:  shardFiles,
		ggufToHF:    ggufToHF,
		archDef:     archDef,
		scratchCtx:  scratchCtx,
	}, nil
}

// loadConfigJSON reads and unmarshals a config.json file.
//
// Uses json.Decoder.UseNumber() so that numeric values are preserved as
// json.Number (textual form) instead of being eagerly decoded as float64.
// This matters because Go's default JSON decoding collapses integer and float
// literals into the same float64 type, erasing the int-vs-float distinction
// that the downstream param resolver depends on. With json.Number, the
// toUint32 / toFloat32 helpers can inspect the textual form and reject
// fractional values when an integer is expected (e.g. rms_norm_eps = 1e-06
// must not silently truncate to uint32(0)).
func loadConfigJSON(path string) (map[string]any, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	var cfg map[string]any
	if err := dec.Decode(&cfg); err != nil {
		return nil, fmt.Errorf("parsing config.json: %w", err)
	}
	return cfg, nil
}

// extractHFClass reads config.json["architectures"][0].
func extractHFClass(cfg map[string]any) (string, error) {
	archAny, ok := cfg["architectures"]
	if !ok {
		return "", fmt.Errorf("config.json: no 'architectures' field")
	}
	archArr, ok := archAny.([]any)
	if !ok || len(archArr) == 0 {
		return "", fmt.Errorf("config.json: 'architectures' is not a non-empty array")
	}
	hfClass, ok := archArr[0].(string)
	if !ok {
		return "", fmt.Errorf("config.json: architectures[0] is not a string")
	}
	return hfClass, nil
}

// buildParamValues constructs a GGUF-key → value map.
// The stmap says HF_PARAM → GGUF_KEY; we read the HF param value from
// config.json and store it under the GGUF key name.
//
// Supports dotted key paths (e.g. "text_config.num_hidden_layers") for
// models like Qwen3.5 that nest text model parameters under a sub-object.
func buildParamValues(archDef *ArchDef, stmap *ArchSTMap, configJSON map[string]any) map[string]any {
	result := make(map[string]any)
	for hfParam, ggufKey := range stmap.Params {
		if val, ok := resolveNested(configJSON, hfParam); ok {
			result[ggufKey] = val
		}
	}
	result["general.architecture"] = archDef.Architecture.Name
	return result
}

// resolveNested traverses configJSON using a dotted key path.
// e.g. resolveNested(cfg, "text_config.hidden_size") → cfg["text_config"]["hidden_size"].
func resolveNested(m map[string]any, key string) (any, bool) {
	parts := strings.SplitN(key, ".", 2)
	val, ok := m[parts[0]]
	if !ok {
		return nil, false
	}
	if len(parts) == 1 {
		return val, true
	}
	sub, ok := val.(map[string]any)
	if !ok {
		return nil, false
	}
	return resolveNested(sub, parts[1])
}

// openShardFiles opens all shard files for ReadAt.
func openShardFiles(stDir string, shards []string) (map[string]*os.File, error) {
	files := make(map[string]*os.File, len(shards))
	for _, shard := range shards {
		path := filepath.Join(stDir, shard)
		f, err := os.Open(path)
		if err != nil {
			// Close already-opened files on failure.
			for _, ff := range files {
				ff.Close()
			}
			return nil, fmt.Errorf("opening shard %s: %w", shard, err)
		}
		files[shard] = f
	}
	return files, nil
}

// buildSTTensorSpecs precomputes TensorSpec for all tensors in the index.
//
// Key detail: safetensors (PyTorch) stores shapes in row-major (C) order:
// [out, in] for weight matrices. GGML stores dimensions as ne[0]=inner,
// ne[1]=outer (column-major / Fortran). The data layout is identical — ne0
// is always the contiguous dimension. So we reverse the shape array.
func buildSTTensorSpecs(index *SafetensorsIndex) map[string]TensorSpec {
	specs := make(map[string]TensorSpec, len(index.Tensors))
	for name, entry := range index.Tensors {
		ggmlType, known := stDtypeToGGML[entry.Dtype]
		if !known {
			continue
		}

		var ne [4]int64
		for i := range 4 {
			ne[i] = 1
		}
		rank := len(entry.Shape)
		for i := 0; i < rank && i < 4; i++ {
			// Reverse: HF shape[out, in] → GGML ne[in, out].
			ne[i] = entry.Shape[rank-1-i]
		}

		// Safetensors tensors can have rank > 2 (e.g. conv1d as [8192, 1, 4]).
		// GGML expects weight matrices to be 2D [ne0, ne1]. Fold trailing dims
		// into ne1 — the raw data is identical (row-major), we just declare a
		// different 2D view so ggml_is_matrix() passes.
		for dim := 2; dim < 4; dim++ {
			ne[1] *= ne[dim]
			ne[dim] = 1
		}

		// Rank-1 weight tensors (norm weights, biases, scale vectors) are used as
		// operands in element-wise binary ops. The Metal op_bin kernel requires
		// both operands to be F32 (see ggml-metal-ops.cpp:ggml_metal_op_bin).
		// Matches GGUF convention: llama.cpp's HF→GGUF converter writes all 1D
		// weight tensors as F32.
		if rank == 1 && ggmlType == ggml.TypeF16 {
			ggmlType = ggml.TypeF32
		}
		// ssm_conv1d weights must be F32 for ggml's SSM conv kernel (no F16 kernel exists).
		if strings.Contains(name, "conv1d") && ggmlType == ggml.TypeF16 {
			ggmlType = ggml.TypeF32
		}

		size := tensorSizeForType(ggmlType, ne)
		if size == 0 {
			continue
		}

		specs[name] = TensorSpec{
			Type: ggmlType,
			Ne:   ne,
			Size: size,
		}
	}
	return specs
}

// ---------------------------------------------------------------------------
// GGUF → HF name resolution
// ---------------------------------------------------------------------------

// buildGGUFToHFMap constructs a translation map from GGUF tensor names to
// HF-format tensor names, using the stmap file and the safetensors index.
//
// Strategy:
//  1. Global tensors: stmap.GlobalTensors maps our_short_name → hf_full_name
//     and GGUF global names are just the short names.
//  2. Per-layer tensors: iterate every HF tensor in the index, try to match
//     it against the HF prefix pattern. When matched, construct the GGUF name
//     by applying the same layer index to the GGUF prefix.
func buildGGUFToHFMap(stmap *ArchSTMap, index *SafetensorsIndex) (map[string]string, error) {
	ggufToHF := make(map[string]string)

	// 1. Global tensors: our_short_name → hf_full_name (direct)
	for ourShort, hfFull := range stmap.GlobalTensors {
		ggufToHF[ourShort] = hfFull
	}

	// 2. Build reverse lookup: HF_short_name → our_short_name for per-layer tensors.
	hfShortToOur := make(map[string]string, len(stmap.Tensors))
	for ourShort, hfShort := range stmap.Tensors {
		hfShortToOur[hfShort] = ourShort
	}

	// 3. Split HF prefix on {N} into before/after parts.
	//    e.g. "model.layers.{N}." → before="model.layers.", after="."
	before, after, ok := strings.Cut(stmap.LayerPrefixHF, "{N}")
	if !ok {
		return nil, fmt.Errorf("HF layer prefix %q must contain {N} placeholder", stmap.LayerPrefixHF)
	}

	// 4. Match each HF tensor against the prefix pattern.
	for hfName := range index.Tensors {
		layerIdx, suffix, matched := matchHFName(hfName, before, after)
		if !matched {
			continue
		}
		ourShort, ok := hfShortToOur[suffix]
		if !ok {
			// Tensor exists in HF but isn't mapped in the stmap — skip.
			continue
		}
		// Construct GGUF name: replace @{layer_idx} in GGUF prefix, append our_short_name.
		ggufName := strings.ReplaceAll(stmap.LayerPrefixGGUF, "@{layer_idx}", strconv.Itoa(layerIdx)) + ourShort
		ggufToHF[ggufName] = hfName
	}

	return ggufToHF, nil
}

// matchHFName checks if hfName matches the pattern before + digits + after + suffix.
// On success, returns (layerIndex, suffixAfterPrefix, true).
func matchHFName(hfName, before, after string) (int, string, bool) {
	if !strings.HasPrefix(hfName, before) {
		return 0, "", false
	}
	rest := hfName[len(before):]
	if after != "" {
		idx := strings.Index(rest, after)
		if idx == 0 {
			return 0, "", false
		}
		digits := rest[:idx]
		suffix := rest[idx+len(after):]
		n, err := strconv.Atoi(digits)
		if err != nil || n < 0 {
			return 0, "", false
		}
		return n, suffix, true
	}
	// No "after" part — rest starts with digits, followed by optional suffix.
	i := 0
	for i < len(rest) && rest[i] >= '0' && rest[i] <= '9' {
		i++
	}
	if i == 0 {
		return 0, "", false
	}
	digits := rest[:i]
	suffix := rest[i:]
	n, err := strconv.Atoi(digits)
	if err != nil || n < 0 {
		return 0, "", false
	}
	return n, suffix, true
}

// ---------------------------------------------------------------------------
// GGUFReader interface implementations
// ---------------------------------------------------------------------------

func (r *stModelReader) GetU32(key string) (uint32, bool) {
	val, ok := r.paramValues[key]
	if !ok {
		return 0, false
	}
	return toUint32(val)
}

func (r *stModelReader) GetF32(key string) (float32, bool) {
	val, ok := r.paramValues[key]
	if !ok {
		return 0, false
	}
	return toFloat32(val)
}

func (r *stModelReader) GetArrInts(key string) ([]int, bool) {
	// Check GGUF metadata from the stmap.
	if val, ok := r.paramValues[key]; ok {
		if arr, ok := toIntArr(val); ok {
			return arr, true
		}
	}
	return nil, false
}

func (r *stModelReader) GetArrBools(key string) ([]bool, bool) {
	// Check GGUF metadata from the stmap.
	if val, ok := r.paramValues[key]; ok {
		if arr, ok := toBoolArr(val); ok {
			return arr, true
		}
	}
	return nil, false
}

func (r *stModelReader) GetTensorDim(ggufName string, dim int) (int64, bool) {
	hfName, ok := r.ggufToHF[ggufName+".weight"]
	if !ok {
		hfName, ok = r.ggufToHF[ggufName]
	}
	if !ok {
		return 0, false
	}
	spec, ok := r.tensorSpecs[hfName]
	if !ok {
		return 0, false
	}
	if dim < 0 || dim >= 4 {
		return 0, false
	}
	return spec.Ne[dim], true
}

// ---------------------------------------------------------------------------
// Tensor enumeration
// ---------------------------------------------------------------------------

func (r *stModelReader) TensorCount() int {
	return len(r.ggufToHF)
}

func (r *stModelReader) TensorNames() []string {
	names := make([]string, 0, len(r.ggufToHF))
	for n := range r.ggufToHF {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

func (r *stModelReader) TensorSpec(ggufName string) (TensorSpec, bool) {
	hfName, ok := r.ggufToHF[ggufName]
	if !ok {
		return TensorSpec{}, false
	}
	spec, ok := r.tensorSpecs[hfName]
	return spec, ok
}

// ---------------------------------------------------------------------------
// Tensor data loading
// ---------------------------------------------------------------------------

func (r *stModelReader) ReadTensor(ggufName string, buf []byte) error {
	hfName, ok := r.ggufToHF[ggufName]
	if !ok {
		return fmt.Errorf("GGUF tensor name %q not mapped to HF (not in stmap or missing from index)", ggufName)
	}

	entry, ok := r.index.Tensors[hfName]
	if !ok {
		return fmt.Errorf("tensor %q not found in safetensors index", hfName)
	}

	spec, ok := r.tensorSpecs[hfName]
	if !ok {
		return fmt.Errorf("tensor %q spec not available", hfName)
	}

	if len(buf) < spec.Size {
		return fmt.Errorf("buffer too small for tensor %q: need %d, have %d", hfName, spec.Size, len(buf))
	}

	f, ok := r.shardFiles[entry.Shard]
	if !ok {
		return fmt.Errorf("shard file %q not open", entry.Shard)
	}

	if entry.DataSize == 0 {
		return nil
	}

	// Fast path: direct bytewise copy when the source dtype maps 1:1 to the
	// target ggml type AND no numeric transform applies. BF16 never qualifies
	// (raw bytes aren't valid F16 — it must be converted through F32).
	needsTransform := tensorMatchesAnyTransform(r.stmap.Transforms, ggufName)
	if !needsTransform && entry.Dtype != "BF16" {
		if srcGGML, known := stDtypeToGGML[entry.Dtype]; known && srcGGML == spec.Type {
			if _, err := f.ReadAt(buf[:spec.Size], entry.DataOffset); err != nil {
				return fmt.Errorf("reading tensor %q (HF: %q) from shard %s: %w", ggufName, hfName, entry.Shard, err)
			}
			return nil
		}
	}

	// Slow path: read raw bytes → run a ggml graph that casts to F32, applies
	// element-wise transforms, casts to the destination type, and writes
	// directly into buf. The whole pipeline runs on ggml-aligned arena memory
	// so there is no possibility of unaligned ggml ops reading from a Go slice
	// pointer (the alignment hazard of the previous TensorSetData approach).
	//
	// When a structural transform (reorder_v_heads) applies, we split into
	// two passes: cast→F32 (with optional pre-reorder element-wise ops) into a
	// host F32 buffer, reorder in Go, then F32→dst (with optional post-reorder
	// element-wise ops). The structural transform cannot be chunked along the
	// flat element axis the way element-wise ops can.
	realSrcType, ok := stDtypeToGGMLRaw(entry.Dtype)
	if !ok {
		return fmt.Errorf("unsupported source dtype %q for tensor %q", entry.Dtype, ggufName)
	}

	raw := make([]byte, entry.DataSize)
	if _, err := f.ReadAt(raw, entry.DataOffset); err != nil {
		return fmt.Errorf("reading raw bytes for tensor %q (HF: %q): %w", ggufName, hfName, err)
	}

	numEls := neElements(spec.Ne)
	batch := splitTensorTransforms(r.stmap.Transforms, ggufName)

	if !batch.hasStructural() {
		// Single-pass element-wise pipeline. batch.post is empty by construction
		// when no structural transform exists, so we only need batch.pre.
		ops, err := buildElemwiseOps(batch.pre)
		if err != nil {
			return fmt.Errorf("tensor %q: %w", ggufName, err)
		}
		if err := r.convert(numEls, realSrcType, raw, spec.Type, buf[:spec.Size], ops); err != nil {
			return fmt.Errorf("tensor %q convert: %w", ggufName, err)
		}
		return nil
	}

	// Two-pass pipeline (raw → F32 → reorder → dst). The host F32 buffer is
	// aliased as bytes for transit through the ggml convert helper; the
	// reorder pass operates on the F32 view of the same memory.
	f32Buf := make([]float32, numEls)
	f32Bytes := unsafe.Slice((*byte)(unsafe.Pointer(&f32Buf[0])), int(numEls)*4)

	preOps, err := buildElemwiseOps(batch.pre)
	if err != nil {
		return fmt.Errorf("tensor %q pre ops: %w", ggufName, err)
	}
	if err := r.convert(numEls, realSrcType, raw, ggml.TypeF32, f32Bytes, preOps); err != nil {
		return fmt.Errorf("tensor %q convert (pre-reorder): %w", ggufName, err)
	}

	if err := applyStructuralTransforms(batch.structural, ggufName, f32Buf, spec); err != nil {
		return err
	}

	postOps, err := buildElemwiseOps(batch.post)
	if err != nil {
		return fmt.Errorf("tensor %q post ops: %w", ggufName, err)
	}
	if err := r.convert(numEls, ggml.TypeF32, f32Bytes, spec.Type, buf[:spec.Size], postOps); err != nil {
		return fmt.Errorf("tensor %q convert (post-reorder): %w", ggufName, err)
	}
	return nil
}

// convert runs the slow-path conversion pipeline in chunks against the
// reader's scratch arena: srcBytes (encoded as srcType) → optional Cast→F32 →
// element-wise op chain → optional Cast→dstType → dstBytes. Chunking is along
// the flat element dimension and is correct only for element-wise pipelines;
// structural transforms must be applied separately by the caller against a
// full-tensor F32 buffer between two convert calls.
//
// All tensor data lives in the scratch arena, which is allocated with
// AllocPermAllow so that ggml owns and aligns the storage. Source bytes are
// memcpy'd directly into the tensor's arena-backed data pointer (and the
// result memcpy'd back out the same way) — never aliased with a Go pointer
// via TensorSetData, which is the alignment hazard we are eliminating. We
// deliberately do not use ggml_backend_tensor_set/get here: those require a
// backend buffer association which arena-only tensors do not have.
func (r *stModelReader) convert(
	numEls int64,
	srcType ggml.GGMLType, srcBytes []byte,
	dstType ggml.GGMLType, dstBytes []byte,
	ops []stElemwiseOp,
) error {
	if r.scratchCtx == nil {
		return fmt.Errorf("convert: scratch context not initialized")
	}
	srcElem := stElementSize(srcType)
	dstElem := stElementSize(dstType)
	if srcElem == 0 {
		return fmt.Errorf("convert: unsupported srcType=%d", srcType)
	}
	if dstElem == 0 {
		return fmt.Errorf("convert: unsupported dstType=%d", dstType)
	}
	if int64(len(srcBytes)) < numEls*int64(srcElem) {
		return fmt.Errorf("convert: src buffer too small (%d < %d)", len(srcBytes), numEls*int64(srcElem))
	}
	if int64(len(dstBytes)) < numEls*int64(dstElem) {
		return fmt.Errorf("convert: dst buffer too small (%d < %d)", len(dstBytes), numEls*int64(dstElem))
	}

	// Trivial pipeline: no type change, no ops. Just memcpy. The fast path in
	// ReadTensor should normally catch this case, but tolerate it here too.
	if srcType == dstType && len(ops) == 0 {
		copy(dstBytes[:numEls*int64(dstElem)], srcBytes[:numEls*int64(srcElem)])
		return nil
	}

	for chunkStart := int64(0); chunkStart < numEls; chunkStart += stMaxConvertChunkElems {
		chunkEls := int64(stMaxConvertChunkElems)
		if chunkStart+chunkEls > numEls {
			chunkEls = numEls - chunkStart
		}

		r.scratchCtx.Rewind()

		input := ggml.NewTensor1D(r.scratchCtx, srcType, chunkEls)
		if input.IsNil() {
			return fmt.Errorf("convert: NewTensor1D(srcType=%d, n=%d) returned nil — scratch arena exhausted?", srcType, chunkEls)
		}
		// Memcpy bytes directly into the tensor's arena-allocated, aligned data
		// region. We deliberately do not use ggml_backend_tensor_set here: that
		// API requires a backend buffer association which arena-only tensors do
		// not have. Direct memcpy is correct because the destination (arena
		// memory) is aligned; the source (Go []byte) does not need to be.
		srcOff := chunkStart * int64(srcElem)
		srcLen := chunkEls * int64(srcElem)
		inputData := ggml.TensorData(input)
		if inputData == nil {
			return fmt.Errorf("convert: input tensor has nil data — context not in AllocPermAllow mode?")
		}
		copy(unsafe.Slice((*byte)(inputData), int(srcLen)), srcBytes[srcOff:srcOff+srcLen])

		// Stage 1: cast into F32 working type (skip if already F32).
		var f32node ggml.Tensor
		if srcType == ggml.TypeF32 {
			f32node = input
		} else {
			f32node = ggml.Cast(r.scratchCtx, input, ggml.TypeF32)
			if f32node.IsNil() {
				return fmt.Errorf("convert: Cast srcType=%d → F32 returned nil", srcType)
			}
		}

		// Stage 2: chain element-wise ops in declared order.
		node := f32node
		for _, op := range ops {
			node = op(r.scratchCtx, node)
			if node.IsNil() {
				return fmt.Errorf("convert: element-wise op returned nil tensor")
			}
		}

		// Stage 3: cast to destination type (skip if already correct).
		var final ggml.Tensor
		if dstType == ggml.TypeF32 {
			final = node
		} else {
			final = ggml.Cast(r.scratchCtx, node, dstType)
			if final.IsNil() {
				return fmt.Errorf("convert: Cast → dstType=%d returned nil", dstType)
			}
		}

		gf := ggml.NewGraph(r.scratchCtx, 16)
		if gf == nil {
			return fmt.Errorf("convert: NewGraph returned nil")
		}
		gf.BuildForwardExpand(final)
		ggml.GraphCompute(r.scratchCtx, gf, 1)

		// Memcpy result bytes out of the aligned tensor arena directly into the
		// caller's destination slice. Same justification as the input write:
		// arena memory is aligned, the destination (Go []byte) need not be.
		dstOff := chunkStart * int64(dstElem)
		dstLen := chunkEls * int64(dstElem)
		finalData := ggml.TensorData(final)
		if finalData == nil {
			return fmt.Errorf("convert: final tensor has nil data pointer")
		}
		copy(dstBytes[dstOff:dstOff+dstLen], unsafe.Slice((*byte)(finalData), int(dstLen)))
	}
	return nil
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

func (r *stModelReader) Close() error {
	if r.scratchCtx != nil {
		r.scratchCtx.Free()
		r.scratchCtx = nil
	}
	var firstErr error
	for _, f := range r.shardFiles {
		if err := f.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// ---------------------------------------------------------------------------
// Memory requirements estimation
// ---------------------------------------------------------------------------

func (r *stModelReader) MinMemoryRequired(maxSeqLen int) MemReq {
	// --- Weight VRAM ---
	var weightVRAM uint64
	for _, spec := range r.tensorSpecs {
		weightVRAM += uint64(spec.Size)
	}

	// --- Cache VRAM ---
	cacheVRAM := r.estimateCacheVRAM(maxSeqLen)

	// --- Non-GPU RAM ---
	// Conservatively estimate safetensors index, shard file handles, config.json, stmap data
	nonGPU := uint64(128 * 1024 * 1024)

	return MemReq{WeightVRAM: weightVRAM, CacheVRAM: cacheVRAM, OverheadRAM: nonGPU}
}

func (r *stModelReader) estimateCacheVRAM(maxSeqLen int) uint64 {
	arch := r.extractString("general.architecture")
	nLayers := int(r.extractUint64(arch + ".block_count"))
	nKVHeads := int(r.extractUint64(arch + ".attention.head_count_kv"))
	headDimK := int(r.extractUint64(arch + ".attention.key_length"))
	headDimV := int(r.extractUint64(arch + ".attention.value_length"))

	if nLayers == 0 || nKVHeads == 0 || headDimK == 0 || headDimV == 0 {
		total := uint64(0)
		for _, spec := range r.tensorSpecs {
			total += uint64(spec.Size)
		}
		if total == 0 {
			return 0
		}
		return total / 5
	}

	perTokenPerLayer := 2 * (headDimK + headDimV) * nKVHeads * 4
	cacheVRAM := uint64(maxSeqLen * nLayers * perTokenPerLayer)

	_, hasSSM := r.archDef.Blocks["recurrent_ssm"]

	// SSM state: models using SSM have per-layer conv state — estimate as similar footprint to KV.
	// if arch == "qwen35" || arch == "llada" || arch == "llada-moe" {
	if hasSSM {
		cacheVRAM *= 2
	}

	cacheVRAM = cacheVRAM * 6 / 5
	return cacheVRAM
}

func (r *stModelReader) extractString(key string) string {
	val, ok := r.paramValues[key]
	if !ok {
		return ""
	}
	s, ok := val.(string)
	if !ok {
		return ""
	}
	return s
}

func (r *stModelReader) extractUint64(key string) uint64 {
	val, ok := r.paramValues[key]
	if !ok {
		return 0
	}
	switch v := val.(type) {
	case json.Number:
		if i, err := v.Int64(); err == nil && i >= 0 {
			return uint64(i)
		}
		if f, err := v.Float64(); err == nil && f >= 0 {
			return uint64(f)
		}
		return 0
	case uint64:
		return v
	case uint32:
		return uint64(v)
	case uint:
		return uint64(v)
	case int64:
		return uint64(v)
	case int:
		return uint64(v)
	case float64:
		return uint64(v)
	case float32:
		return uint64(v)
	default:
		return 0
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// toIntArr converts a JSON array value to []int.
func toIntArr(val any) ([]int, bool) {
	arr, ok := val.([]any)
	if !ok {
		return nil, false
	}
	result := make([]int, len(arr))
	for i, v := range arr {
		n, ok := toFloat64(v)
		if !ok {
			return nil, false
		}
		result[i] = int(n)
	}
	return result, true
}

// toBoolArr converts a JSON array value to []bool.
func toBoolArr(val any) ([]bool, bool) {
	arr, ok := val.([]any)
	if !ok {
		return nil, false
	}
	result := make([]bool, len(arr))
	for i, v := range arr {
		b, ok := v.(bool)
		if !ok {
			// Also accept 0/1 integers.
			n, ok := toFloat64(v)
			if !ok {
				return nil, false
			}
			result[i] = n != 0
		} else {
			result[i] = b
		}
	}
	return result, true
}

func toFloat64(v any) (float64, bool) {
	switch n := v.(type) {
	case json.Number:
		f, err := n.Float64()
		if err != nil {
			return 0, false
		}
		return f, true
	case float64:
		return n, true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	default:
		return 0, false
	}
}

// toUint32 converts a config.json numeric value to uint32.
//
// Strict: rejects non-integer values and anything outside [0, MaxUint32].
// This is essential — the previous version blindly truncated float64 to
// uint32, causing catastrophic silent failures when a semantically-float
// param (e.g. rms_norm_eps = 1e-06) was probed as uint32 during param
// resolution: it would "succeed" with value 0, prevent the subsequent GetF32
// probe from running, and the downstream consumer would read zero where the
// true float value belonged.
//
// With json.Number inputs (from UseNumber()-enabled JSON decoder), we prefer
// integer parsing and fall back to a whole-number float check. Values with
// fractional parts or exponents that don't evaluate to a non-negative whole
// number in range are rejected — which is exactly what resolveParam needs to
// move on to GetF32.
func toUint32(val any) (uint32, bool) {
	switch v := val.(type) {
	case json.Number:
		if i, err := v.Int64(); err == nil {
			if i < 0 || i > math.MaxUint32 {
				return 0, false
			}
			return uint32(i), true
		}
		// Integer parse failed — the literal has a decimal point or exponent.
		// Accept only if the float value is exactly a non-negative whole
		// number within uint32 range (e.g. "1e7" == 10000000.0 is fine; but
		// "1e-06" fails because Trunc(1e-06) != 1e-06).
		f, err := v.Float64()
		if err != nil {
			return 0, false
		}
		return floatToUint32(f)
	case float64:
		return floatToUint32(v)
	case float32:
		return floatToUint32(float64(v))
	case uint64:
		if v > math.MaxUint32 {
			return 0, false
		}
		return uint32(v), true
	case int64:
		if v < 0 || v > math.MaxUint32 {
			return 0, false
		}
		return uint32(v), true
	case int:
		if v < 0 || int64(v) > math.MaxUint32 {
			return 0, false
		}
		return uint32(v), true
	case uint:
		if uint64(v) > math.MaxUint32 {
			return 0, false
		}
		return uint32(v), true
	default:
		return 0, false
	}
}

// floatToUint32 accepts a float only if it is a non-negative whole number
// representable as uint32.
func floatToUint32(f float64) (uint32, bool) {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return 0, false
	}
	if f < 0 || f > math.MaxUint32 {
		return 0, false
	}
	if f != math.Trunc(f) {
		return 0, false
	}
	return uint32(f), true
}

// toFloat32 converts a config.json numeric value to float32.
func toFloat32(val any) (float32, bool) {
	switch v := val.(type) {
	case json.Number:
		f, err := v.Float64()
		if err != nil {
			return 0, false
		}
		return float32(f), true
	case float64:
		return float32(v), true
	case float32:
		return v, true
	case int64:
		return float32(v), true
	case int:
		return float32(v), true
	case uint64:
		return float32(v), true
	case uint:
		return float32(v), true
	default:
		return 0, false
	}
}
