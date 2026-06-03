package arch

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

// Derived metadata and derived tensor handlers compute load-time values
// (param values and synthesized tensors, respectively) for safetensors-loaded
// models. They are the runtime side of the [[derived_metadata]] and
// [[derived_tensors]] stmap DSL blocks declared in stmap.go.
//
// Each op is registered by name in the corresponding registry below and
// invoked once at NewModelReaderSafetensors time per matching stmap entry.

// derivedMetadataHandler computes a GGUF metadata value at load time from
// config.json fields. The returned value is stored in the safetensors
// reader's paramValues map under spec.Target, where it is subsequently read
// by GetU32 / GetF32 / GetArrInts / GetArrBools just like any other param.
type derivedMetadataHandler func(spec DerivedMetadataSpec, configJSON map[string]any) (any, error)

// derivedTensorHandler synthesizes a tensor at load time. The returned
// []float32 is the flat row-major data (the bench's safetensors reader
// stores synthesized tensors as F32 regardless of intended downstream type;
// the loader's slow-path conversion machinery already handles F32→F16 if a
// consumer wants F16, but rank-1 weights are always F32 by the rank-1 rule
// in buildSTTensorSpecs). The returned ne[4] is the tensor shape.
type derivedTensorHandler func(spec DerivedTensorSpec, configJSON map[string]any) ([]float32, [4]int64, error)

// derivedMetadataOps is the registry of named handlers for the
// [[derived_metadata]] stmap block. Adding a new op = adding an entry here.
var derivedMetadataOps = map[string]derivedMetadataHandler{
	"string_array_eq": handleStringArrayEq,
	"copy_param":      handleCopyParam,
	"config_key_present": handleConfigKeyPresent,
}

// derivedTensorOps is the registry of named handlers for the
// [[derived_tensors]] stmap block.
var derivedTensorOps = map[string]derivedTensorHandler{
	"rope_freqs_proportional": handleRopeFreqsProportional,
}

// resolveDerivedMetadataOp looks up a derived_metadata op handler by name.
func resolveDerivedMetadataOp(op string) (derivedMetadataHandler, error) {
	h, ok := derivedMetadataOps[op]
	if !ok {
		return nil, fmt.Errorf("unknown derived_metadata op %q", op)
	}
	return h, nil
}

// resolveDerivedTensorOp looks up a derived_tensors op handler by name.
func resolveDerivedTensorOp(op string) (derivedTensorHandler, error) {
	h, ok := derivedTensorOps[op]
	if !ok {
		return nil, fmt.Errorf("unknown derived_tensors op %q", op)
	}
	return h, nil
}

// ---------------------------------------------------------------------------
// Handlers — derived_metadata
// ---------------------------------------------------------------------------

// handleStringArrayEq translates a string-array config.json field into a
// []bool by comparing each element against a constant match value. Used by
// Gemma 4 to translate text_config.layer_types (["sliding_attention",
// "full_attention", ...]) into the boolean sliding_window_pattern array that
// the GGUF metadata convention expects.
//
// Required params:
//
//	source = "<config.json dotted key>"
//	match  = "<value to compare each element to>"
func handleStringArrayEq(spec DerivedMetadataSpec, cfg map[string]any) (any, error) {
	source, err := paramString(spec.Params, "source")
	if err != nil {
		return nil, fmt.Errorf("op %q target %q: %w", spec.Op, spec.Target, err)
	}
	match, err := paramString(spec.Params, "match")
	if err != nil {
		return nil, fmt.Errorf("op %q target %q: %w", spec.Op, spec.Target, err)
	}
	val, ok := resolveNested(cfg, source)
	if !ok {
		return nil, fmt.Errorf("op %q target %q: source key %q not found in config.json", spec.Op, spec.Target, source)
	}
	arr, ok := val.([]any)
	if !ok {
		return nil, fmt.Errorf("op %q target %q: source key %q value is %T, want array", spec.Op, spec.Target, source, val)
	}
	out := make([]any, len(arr))
	for i, v := range arr {
		s, ok := v.(string)
		if !ok {
			return nil, fmt.Errorf("op %q target %q: source[%d] is %T, want string", spec.Op, spec.Target, i, v)
		}
		out[i] = s == match
	}
	return out, nil
}

// handleConfigKeyPresent sets the target metadata key to uint32(1) when
// `source` resolves to a non-nil, non-empty value in config.json, and
// uint32(0) otherwise. Used as the canonical signal for "the model file
// declares this capability" — most notably for multimodal detection,
// where `vision_config` presence in the HF config is the upstream
// convention (mirroring the mmproj GGUF's `clip.has_vision_encoder`
// metadata flag).
//
// "Empty" means: nil, "", an empty []any or empty map[string]any. A
// scalar value like `false` or `0` still counts as present — the op is
// a presence test, not a truthiness test. If a future use case needs
// truthiness instead, add a separate op rather than re-overloading this
// one.
//
// Required params:
//
//	source = "<config.json dotted key>"
func handleConfigKeyPresent(spec DerivedMetadataSpec, cfg map[string]any) (any, error) {
	source, err := paramString(spec.Params, "source")
	if err != nil {
		return nil, fmt.Errorf("op %q target %q: %w", spec.Op, spec.Target, err)
	}
	val, ok := resolveNested(cfg, source)
	if !ok || val == nil {
		return uint32(0), nil
	}
	switch v := val.(type) {
	case string:
		if v == "" {
			return uint32(0), nil
		}
	case []any:
		if len(v) == 0 {
			return uint32(0), nil
		}
	case map[string]any:
		if len(v) == 0 {
			return uint32(0), nil
		}
	}
	return uint32(1), nil
}

// handleCopyParam reads one config.json field and writes its value verbatim
// under a different GGUF metadata key. Used when the converter writes the
// same source value under multiple GGUF keys (e.g. Gemma 4 sets both
// gemma4.attention.key_length and gemma4.rope.dimension_count from
// text_config.global_head_dim). The stmap [params] map is a 1:1 HF→GGUF
// mapping, so the second target requires this derived op.
//
// Required params:
//
//	source = "<config.json dotted key>"
func handleCopyParam(spec DerivedMetadataSpec, cfg map[string]any) (any, error) {
	source, err := paramString(spec.Params, "source")
	if err != nil {
		return nil, fmt.Errorf("op %q target %q: %w", spec.Op, spec.Target, err)
	}
	val, ok := resolveNested(cfg, source)
	if !ok {
		return nil, fmt.Errorf("op %q target %q: source key %q not found in config.json", spec.Op, spec.Target, source)
	}
	return val, nil
}

// ---------------------------------------------------------------------------
// Handlers — derived_tensors
// ---------------------------------------------------------------------------

// handleRopeFreqsProportional synthesizes the rope_freqs.weight tensor for
// Gemma 4's full-attention layers (and any architecture using the same
// proportional-RoPE trick on top of ggml's full-rotary NeoX primitive).
//
// The "proportional" RoPE rotates only the first partial_rotary_factor of
// each head's dimensions; the rest are unrotated. ggml's NeoX RoPE op
// rotates all dims, but the freq_factors tensor provides per-pair theta
// dividers — setting freq_factor = 1e30 for unrotated pairs makes their
// effective theta ≈ 0, suppressing their rotation. See llama.cpp
// Gemma4Model.generate_extra_tensors for the reference implementation.
//
// Output: F32 tensor of length head_dim/2 with values
//
//	[1.0] * n_rot + [1e30] * n_unrot
//
// where:
//
//	n_rot   = int(head_dim * partial_rotary_factor / 2)
//	n_unrot = int(head_dim / 2) - n_rot
//
// Required params:
//
//	head_dim_source       = "<config.json dotted key>"  // e.g. text_config.global_head_dim
//	partial_rotary_source = "<config.json dotted key>"  // e.g. text_config.rope_parameters.full_attention.partial_rotary_factor
func handleRopeFreqsProportional(spec DerivedTensorSpec, cfg map[string]any) ([]float32, [4]int64, error) {
	headDimKey, err := paramString(spec.Params, "head_dim_source")
	if err != nil {
		return nil, [4]int64{}, fmt.Errorf("op %q target %q: %w", spec.Op, spec.Target, err)
	}
	partialKey, err := paramString(spec.Params, "partial_rotary_source")
	if err != nil {
		return nil, [4]int64{}, fmt.Errorf("op %q target %q: %w", spec.Op, spec.Target, err)
	}
	headDimAny, ok := resolveNested(cfg, headDimKey)
	if !ok {
		return nil, [4]int64{}, fmt.Errorf("op %q target %q: head_dim_source %q not found in config.json", spec.Op, spec.Target, headDimKey)
	}
	partialAny, ok := resolveNested(cfg, partialKey)
	if !ok {
		return nil, [4]int64{}, fmt.Errorf("op %q target %q: partial_rotary_source %q not found in config.json", spec.Op, spec.Target, partialKey)
	}
	headDim, ok := toUint32(headDimAny)
	if !ok {
		return nil, [4]int64{}, fmt.Errorf("op %q target %q: head_dim value %v (%T) is not a valid integer", spec.Op, spec.Target, headDimAny, headDimAny)
	}
	partial, ok := toFloat32(partialAny)
	if !ok {
		return nil, [4]int64{}, fmt.Errorf("op %q target %q: partial_rotary value %v (%T) is not a valid float", spec.Op, spec.Target, partialAny, partialAny)
	}
	if headDim%2 != 0 {
		return nil, [4]int64{}, fmt.Errorf("op %q target %q: head_dim=%d is not even", spec.Op, spec.Target, headDim)
	}
	nRot := int(float64(headDim) * float64(partial) / 2)
	half := int(headDim / 2)
	nUnrot := half - nRot
	if nRot < 0 || nUnrot < 0 {
		return nil, [4]int64{}, fmt.Errorf("op %q target %q: invalid n_rot=%d n_unrot=%d (head_dim=%d partial=%v)", spec.Op, spec.Target, nRot, nUnrot, headDim, partial)
	}
	out := make([]float32, half)
	for i := range nRot {
		out[i] = 1.0
	}
	for i := nRot; i < half; i++ {
		out[i] = 1e30
	}
	return out, [4]int64{int64(half), 1, 1, 1}, nil
}

// handleConv3DTemporalSplit synthesizes one Conv2D kernel by slicing a single
// HF Conv3D weight along its temporal axis. Used by Qwen3-VL's vision patch
// embedder: HF stores a single `patch_embed.proj.weight` Conv3D of numpy shape
// [oc, ic, kt, kh, kw] (temporal_patch_size kt = 2); the bench's vision tower
// (and the llama.cpp mmproj GGUF) expect two summed Conv2D kernels, one per
// temporal index. This mirrors llama.cpp conversion/qwen3vl.py
// Qwen3VLVisionModel.modify_tensors:
//
//	yield ("v.patch_embd.weight",   data_torch[:, :, 0, ...])
//	yield ("v.patch_embd.weight.1", data_torch[:, :, 1, ...])
//
// The produced tensor's data is the row-major slice [oc, ic, kh, kw]; its ggml
// ne is the reversed shape [kw, kh, ic, oc] (matching buildSTTensorSpecs'
// row-major→column-major reversal and the mmproj GGUF's stored ne).
//
// This op exists because the slice is a one-source→two-output operation with a
// rank reduction (5D→4D), which the in-place same-size [[transforms]]
// mechanism cannot express. It reads the source tensor directly from its
// shard (unlike the config-only derivedTensorOps registry), so it is dispatched
// separately by buildDerivedTensors.
//
// Required params:
//
//	source         = "<HF tensor name>"  // the rank-5 Conv3D weight
//	temporal_index = <int>               // which kt slice (0..kt-1) to extract
func handleConv3DTemporalSplit(spec DerivedTensorSpec, index *SafetensorsIndex, stDir string) ([]float32, [4]int64, error) {
	source, err := paramString(spec.Params, "source")
	if err != nil {
		return nil, [4]int64{}, fmt.Errorf("op %q target %q: %w", spec.Op, spec.Target, err)
	}
	tIdx, err := paramInt(spec.Params, "temporal_index")
	if err != nil {
		return nil, [4]int64{}, fmt.Errorf("op %q target %q: %w", spec.Op, spec.Target, err)
	}

	entry, ok := index.Tensors[source]
	if !ok {
		return nil, [4]int64{}, fmt.Errorf("op %q target %q: source tensor %q not in safetensors index", spec.Op, spec.Target, source)
	}
	if len(entry.Shape) != 5 {
		return nil, [4]int64{}, fmt.Errorf("op %q target %q: source %q must be rank-5 [oc,ic,kt,kh,kw], got shape %v", spec.Op, spec.Target, source, entry.Shape)
	}
	oc, ic, kt, kh, kw := entry.Shape[0], entry.Shape[1], entry.Shape[2], entry.Shape[3], entry.Shape[4]
	if tIdx < 0 || int64(tIdx) >= kt {
		return nil, [4]int64{}, fmt.Errorf("op %q target %q: temporal_index %d out of range [0,%d)", spec.Op, spec.Target, tIdx, kt)
	}

	src, err := readSTTensorF32(index, stDir, source)
	if err != nil {
		return nil, [4]int64{}, fmt.Errorf("op %q target %q: %w", spec.Op, spec.Target, err)
	}
	if int64(len(src)) != oc*ic*kt*kh*kw {
		return nil, [4]int64{}, fmt.Errorf("op %q target %q: source %q decoded %d elems, expected %d", spec.Op, spec.Target, source, len(src), oc*ic*kt*kh*kw)
	}

	// Gather the kt-th temporal slice. Input is row-major over (oc,ic,kt,kh,kw);
	// output is row-major over (oc,ic,kh,kw) for the fixed kt. The (kh,kw) plane
	// is contiguous in both, so copy it plane-by-plane.
	plane := kh * kw
	out := make([]float32, oc*ic*plane)
	for o := int64(0); o < oc; o++ {
		for i := int64(0); i < ic; i++ {
			srcBase := (((o*ic+i)*kt+int64(tIdx))*kh)*kw
			dstBase := (o*ic + i) * plane
			copy(out[dstBase:dstBase+plane], src[srcBase:srcBase+plane])
		}
	}
	// ggml ne = reversed numpy shape [oc, ic, kh, kw].
	return out, [4]int64{kw, kh, ic, oc}, nil
}

// readSTTensorF32 reads a single safetensors tensor and decodes it to a flat
// row-major []float32. Supports F32, F16, and BF16 source dtypes — the dtypes
// HF model weights actually use. Reads directly from the shard file (the
// caller is a load-time derived-tensor builder that runs before the reader's
// shard handles or scratch ggml context exist).
func readSTTensorF32(index *SafetensorsIndex, stDir, name string) ([]float32, error) {
	entry, ok := index.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("tensor %q not in index", name)
	}
	f, err := os.Open(filepath.Join(stDir, entry.Shard))
	if err != nil {
		return nil, fmt.Errorf("opening shard %q: %w", entry.Shard, err)
	}
	defer f.Close()

	raw := make([]byte, entry.DataSize)
	if _, err := f.ReadAt(raw, entry.DataOffset); err != nil {
		return nil, fmt.Errorf("reading %q from shard %q: %w", name, entry.Shard, err)
	}

	switch entry.Dtype {
	case "F32":
		if int64(len(raw))%4 != 0 {
			return nil, fmt.Errorf("tensor %q: F32 byte length %d not a multiple of 4", name, len(raw))
		}
		out := make([]float32, len(raw)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return out, nil
	case "F16":
		if int64(len(raw))%2 != 0 {
			return nil, fmt.Errorf("tensor %q: F16 byte length %d not even", name, len(raw))
		}
		out := make([]float32, len(raw)/2)
		for i := range out {
			out[i] = f16BitsToF32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
		return out, nil
	case "BF16":
		if int64(len(raw))%2 != 0 {
			return nil, fmt.Errorf("tensor %q: BF16 byte length %d not even", name, len(raw))
		}
		out := make([]float32, len(raw)/2)
		for i := range out {
			// BF16 = high 16 bits of the IEEE-754 F32 representation.
			out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("tensor %q: unsupported dtype %q for F32 decode", name, entry.Dtype)
	}
}

// f16BitsToF32 converts an IEEE-754 half-precision bit pattern to float32.
func f16BitsToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	switch {
	case exp == 0:
		if mant == 0 {
			return math.Float32frombits(sign) // ±0
		}
		// Subnormal: normalize.
		e := -1
		for mant&0x400 == 0 {
			mant <<= 1
			e++
		}
		mant &= 0x3ff
		return math.Float32frombits(sign | uint32(127-15-e)<<23 | mant<<13)
	case exp == 0x1f:
		return math.Float32frombits(sign | 0xff<<23 | mant<<13) // Inf/NaN
	default:
		return math.Float32frombits(sign | (exp+(127-15))<<23 | mant<<13)
	}
}
