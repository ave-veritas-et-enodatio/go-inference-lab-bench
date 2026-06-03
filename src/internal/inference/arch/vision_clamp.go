package arch

import (
	"encoding/binary"
	"math"
	"strings"

	ggml "inference-lab-bench/internal/ggml"
)

// Clamp-scalar suffixes for Gemma 4's Gemma4ClippableLinear. Each vision-tower
// linear weight is paired with four sibling scalar tensors named
// `<base>.input_min`, `.input_max`, `.output_min`, `.output_max`. These names
// are identical across formats (the GGUF mmproj converter preserves them), so
// they double as the canonical reader-tensor names registered in
// buildGGUFToHFMap and as the discovery key in LoadVisionClampsFromReader.
const (
	clampInputMinSuffix  = ".input_min"
	clampInputMaxSuffix  = ".input_max"
	clampOutputMinSuffix = ".output_min"
	clampOutputMaxSuffix = ".output_max"
)

// LinearClamp holds the four scalar bounds that Gemma 4's
// `Gemma4ClippableLinear` applies around every linear projection inside
// the vision tower. The layout in the safetensors file is a quartet of
// 0-rank BF16 tensors sibling to the weight:
//
//	<base>.linear.weight
//	<base>.input_min
//	<base>.input_max
//	<base>.output_min
//	<base>.output_max
//
// Without these clamps, vision-tower activations grow unboundedly through
// the 16 layers — captured layer-15 values exceed ±1000 vs. the trained
// range of roughly ±10. Mirrors `clip_graph_gemma4v::build_mm` in
// llama.cpp/tools/mtmd/models/gemma4v.cpp:138-151.
type LinearClamp struct {
	InMin, InMax   float32
	OutMin, OutMax float32
}

// Active reports whether this clamp has any effect. A clamp with min ≥
// max (sentinel) or default-init values is a no-op and the caller may
// skip the clamp op for graph efficiency.
func (c LinearClamp) Active() bool {
	return c.InMax > c.InMin && c.OutMax > c.OutMin
}

// VisionClamps bundles per-layer and projector clamp info, keyed parallel
// to the corresponding tensor maps in VisionTensors. Missing entries (or
// inactive LinearClamp values) signal "no clamp" — the graph code should
// then fall back to bare MulMat.
type VisionClamps struct {
	// Layer[il][logicalName] holds the clamp for one per-layer linear
	// weight (e.g. "attn_q", "ffn_down").
	Layer []map[string]LinearClamp
	// Projector[logicalName] for the projector weight (e.g. "proj").
	Projector map[string]LinearClamp
}

// LoadVisionClampsFromReader extracts the Gemma4ClippableLinear scalars for
// every vision-tower linear weight by reading them through the ModelReader
// boundary. Because both the GGUF and safetensors readers expose clamp
// scalars under canonical names (`v.blk.<N>.attn_q.input_max` etc.), this one
// function serves both formats. Returns (nil, nil) when there are no clamps to
// load (no [vision] block, no scalars present) — not an error.
//
// The scan is one-shot at model load time. Result is attached to the
// VisionTensors struct and consumed by BuildVisionGraph.
func LoadVisionClampsFromReader(reader ModelReader, def *ArchDef, vr *VisionResolved) (*VisionClamps, error) {
	if def == nil || def.Vision == nil || vr == nil || reader == nil {
		return nil, nil
	}

	// First pass: group by base name. A base is clamp-bearing only when all
	// four scalars are present and finite. We anchor on `.input_max` and
	// derive the remaining three from the same base.
	clampByBase := map[string]LinearClamp{}
	for _, name := range reader.TensorNames() {
		if !strings.HasSuffix(name, clampInputMaxSuffix) {
			continue
		}
		base := strings.TrimSuffix(name, clampInputMaxSuffix)
		inMax, ok1 := readClampScalar(reader, base+clampInputMaxSuffix)
		inMin, ok2 := readClampScalar(reader, base+clampInputMinSuffix)
		outMax, ok3 := readClampScalar(reader, base+clampOutputMaxSuffix)
		outMin, ok4 := readClampScalar(reader, base+clampOutputMinSuffix)
		if !(ok1 && ok2 && ok3 && ok4) {
			continue
		}
		if math.IsNaN(float64(inMin)) || math.IsNaN(float64(inMax)) ||
			math.IsNaN(float64(outMin)) || math.IsNaN(float64(outMax)) {
			continue
		}
		clampByBase[base] = LinearClamp{InMin: inMin, InMax: inMax, OutMin: outMin, OutMax: outMax}
	}
	if len(clampByBase) == 0 {
		return nil, nil
	}

	// Second pass: map logical names → clamps via the canonical weight names
	// in vr. The clamp base is the weight name with `.weight` stripped (e.g.
	// "v.blk.0.attn_q.weight" → "v.blk.0.attn_q"), matching the canonical
	// names registered by buildGGUFToHFMap / present in the GGUF mmproj.
	out := &VisionClamps{}
	out.Layer = make([]map[string]LinearClamp, vr.NLayers)
	for il := 0; il < vr.NLayers; il++ {
		layerMap := map[string]LinearClamp{}
		for logicalName, weightName := range vr.Layers[il] {
			base := strings.TrimSuffix(weightName, ".weight")
			if c, ok := clampByBase[base]; ok {
				layerMap[strings.TrimSuffix(logicalName, ".weight")] = c
			}
		}
		out.Layer[il] = layerMap
	}

	if vr.Projector != nil {
		projMap := map[string]LinearClamp{}
		for logicalName, weightName := range vr.Projector {
			base := strings.TrimSuffix(weightName, ".weight")
			if c, ok := clampByBase[base]; ok {
				projMap[strings.TrimSuffix(logicalName, ".weight")] = c
			}
		}
		out.Projector = projMap
	}

	return out, nil
}

// readClampScalar reads a single scalar clamp value through the ModelReader.
// Returns (value, true) on success, (0, false) when the tensor is absent,
// unreadable, or stored in a type we don't decode. Decodes the first element
// according to the reader-reported ggml type: clamp scalars arrive as F32
// (GGUF and any rank-1 safetensors scalar) or F16 (rank-0 BF16 safetensors
// scalars are converted BF16→F16 by buildSTTensorSpecs).
func readClampScalar(reader ModelReader, name string) (float32, bool) {
	spec, ok := reader.TensorSpec(name)
	if !ok {
		return 0, false
	}
	buf := make([]byte, spec.Size)
	if err := reader.ReadTensor(name, buf); err != nil {
		return 0, false
	}
	switch spec.Type {
	case ggml.TypeF32:
		if len(buf) < 4 {
			return 0, false
		}
		return math.Float32frombits(binary.LittleEndian.Uint32(buf[:4])), true
	case ggml.TypeF16:
		if len(buf) < 2 {
			return 0, false
		}
		return f16ToF32(binary.LittleEndian.Uint16(buf[:2])), true
	case ggml.TypeBF16:
		// Defensive: no current reader returns raw BF16 for clamp scalars
		// (safetensors converts to F16), but decode it correctly if one ever does.
		if len(buf) < 2 {
			return 0, false
		}
		return math.Float32frombits(uint32(binary.LittleEndian.Uint16(buf[:2])) << 16), true
	default:
		return 0, false
	}
}

// f16ToF32 converts an IEEE 754 half-precision value to float32, handling
// subnormals correctly (the prior safetensors path approximated subnormals as
// zero — the F[9] bug this refactor fixes).
func f16ToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		// Subnormal: normalize into the F32 exponent range.
		e := -1
		for mant&0x400 == 0 {
			mant <<= 1
			e++
		}
		mant &= 0x3ff
		return math.Float32frombits(sign | uint32(127-15-e)<<23 | mant<<13)
	case 0x1f:
		return math.Float32frombits(sign | 0xff<<23 | mant<<13)
	default:
		return math.Float32frombits(sign | (exp+(127-15))<<23 | mant<<13)
	}
}
