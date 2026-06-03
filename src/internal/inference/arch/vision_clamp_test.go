package arch

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	ggml "inference-lab-bench/internal/ggml"
)

// TestVisionClampsReachableThroughSafetensorsReader proves that Gemma 4's
// Gemma4ClippableLinear clamp scalars are now reachable through the
// safetensors ModelReader. This is the previously-broken path: clamp scalars
// are undeclared in the stmap short-name tables and were silently skipped by
// buildGGUFToHFMap, so the old direct-file loader had to bypass the reader.
// The clamp-sibling discovery pass now registers them under canonical names.
//
// Reader-level (no ggml/GPU init): asserts that TensorSpec returns ok for a
// canonical vision clamp name, that the unified loader builds a non-empty
// per-layer clamp map with at least one Active() clamp, and that all values
// are finite.
//
// Skipped when models/gemma-4-E4B-it.st is not present locally.
func TestVisionClampsReachableThroughSafetensorsReader(t *testing.T) {
	root := findRepoRoot(t)
	if root == "" {
		t.Skip("repo root with models/arch not found relative to package dir")
	}
	stDir := filepath.Join(root, "models", "gemma-4-E4B-it.st")
	archDir := filepath.Join(root, "models", "arch")
	if _, err := os.Stat(filepath.Join(stDir, "config.json")); err != nil {
		t.Skipf("gemma-4-E4B-it.st not available locally: %v", err)
	}

	archDef, err := Load(archDir, "gemma4")
	if err != nil {
		t.Fatalf("Load(gemma4): %v", err)
	}

	reader, err := NewModelReaderSafetensors(archDef, stDir, archDir, MmprojEnabled)
	if err != nil {
		t.Fatalf("NewModelReaderSafetensors: %v", err)
	}
	defer reader.Close()

	// ---- Part 1: clamp scalars are now reachable under canonical names. ----
	// The vision tower's q_proj clamp maps to "v.blk.0.attn_q.input_max".
	// Before the fix this returned (_, false) — the scalar was unmapped.
	const canonical = "v.blk.0.attn_q.input_max"
	spec, ok := reader.TensorSpec(canonical)
	if !ok {
		t.Fatalf("TensorSpec(%q) not reachable — clamp-sibling discovery pass missing or broken", canonical)
	}
	// Safetensors rank-0 BF16 scalars are surfaced as F16 by buildSTTensorSpecs.
	if spec.Type != ggml.TypeF16 && spec.Type != ggml.TypeF32 {
		t.Errorf("clamp scalar %q has unexpected type %d (want F16 or F32)", canonical, spec.Type)
	}
	v, ok := readClampScalar(reader, canonical)
	if !ok {
		t.Errorf("readClampScalar(%q) failed", canonical)
	}
	if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
		t.Errorf("clamp scalar %q = %v, want finite", canonical, v)
	}

	// ---- Part 2: the unified loader builds usable per-layer clamps. ----
	vr, err := ResolveVisionWeights(archDef, &ResolvedParams{
		Ints:   map[string]int{ParamNLayers: layerCountFromCanonical(reader)},
		Floats: map[string]float32{},
	})
	if err != nil {
		t.Fatalf("ResolveVisionWeights: %v", err)
	}
	if vr == nil {
		t.Fatal("ResolveVisionWeights returned nil for a vision arch")
	}

	clamps, err := LoadVisionClampsFromReader(reader, archDef, vr)
	if err != nil {
		t.Fatalf("LoadVisionClampsFromReader: %v", err)
	}
	if clamps == nil {
		t.Fatal("LoadVisionClampsFromReader returned nil — no clamps found through the reader")
	}

	nonEmptyLayers, activeCount := 0, 0
	for il, lm := range clamps.Layer {
		if len(lm) == 0 {
			continue
		}
		nonEmptyLayers++
		for key, c := range lm {
			for _, f := range []float32{c.InMin, c.InMax, c.OutMin, c.OutMax} {
				if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
					t.Errorf("layer %d %q clamp has non-finite bound %v", il, key, f)
				}
			}
			if c.Active() {
				activeCount++
			}
		}
	}
	if nonEmptyLayers == 0 {
		t.Fatal("no layer has any clamp — reachability fix did not surface vision clamps")
	}
	if activeCount == 0 {
		t.Error("no clamp is Active() — every loaded clamp is a no-op, which is unexpected for Gemma 4 vision")
	}
}

// layerCountFromCanonical reads vision.block_count from the reader, with a
// conservative fallback so ResolveVisionWeights produces enough layer slots to
// cover the clamps present in the file.
func layerCountFromCanonical(reader ModelReader) int {
	if n, ok := reader.GetU32("vision.block_count"); ok && n > 0 {
		return int(n)
	}
	// Fallback: count distinct v.blk.<N> indices among clamp names.
	maxIdx := -1
	for _, name := range reader.TensorNames() {
		if !strings.HasPrefix(name, "v.blk.") || !strings.HasSuffix(name, clampInputMaxSuffix) {
			continue
		}
		rest := strings.TrimPrefix(name, "v.blk.")
		dot := strings.IndexByte(rest, '.')
		if dot <= 0 {
			continue
		}
		var idx int
		for _, c := range rest[:dot] {
			if c < '0' || c > '9' {
				idx = -1
				break
			}
			idx = idx*10 + int(c-'0')
		}
		if idx > maxIdx {
			maxIdx = idx
		}
	}
	return maxIdx + 1
}

// TestF16ToF32Subnormals verifies the F16→F32 conversion against known values,
// including the subnormal range the prior implementation zeroed out.
func TestF16ToF32Subnormals(t *testing.T) {
	cases := []struct {
		name string
		bits uint16
		want float32
	}{
		{"one", 0x3C00, 1.0},
		{"two", 0x4000, 2.0},
		{"neg_one", 0xBC00, -1.0},
		{"zero", 0x0000, 0.0},
		{"half", 0x3800, 0.5},
		{"smallest_subnormal", 0x0001, float32(math.Ldexp(1, -24))},
		{"largest_subnormal", 0x03FF, float32(math.Ldexp(1023, -24))},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := f16ToF32(tc.bits)
			if got != tc.want {
				t.Errorf("f16ToF32(0x%04X) = %v, want %v", tc.bits, got, tc.want)
			}
		})
	}
}
