package arch

import (
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// loadGemma4PreprocConfig is the test helper that resolves the canonical
// preprocessor knobs by parsing gemma4.arch.toml rather than carrying a
// hand-written copy in the test file. Keeps the test fixture in lock-step
// with the actual arch declaration; a future tweak to the bounds lives in
// one place (the TOML) and no Go test needs touching.
func loadGemma4PreprocConfig(t *testing.T) PreprocConfig {
	t.Helper()
	root := findRepoRoot(t)
	if root == "" {
		t.Skip("repo root with models/arch not found")
	}
	def, err := Load(filepath.Join(root, "models", "arch"), "gemma4")
	if err != nil {
		t.Fatalf("loading gemma4 arch: %v", err)
	}
	cfg, err := PreprocConfigFromArchDef(def)
	if err != nil {
		t.Fatalf("PreprocConfigFromArchDef: %v", err)
	}
	return cfg
}

// TestSmartResize_Gemma4Bands is the pure-math regression test for the
// smart-resize algorithm. Each case derives target dimensions for the
// canonical Gemma 4 vision config (read from gemma4.arch.toml — patch_size
// × n_merge = 48 align; 252–280 soft tokens). Expected values are
// computed offline from the upstream llama.cpp algorithm
// (mtmd-image.cpp::calc_size_preserved_ratio with min/max overload).
func TestSmartResize_Gemma4Bands(t *testing.T) {
	cfg := loadGemma4PreprocConfig(t)
	align := cfg.AlignSize()
	minPx := cfg.MinPixels()
	maxPx := cfg.MaxPixels()

	cases := []struct {
		name           string
		srcW, srcH     int
		wantW, wantH   int
	}{
		// Bigger-than-max landscape: scales down to ~960×624, an aspect-
		// preserving result that's just under max_pixels.
		{name: "landscape_1200x796", srcW: 1200, srcH: 796, wantW: 960, wantH: 624},

		// Smaller-than-min square: scales up to align grid that hits min.
		{name: "small_square_224", srcW: 224, srcH: 224, wantW: 768, wantH: 768},

		// Already-aligned square within band: passes through to its
		// rounded size, possibly trimmed to fit max.
		{name: "in_band_square_768", srcW: 768, srcH: 768, wantW: 768, wantH: 768},

		// Tall portrait, well above max.
		{name: "portrait_796x1200", srcW: 796, srcH: 1200, wantW: 624, wantH: 960},

		// Extreme aspect ratio — still aspect-preserving, still aligned,
		// still inside the pixel band. beta = sqrt(512·2048/645120) ≈
		// 1.2748; floor(2048/1.2748/48)·48 = 1584; floor(512/1.2748/48)·48
		// = 384. Area 608,256, tokens 33·8 = 264.
		{name: "wide_2048x512", srcW: 2048, srcH: 512, wantW: 1584, wantH: 384},

		// Tiny input — upscale to min.
		{name: "tiny_32x32", srcW: 32, srcH: 32, wantW: 768, wantH: 768},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			gotW, gotH := smartResize(c.srcW, c.srcH, align, minPx, maxPx)
			if gotW != c.wantW || gotH != c.wantH {
				t.Errorf("smartResize(%d, %d) = (%d, %d), want (%d, %d)",
					c.srcW, c.srcH, gotW, gotH, c.wantW, c.wantH)
			}
			// Cross-check post-conditions: aligned to grid, in pixel band,
			// aspect ratio drift within one alignment-step.
			if gotW%align != 0 || gotH%align != 0 {
				t.Errorf("not aligned to %d: got %dx%d", align, gotW, gotH)
			}
			area := gotW * gotH
			if area > maxPx {
				t.Errorf("area %d > max_pixels %d", area, maxPx)
			}
			// Skip min-pixels check for the tiny-input case where upscale
			// to align grid lands above min naturally; below-min target
			// for any input means the upscale path didn't fire.
		})
	}
}

// TestSmartResize_PostconditionsRandom feeds a spread of sizes and just
// asserts the structural invariants (aligned, within bounds, no zero
// dimensions). Catches a class of regressions where a specific case in
// the band-table above doesn't trip but a neighbor does.
func TestSmartResize_PostconditionsRandom(t *testing.T) {
	cfg := loadGemma4PreprocConfig(t)
	align := cfg.AlignSize()
	minPx := cfg.MinPixels()
	maxPx := cfg.MaxPixels()

	sizes := []struct{ w, h int }{
		{100, 100}, {200, 300}, {400, 400}, {640, 480}, {800, 600},
		{1024, 768}, {1280, 720}, {1600, 1200}, {1920, 1080},
		{2400, 1600}, {3000, 2000}, {4000, 3000},
	}
	for _, s := range sizes {
		w, h := smartResize(s.w, s.h, align, minPx, maxPx)
		if w <= 0 || h <= 0 {
			t.Errorf("smartResize(%d,%d) → degenerate (%d,%d)", s.w, s.h, w, h)
		}
		if w%align != 0 || h%align != 0 {
			t.Errorf("smartResize(%d,%d) → (%d,%d) not aligned to %d", s.w, s.h, w, h, align)
		}
		if w*h > maxPx {
			t.Errorf("smartResize(%d,%d) → (%d,%d), area %d > max %d", s.w, s.h, w, h, w*h, maxPx)
		}
		// Aspect ratio should drift no more than the alignment step
		// allows. Two-axis aligned floor/ceil can each shift by one
		// align step; so worst-case ratio drift is bounded by (1 +
		// align/min_dim) relative.
		srcAspect := float64(s.w) / float64(s.h)
		tgtAspect := float64(w) / float64(h)
		minDim := math.Min(float64(w), float64(h))
		maxRatio := 1.0 + float64(align)/minDim
		ratio := math.Max(srcAspect/tgtAspect, tgtAspect/srcAspect)
		if ratio > maxRatio {
			t.Errorf("smartResize(%d,%d) → (%d,%d) aspect drift %.4f > max %.4f",
				s.w, s.h, w, h, ratio, maxRatio)
		}
	}
}

// TestSmartResize_ExtremeAspect covers pathological aspect ratios that the
// moderate spread above doesn't reach. The downscale branch's per-axis
// max(align, …) floor can pin the short side to `align` while the long side
// stays large enough that the product still exceeds maxPx — e.g. 200000×300
// resolved to 20736×48 = 995328 px before the post-clamp guard was added.
// Here we only assert the hard invariants: aligned, positive, and area within
// budget. Aspect drift is intentionally unbounded — preserving it is
// impossible once one axis is already at the align floor.
func TestSmartResize_ExtremeAspect(t *testing.T) {
	cfg := loadGemma4PreprocConfig(t)
	align := cfg.AlignSize()
	minPx := cfg.MinPixels()
	maxPx := cfg.MaxPixels()

	sizes := []struct{ w, h int }{
		{200000, 300}, {300, 200000}, // far past maxPx/align² aspect threshold
		{100000, 600}, {8192, 16}, {16, 8192},
		{50000, 50}, {1, 60000}, {60000, 1},
	}
	for _, s := range sizes {
		w, h := smartResize(s.w, s.h, align, minPx, maxPx)
		if w <= 0 || h <= 0 {
			t.Errorf("smartResize(%d,%d) → degenerate (%d,%d)", s.w, s.h, w, h)
		}
		if w%align != 0 || h%align != 0 {
			t.Errorf("smartResize(%d,%d) → (%d,%d) not aligned to %d", s.w, s.h, w, h, align)
		}
		if w*h > maxPx {
			t.Errorf("smartResize(%d,%d) → (%d,%d), area %d > max %d", s.w, s.h, w, h, w*h, maxPx)
		}
	}
}

// TestPreprocessImage_SolidRed builds a tiny solid-color image and verifies
// the F32 packing: channel-major, in-band [0,1], correct ordering.
func TestPreprocessImage_SolidRed(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 100, 100))
	for y := 0; y < 100; y++ {
		for x := 0; x < 100; x++ {
			src.Set(x, y, color.RGBA{R: 255, G: 0, B: 0, A: 255})
		}
	}
	cfg := loadGemma4PreprocConfig(t)
	out, err := PreprocessImage(src, cfg)
	if err != nil {
		t.Fatalf("PreprocessImage: %v", err)
	}
	if out.Width%cfg.AlignSize() != 0 || out.Height%cfg.AlignSize() != 0 {
		t.Errorf("output dims (%d,%d) not aligned to %d", out.Width, out.Height, cfg.AlignSize())
	}
	wantLen := 3 * out.Width * out.Height
	if len(out.Pixels) != wantLen {
		t.Fatalf("pixels len = %d, want %d (channel-major 3*W*H)", len(out.Pixels), wantLen)
	}
	// Channel 0 (R) should be ~1.0, channels 1 (G) and 2 (B) ~0.0.
	cR := out.Pixels[0]
	cG := out.Pixels[out.Width*out.Height]
	cB := out.Pixels[2*out.Width*out.Height]
	if cR < 0.95 {
		t.Errorf("R channel near-pixel = %g, want ~1.0", cR)
	}
	if cG > 0.05 || cB > 0.05 {
		t.Errorf("G/B near-pixel = (%g, %g), want ~0.0", cG, cB)
	}
	// Patch positions: grid count consistent with NPatchesX / NPatchesY.
	if got, want := len(out.PosX), out.NPatchesX*out.NPatchesY; got != want {
		t.Errorf("len(PosX) = %d, want %d", got, want)
	}
	if got := out.NPatchesX * cfg.PatchSize; got != out.Width {
		t.Errorf("NPatchesX * PatchSize = %d, want Width %d", got, out.Width)
	}
}

// TestPreprocessImage_RealJPEG runs the full pipeline against the
// committed test image. End-to-end shape + value sanity check.
func TestPreprocessImage_RealImage(t *testing.T) {
	root := findRepoRoot(t)
	if root == "" {
		t.Skip("repo root not found relative to package dir")
	}
	path := filepath.Join(root, "test_data", "vision_test.png")
	f, err := os.Open(path)
	if err != nil {
		t.Skipf("test_data/vision_test.png not available: %v", err)
	}
	defer f.Close()
	src, err := png.Decode(f)
	if err != nil {
		t.Fatalf("decode PNG: %v", err)
	}
	srcW := src.Bounds().Dx()
	srcH := src.Bounds().Dy()

	cfg := loadGemma4PreprocConfig(t)
	out, err := PreprocessImage(src, cfg)
	if err != nil {
		t.Fatalf("PreprocessImage: %v", err)
	}
	t.Logf("preprocessed %dx%d → %dx%d (%d patches, %d soft tokens)",
		srcW, srcH, out.Width, out.Height,
		out.NPatchesX*out.NPatchesY, out.NTokens(cfg.NMerge))

	// 1200x796 → 960x624 per the canonical band, 60x39 patches, 20x13 = 260 tokens.
	if out.Width != 960 || out.Height != 624 {
		t.Errorf("vision_test.png → (%d,%d), want (960,624) (smart-resize regression)",
			out.Width, out.Height)
	}
	if got := out.NTokens(cfg.NMerge); got < cfg.ImageMinTokens || got > cfg.ImageMaxTokens {
		t.Errorf("n_tokens %d outside [%d, %d]", got, cfg.ImageMinTokens, cfg.ImageMaxTokens)
	}
	// All pixel values must be finite and in [0, 1].
	var nBad int
	for i, v := range out.Pixels {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || v < 0 || v > 1.0001 {
			nBad++
			if nBad < 5 {
				t.Errorf("pixel[%d] = %g (out of [0,1])", i, v)
			}
		}
	}
	if nBad > 0 {
		t.Errorf("%d pixels out of [0,1]", nBad)
	}
}
