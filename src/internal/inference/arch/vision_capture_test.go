package arch

import (
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"
	"unsafe"

	"inference-lab-bench/internal/ggml"
)

// TestVisionCapture_Gemma4E2B_PreSized exercises the Phase 8 numerical-
// equivalence capture infrastructure against the pre-sized 960×624 PNG
// fixture. The image dimensions are exactly the smart-resize target
// (60×39 patches → 20×13 = 260 pooled tokens), so the preprocessor is
// a no-op — both bench and llama-mtmd should receive identical pixel
// buffers, isolating any downstream divergence to the encoder graph
// itself.
//
// The test:
//   - Loads gemma-4-E2B-it.st and the pre-sized PNG.
//   - Runs ForwardStateless with CaptureNamed enabled.
//   - Asserts every expected vision-encoder checkpoint produced output
//     and that the values are finite.
//   - Dumps the captured tensors to `bin/vision_capture/` for later
//     diff against llama-mtmd's matching cb()-driven dumps.
//
// Skipped when gemma-4-E2B-it.st or the PNG fixture is absent.
func TestVisionCapture_Gemma4E2B_PreSized(t *testing.T) {
	root := findRepoRoot(t)
	if root == "" {
		t.Skip("repo root not found relative to package dir")
	}
	stDir := filepath.Join(root, "models", "gemma-4-E2B-it.st")
	archDir := filepath.Join(root, "models", "arch")
	imgPath := filepath.Join(root, "test_data", "vision_test_960x624.png")
	if _, err := os.Stat(filepath.Join(stDir, "config.json")); err != nil {
		t.Skipf("vision capture test skipped: %s missing — drop a gemma-4-E2B-it.st "+
			"safetensors directory there to enable. (err: %v)", stDir, err)
	}
	if _, err := os.Stat(imgPath); err != nil {
		t.Skipf("vision capture test skipped: %s missing — should be committed under "+
			"test_data/. (err: %v)", imgPath, err)
	}

	archDef, err := Load(archDir, "gemma4")
	if err != nil {
		t.Fatalf("Load(gemma4): %v", err)
	}
	memStats := ggml.MemoryStats{
		VRAM: ggml.MemoryStat{Total: 1 << 40, Allocated: 0},
		RAM:  ggml.MemoryStat{Total: 1 << 40, Allocated: 0},
	}
	model, err := NewGenericModelFromSafetensors(memStats, 4096, archDef, stDir, archDir, MmprojEnabled)
	if err != nil {
		t.Fatalf("NewGenericModelFromSafetensors: %v", err)
	}
	defer model.Close()

	// Decode + preprocess. With the 960×624 fixture, smart-resize is a
	// no-op — Width/Height come out unchanged.
	f, err := os.Open(imgPath)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer f.Close()
	src, err := png.Decode(f)
	if err != nil {
		t.Fatalf("png.Decode: %v", err)
	}
	cfg, err := PreprocConfigFromArchDef(model.Def)
	if err != nil {
		t.Fatalf("PreprocConfigFromArchDef: %v", err)
	}
	pp, err := PreprocessImage(src, cfg)
	if err != nil {
		t.Fatalf("PreprocessImage: %v", err)
	}
	if pp.Width != 960 || pp.Height != 624 {
		t.Fatalf("preprocessor changed dims: expected 960×624, got %d×%d "+
			"(fixture should bypass smart-resize)", pp.Width, pp.Height)
	}

	// Build a minimal token stream with one image placeholder. The
	// capture test focuses on the vision encoder + splice, so we keep
	// the surrounding tokens to a fixed minimum: [BOS, <image>] suffices
	// to exercise the splice path.
	imagePlaceholder, ok := lookupImagePlaceholderID(t, archDef)
	if !ok {
		t.Skip("model has no image placeholder token — not a multimodal model")
	}
	_ = imagePlaceholder

	// Run the engine forward via the same path as production, but with
	// CaptureNamed enabled. Build a minimal prefill: just enough tokens
	// to host one image placeholder run.
	nImageTokens := pp.NTokens(cfg.NMerge)
	const nLeadingText = 4
	tokenIDs := make([]int32, nLeadingText+nImageTokens)
	for i := 0; i < nLeadingText; i++ {
		tokenIDs[i] = 1 // arbitrary non-zero text-token IDs
	}
	for i := nLeadingText; i < len(tokenIDs); i++ {
		tokenIDs[i] = imagePlaceholder
	}

	visionInput := VisionSpliceInput{
		Preprocessed: pp,
		Start:        nLeadingText,
		Length:       nImageTokens,
	}

	caps := &ForwardCaptures{Flags: CaptureNamed}
	_, err = model.ForwardStateless(tokenIDs, caps, false, []VisionSpliceInput{visionInput})
	if err != nil {
		t.Fatalf("ForwardStateless: %v", err)
	}

	// Expected capture points — match BuildVisionGraph's NamedTensor calls
	// + the decoder-side splice captures.
	wantNames := []string{
		"vision.inp_raw_scaled",
		"vision.patch_embd",
		"vision.inp_pos_embd",
		"vision.layer_0",
		"vision.layer_15", // last vision layer (Gemma 4 vision tower has 16)
		"vision.pooled",
		"vision.projected",
		"vision.projected_normed",
		"decoder.inp_embd_pre_splice",
		"decoder.inp_embd_post_splice",
	}
	for _, name := range wantNames {
		data, ok := caps.NamedTensors[name]
		if !ok {
			t.Errorf("missing capture %q", name)
			continue
		}
		if len(data) == 0 {
			t.Errorf("capture %q empty", name)
			continue
		}
		nNaN, nInf := 0, 0
		for _, v := range data {
			f := float64(v)
			if math.IsNaN(f) {
				nNaN++
			}
			if math.IsInf(f, 0) {
				nInf++
			}
		}
		if nNaN > 0 || nInf > 0 {
			t.Errorf("capture %q: %d NaN / %d Inf of %d", name, nNaN, nInf, len(data))
		}
	}

	// Dump captures to disk for later diff against llama-mtmd output.
	outDir := filepath.Join(root, "bin", "vision_capture", "bench")
	if err := DumpNamedTensors(caps, outDir); err != nil {
		t.Fatalf("DumpNamedTensors: %v", err)
	}
	t.Logf("dumped %d captures to %s", len(caps.NamedTensors), outDir)

	// Spot-check a couple of values for sanity.
	t.Logf("vision.projected_normed: first 4 = %v",
		firstN(caps.NamedTensors["vision.projected_normed"], 4))
	t.Logf("decoder.inp_embd_post_splice: first 4 = %v",
		firstN(caps.NamedTensors["decoder.inp_embd_post_splice"], 4))
}

func lookupImagePlaceholderID(t *testing.T, def *ArchDef) (int32, bool) {
	t.Helper()
	if def == nil || def.Vision == nil || def.Vision.ImageToken == "" {
		return -1, false
	}
	// The image-placeholder token-to-ID lookup lives in the tokenizer,
	// which we don't have access to here without loading it. Instead,
	// use a sentinel that the forward path treats consistently — the
	// SetRows splice will overwrite this position's embedding regardless
	// of the source token ID, and after the RewriteTokenIDsForVision
	// rewrite, the position will look up token_embd[0] anyway. So any
	// non-zero, in-vocab placeholder works for this exercise.
	//
	// Use 2 as a safe placeholder — well within the vocab for any model
	// we've shipped, distinct from the padding token (0) and the BOS (1).
	return 2, true
}

func firstN(s []float32, n int) []float32 {
	if len(s) < n {
		return s
	}
	return s[:n]
}

// Suppress an unused-import warning if the unsafe import drops out;
// keep this so future captures-related work can add raw-byte assertions.
var _ = unsafe.Sizeof(0)
