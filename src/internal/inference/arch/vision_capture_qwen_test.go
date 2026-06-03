package arch

import (
	"fmt"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"

	ggufparser "github.com/gpustack/gguf-parser-go"

	"inference-lab-bench/internal/ggml"
)

// TestVisionCapture_Qwen35_PreSized runs the Qwen3.5-9B GGUF vision tower
// (mmproj-Qwen3.5-9B.gguf) on the committed 960×624 PNG fixture with
// CaptureNamed enabled, then dumps each encoder-stage tensor to
// bin/vision_capture/bench_qwen/ and logs per-stage min/max/mean for the
// checkpoint-diff workflow.
//
// Mirror of TestVisionCapture_Gemma4E2B_PreSized but on the GGUF loader.
// Skipped when the model files are absent.
func TestVisionCapture_Qwen35_PreSized(t *testing.T) {
	root := findRepoRoot(t)
	if root == "" {
		t.Skip("repo root not found relative to package dir")
	}
	modelPath := filepath.Join(root, "models", "Qwen3.5-9B.gguf")
	mmprojPath := filepath.Join(root, "models", "mmproj-Qwen3.5-9B.gguf")
	archDir := filepath.Join(root, "models", "arch")
	imgPath := filepath.Join(root, "test_data", "vision_test_960x624.png")
	for _, p := range []string{modelPath, mmprojPath, imgPath} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("qwen vision capture skipped: %s missing (err: %v)", p, err)
		}
	}

	gf, err := ggufparser.ParseGGUFFile(modelPath)
	if err != nil {
		t.Fatalf("ParseGGUFFile: %v", err)
	}
	archName := gf.Architecture().Architecture
	archDef, err := Load(archDir, archName)
	if err != nil {
		t.Fatalf("Load(%s): %v", archName, err)
	}

	gpu := ggml.GPUInit()
	if gpu == nil {
		t.Fatal("GPUInit nil")
	}
	defer gpu.Free()
	cpu := ggml.CPUInit()
	defer cpu.Free()
	memStats := ggml.DevMemory(gpu, cpu)

	model, err := NewGenericModelFromGGUF(memStats, 4096, archDef, modelPath, archDir, gf, mmprojPath)
	if err != nil {
		t.Fatalf("NewGenericModelFromGGUF: %v", err)
	}
	defer model.Close()

	if model.VisionParams == nil || model.VisionTensors == nil {
		t.Fatalf("vision state not populated: params=%v tensors=%v", model.VisionParams, model.VisionTensors)
	}
	vp := model.VisionParams
	t.Logf("vision params: n_layers=%d n_embd=%d n_heads=%d head_dim=%d n_ff=%d patch=%d rms_eps=%g norm=%s proj=%s",
		vp.NLayers, vp.NEmbd, vp.NHeads, vp.HeadDim, vp.NFF, vp.PatchSize, vp.RMSEps, vp.NormType, vp.ProjectorType)

	cfg, err := PreprocConfigFromArchDef(model.Def)
	if err != nil {
		t.Fatalf("PreprocConfigFromArchDef: %v", err)
	}
	_ = imgPath
	_ = png.Decode

	// Synthetic 64x64 gray image, every pixel = 0.5 — matches
	// llama-mtmd-debug `--image gray -n 64` exactly (do_normalize mean/std
	// = 0.5 maps 0.5 -> 0.0, and bench's ScaleBias(2,-1) does the same), so
	// the captured per-stage sums are directly comparable to the reference
	// dump for op-for-op localization. A uniform image makes pixel layout
	// (planar vs interleaved) irrelevant.
	const synthSide = 64
	patch := cfg.PatchSize
	nMerge := cfg.NMerge
	nPX := synthSide / patch
	nPY := synthSide / patch
	nP := nPX * nPY
	pp := &PreprocessedImage{
		Pixels:    make([]float32, 3*synthSide*synthSide),
		Width:     synthSide,
		Height:    synthSide,
		NPatchesX: nPX,
		NPatchesY: nPY,
		PosX:      make([]int32, nP),
		PosY:      make([]int32, nP),
	}
	// llama-mtmd-debug `--image gray` feeds raw 0.5 straight into the encoder
	// (it bypasses mean/std normalization). bench's graph applies ScaleBias(2,-1),
	// so to make the encoder INPUT (inp_raw_scaled) match the reference's 0.5 we
	// feed 0.75 here: 0.75*2-1 = 0.5. This isolates the comparison to the encoder
	// graph rather than a normalization difference.
	for i := range pp.Pixels {
		pp.Pixels[i] = 0.75
	}
	for p := 0; p < nP; p++ {
		pp.PosX[p] = int32(p % nPX)
		pp.PosY[p] = int32(p / nPX)
	}
	t.Logf("synthetic gray %dx%d (%dx%d patches, %d soft tokens)",
		synthSide, synthSide, nPX, nPY, pp.NTokens(nMerge))

	nImageTokens := pp.NTokens(cfg.NMerge)
	const nLeadingText = 4
	tokenIDs := make([]int32, nLeadingText+nImageTokens)
	for i := 0; i < nLeadingText; i++ {
		tokenIDs[i] = 1
	}
	for i := nLeadingText; i < len(tokenIDs); i++ {
		tokenIDs[i] = 2
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

	// Per-stage stats for the anomaly scan. Include first/last layer plus a
	// couple of interior layers so a mid-stack blowup is visible.
	stageNames := []string{
		"vision.inp_raw_scaled",
		"vision.patch_embd",
		"vision.inp_pos_embd",
	}
	for il := 0; il < vp.NLayers; il++ {
		stageNames = append(stageNames, fmt.Sprintf("vision.layer_%d", il))
	}
	stageNames = append(stageNames,
		"vision.post_ln",
		"vision.merger",
		"vision.projected",
		"decoder.inp_embd_post_splice",
	)
	for _, name := range stageNames {
		data, ok := caps.NamedTensors[name]
		if !ok {
			t.Errorf("missing capture %q", name)
			continue
		}
		st := captureStats(name, "", data)
		first := firstN(data, 6)
		t.Logf("%-32s n=%-7d sum=%-13.4f min=%-11.5g max=%-11.5g NaN=%d Inf=%d first=%v",
			name, st.NElements, st.Mean*float64(st.NFinite), st.Min, st.Max, st.NNaN, st.NInf, first)
	}

	outDir := filepath.Join(root, "bin", "vision_capture", "bench_qwen")
	if err := DumpNamedTensors(caps, outDir); err != nil {
		t.Fatalf("DumpNamedTensors: %v", err)
	}
	t.Logf("dumped %d captures to %s", len(caps.NamedTensors), outDir)
	_ = math.Inf
}
