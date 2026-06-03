package arch

import (
	"fmt"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	ggufparser "github.com/gpustack/gguf-parser-go"

	"inference-lab-bench/internal/ggml"
)

// TestVisionCapture_RealImage_BothTowers runs the bench vision encoder on the
// committed 960×624 PNG for both the Qwen3.5-9B and gemma-4-E4B-it GGUF towers,
// dumping per-stage encoder checkpoints (patch_embd, inp_pos_embd, per-layer,
// post_ln/merger/pooled, projected, post-splice) to bin/vision_capture/<tower>.
//
// This is the REAL-image analogue of the synthetic-gray TestVisionCapture_Qwen35
// harness, for the encoder checkpoint-diff against llama-mtmd-debug's matching
// real-image dumps (MTMD_DEBUG_DUMP). MEASUREMENT ONLY — no inference code is
// touched. Skipped when model files are absent.
func TestVisionCapture_RealImage_BothTowers(t *testing.T) {
	root := findRepoRoot(t)
	if root == "" {
		t.Skip("repo root not found relative to package dir")
	}
	cases := []struct {
		tag      string
		model    string
		mmproj   string
		archName string
	}{
		{"bench_qwen_real", "Qwen3.5-9B.gguf", "mmproj-Qwen3.5-9B.gguf", "qwen35"},
		{"bench_gemma_real", "gemma-4-E4B-it.gguf", "mmproj-gemma-4-E4B-it.gguf", "gemma4"},
	}
	// Two images: vision_test.png exercises the resize path (1200x796 ->
	// 960x624); vision_test_960x624.png is already at target size (no resize),
	// isolating encoder fidelity from preprocessing. Dumps land in
	// bin/vision_capture/<tag>__<image-stem> so a per-image checkpoint-diff
	// against llama's MTMD_DEBUG_GRAPH dump for the matching image is clean.
	images := []string{
		filepath.Join(root, "test_data", "vision_test.png"),
		filepath.Join(root, "test_data", "vision_test_960x624.png"),
	}
	archDir := filepath.Join(root, "models", "arch")

	for _, tc := range cases {
	  for _, imgPath := range images {
		imgStem := strings.TrimSuffix(filepath.Base(imgPath), ".png")
		t.Run(tc.tag+"__"+imgStem, func(t *testing.T) {
			modelPath := filepath.Join(root, "models", tc.model)
			mmprojPath := filepath.Join(root, "models", tc.mmproj)
			for _, p := range []string{modelPath, mmprojPath, imgPath} {
				if _, err := os.Stat(p); err != nil {
					t.Skipf("capture skipped: %s missing (%v)", p, err)
				}
			}

			gf, err := ggufparser.ParseGGUFFile(modelPath)
			if err != nil {
				t.Fatalf("ParseGGUFFile: %v", err)
			}
			archDef, err := Load(archDir, gf.Architecture().Architecture)
			if err != nil {
				t.Fatalf("Load: %v", err)
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
				t.Fatalf("vision state not populated")
			}
			vp := model.VisionParams

			// Decode + preprocess the real image through the production path.
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
			nImageTokens := pp.NTokens(cfg.NMerge)
			t.Logf("[%s] preproc %dx%d -> %dx%d (%dx%d patches, merge=%d, %d soft tokens)",
				tc.tag, src.Bounds().Dx(), src.Bounds().Dy(), pp.Width, pp.Height,
				pp.NPatchesX, pp.NPatchesY, cfg.NMerge, nImageTokens)

			const nLeadingText = 4
			tokenIDs := make([]int32, nLeadingText+nImageTokens)
			for i := 0; i < nLeadingText; i++ {
				tokenIDs[i] = 1
			}
			for i := nLeadingText; i < len(tokenIDs); i++ {
				tokenIDs[i] = 2
			}
			visionInput := VisionSpliceInput{Preprocessed: pp, Start: nLeadingText, Length: nImageTokens}

			caps := &ForwardCaptures{Flags: CaptureNamed}
			if _, err := model.ForwardStateless(tokenIDs, caps, false, []VisionSpliceInput{visionInput}); err != nil {
				t.Fatalf("ForwardStateless: %v", err)
			}

			// Log every captured stage's stats for the per-stage scan.
			names := []string{"vision.inp_raw_scaled", "vision.patch_embd", "vision.inp_pos_embd",
				"vision.ln1_0", "vision.attn_out_0", "vision.ffn_inp_0",
				"vision.ffn_inp_normed_0", "vision.ffn_out_0"}
			for il := 0; il < vp.NLayers; il++ {
				names = append(names, fmt.Sprintf("vision.layer_%d", il))
			}
			names = append(names, "vision.post_ln", "vision.merger", "vision.pooled",
				"vision.projected", "vision.projected_normed",
				"decoder.inp_embd_pre_splice", "decoder.inp_embd_post_splice")
			for _, name := range names {
				data, ok := caps.NamedTensors[name]
				if !ok {
					continue
				}
				st := captureStats(name, "", data)
				t.Logf("%-32s n=%-8d sum=%-14.4f min=%-11.5g max=%-11.5g NaN=%d Inf=%d",
					name, st.NElements, st.Mean*float64(st.NFinite), st.Min, st.Max, st.NNaN, st.NInf)
			}

			outDir := filepath.Join(root, "bin", "vision_capture", tc.tag+"__"+imgStem)
			if err := DumpNamedTensors(caps, outDir); err != nil {
				t.Fatalf("DumpNamedTensors: %v", err)
			}
			t.Logf("[%s/%s] dumped %d captures to %s", tc.tag, imgStem, len(caps.NamedTensors), outDir)
		})
	  }
	}
}
