package arch

import (
	"encoding/binary"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"
	"unsafe"

	"inference-lab-bench/internal/ggml"
)

// visionTestArena sizes the AllocPermAllow arena for the in-test vision
// encoder forward pass. The 16-layer encoder produces ~263 MB kq matrices
// plus softmax/attn-out/residuals per layer; with AllocPermAllow every
// intermediate stays live for the entire pass. Production-path inference
// uses AllocPermDisallow + a backend Sched that releases intermediates
// between ops; this in-test allocator trades memory for simplicity, which
// the M5 Max's 128 GB unified pool absorbs cost-free.
const visionTestArena = 32 << 30 // 32 GiB

// TestVisionEncode_E2E_Gemma4E2B is the single end-to-end vision test:
// load gemma-4-E2B-it.st, preprocess a committed test image, run the
// encoder + projector forward graph, and validate shape + finiteness.
//
// Why one test and not two: there was previously a synthetic-input smoke
// test (TestBuildVisionGraph_Gemma4*) that pre-dated the preprocessor.
// Once the real-image E2E path landed, the smoke test was conceptually
// redundant — same model load, same encoder ops, same projector path —
// so its assertion-only diagnostics (param resolution, per-weight nil
// checks) live here now alongside the forward pass.
//
// Assertions:
//   - Vision params resolve to the values declared in arch.toml + GGUF
//     metadata (n_layers, n_embd, n_heads, head_dim, patch_size).
//   - Every per-layer vision weight handle is non-nil after load.
//   - Encoder runs to completion on real preprocessed data.
//   - Output tensor shape matches the expected post-pool token count
//     given the preprocessed image's patch grid.
//   - Output values are finite.
//
// Skipped when models/gemma-4-E2B-it.st is not available locally.
// E2B shares the exact same vision tower as E4B (16 layers × 768 hidden ×
// 3072 FF) but with a smaller decoder, so it exercises identical encoder
// coverage at lower load cost.
func TestVisionEncode_E2E_Gemma4E2B(t *testing.T) {
	root := findRepoRoot(t)
	if root == "" {
		t.Skip("repo root not found relative to package dir — run inside the repo tree")
	}
	stDir := filepath.Join(root, "models", "gemma-4-E2B-it.st")
	archDir := filepath.Join(root, "models", "arch")
	imgPath := filepath.Join(root, "test_data", "vision_test.png")
	if _, err := os.Stat(filepath.Join(stDir, "config.json")); err != nil {
		t.Skipf("vision E2E test skipped: %s missing. Drop a Gemma 4 multimodal "+
			"safetensors directory (gemma-4-E2B-it.st) at that path to enable — "+
			"see tools/hf_to_gguf.sh for the HF → bench conversion. (stat err: %v)",
			stDir, err)
	}
	if _, err := os.Stat(imgPath); err != nil {
		t.Skipf("vision E2E test skipped: %s missing (this should be a "+
			"committed file; check working tree). (stat err: %v)", imgPath, err)
	}

	// ---- Load model ----
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

	// ---- Load assertions ----
	// Named, fast-failing diagnostics for the common failure modes
	// (missing vision params, missing per-layer weights). Without these,
	// a load regression surfaces as a confusing mid-graph NilTensor cascade
	// instead of an obvious "vision layer N weight X is nil" failure.
	if model.VisionResolved == nil || model.VisionTensors == nil || model.VisionParams == nil {
		t.Fatalf("vision state not populated: resolved=%v tensors=%v params=%v",
			model.VisionResolved, model.VisionTensors, model.VisionParams)
	}
	vp := model.VisionParams
	t.Logf("vision params: n_layers=%d n_embd=%d n_heads=%d head_dim=%d n_ff=%d patch=%d rms_eps=%g",
		vp.NLayers, vp.NEmbd, vp.NHeads, vp.HeadDim, vp.NFF, vp.PatchSize, vp.RMSEps)
	if vp.NLayers != 16 {
		t.Errorf("vision n_layers = %d, want 16", vp.NLayers)
	}
	if vp.NEmbd != 768 || vp.NHeads != 12 || vp.HeadDim != 64 {
		t.Errorf("vision dims = (%d, %d, %d), want (768, 12, 64)", vp.NEmbd, vp.NHeads, vp.HeadDim)
	}
	if vp.PatchSize != 16 {
		t.Errorf("vision patch_size = %d, want 16", vp.PatchSize)
	}
	if model.VisionTensors.Global[WeightVisionPatchEmbd].IsNil() {
		t.Fatal("patch_embd weight is nil")
	}
	if model.VisionTensors.Global[WeightVisionPosEmbd].IsNil() {
		t.Fatal("position_embd weight is nil")
	}
	if model.VisionTensors.Projector[WeightVisionProj].IsNil() {
		t.Fatal("projector proj weight is nil")
	}
	for il := 0; il < vp.NLayers; il++ {
		for _, key := range []string{WeightVisionLN1, WeightVisionLN2, WeightAttnPostNorm, WeightFFNPostNorm,
			WeightAttnQ, WeightAttnK, WeightAttnV, WeightAttnOutput, WeightAttnQNorm, WeightAttnKNorm,
			WeightFFNGate, WeightFFNUp, WeightFFNDown} {
			if model.VisionTensors.Layers[il][key].IsNil() {
				t.Errorf("vision layer %d weight %q is nil", il, key)
			}
		}
	}

	// ---- Preprocess ----
	f, err := os.Open(imgPath)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer f.Close()
	src, err := png.Decode(f)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	// Drive preprocessing config from the model's arch.toml — the
	// single source of truth, no hardcoded Gemma 4 branches in Go.
	cfg, err := PreprocConfigFromArchDef(model.Def)
	if err != nil {
		t.Fatalf("PreprocConfigFromArchDef: %v", err)
	}
	pp, err := PreprocessImage(src, cfg)
	if err != nil {
		t.Fatalf("PreprocessImage: %v", err)
	}
	t.Logf("preprocessed %dx%d → %dx%d (%d patches, %d soft tokens)",
		src.Bounds().Dx(), src.Bounds().Dy(), pp.Width, pp.Height,
		pp.NPatchesX*pp.NPatchesY, pp.NTokens(cfg.NMerge))

	// ---- Allocate input tensors in a graph context ----
	visionCtx := ggml.NewGraphContext(visionTestArena, ggml.AllocPermAllow)
	if visionCtx == nil {
		t.Fatal("NewGraphContext nil")
	}
	defer visionCtx.Free()

	nPatches := pp.NPatchesX * pp.NPatchesY
	inpRaw := ggml.NewTensor4D(visionCtx, ggml.TypeF32, int64(pp.Width), int64(pp.Height), 3, 1)
	posX := ggml.NewTensor1D(visionCtx, ggml.TypeI32, int64(nPatches))
	posY := ggml.NewTensor1D(visionCtx, ggml.TypeI32, int64(nPatches))
	if inpRaw.IsNil() || posX.IsNil() || posY.IsNil() {
		t.Fatal("tensor allocation returned nil")
	}

	// Copy preprocessed F32 pixels straight into the inp_raw tensor —
	// the layouts already match (channel-major, ne[0]=W innermost).
	{
		dst := ggml.TensorData(inpRaw)
		n := len(pp.Pixels)
		dstBytes := unsafe.Slice((*byte)(dst), n*4)
		for i, v := range pp.Pixels {
			binary.LittleEndian.PutUint32(dstBytes[i*4:], math.Float32bits(v))
		}
	}
	// And the per-patch positions.
	{
		dstX := ggml.TensorData(posX)
		dstY := ggml.TensorData(posY)
		bX := unsafe.Slice((*byte)(dstX), nPatches*4)
		bY := unsafe.Slice((*byte)(dstY), nPatches*4)
		for i, v := range pp.PosX {
			binary.LittleEndian.PutUint32(bX[i*4:], uint32(v))
		}
		for i, v := range pp.PosY {
			binary.LittleEndian.PutUint32(bY[i*4:], uint32(v))
		}
	}

	// ---- Build + compute encoder graph ----
	gf := ggml.NewGraph(visionCtx, maxGraphNodes)
	out, err := BuildVisionGraph(visionCtx, gf,
		&VisionInputs{
			InpRaw:    inpRaw,
			PosX:      posX,
			PosY:      posY,
			NPatchesX: pp.NPatchesX,
			NPatchesY: pp.NPatchesY,
		},
		model.VisionParams, model.VisionTensors, model.VisionBuilders, nil)
	if err != nil {
		t.Fatalf("BuildVisionGraph: %v", err)
	}
	if out.IsNil() {
		t.Fatal("BuildVisionGraph returned nil tensor")
	}

	// Expected output: [decoder_n_embd, n_pooled_tokens, 1]. n_pooled =
	// (NPatchesX/n_merge) * (NPatchesY/n_merge); for 1200×796 → 960×624
	// preprocessed → 60×39 patches → 20×13 = 260 pooled tokens.
	decoderNEmbd := int64(model.Params.Ints[ParamNEmbd])
	wantTokens := int64(pp.NTokens(cfg.NMerge))
	if got := out.Ne(0); got != decoderNEmbd {
		t.Errorf("out.Ne(0) = %d, want decoder n_embd %d", got, decoderNEmbd)
	}
	if got := out.Ne(1); got != wantTokens {
		t.Errorf("out.Ne(1) = %d, want pooled tokens %d", got, wantTokens)
	}

	ggml.GraphCompute(visionCtx, gf, 1)

	// Output finiteness check.
	{
		dst := ggml.TensorData(out)
		n := int(out.Ne(0) * out.Ne(1) * out.Ne(2) * out.Ne(3))
		bytes := unsafe.Slice((*byte)(dst), n*4)
		nNaN, nInf := 0, 0
		for i := 0; i < n; i++ {
			v := math.Float32frombits(binary.LittleEndian.Uint32(bytes[i*4:]))
			if math.IsNaN(float64(v)) {
				nNaN++
			}
			if math.IsInf(float64(v), 0) {
				nInf++
			}
		}
		if nNaN > 0 || nInf > 0 {
			t.Errorf("output has %d NaN / %d Inf of %d", nNaN, nInf, n)
		}
		t.Logf("vision output: [%d × %d × %d × %d] = %d elements, all finite",
			out.Ne(0), out.Ne(1), out.Ne(2), out.Ne(3), n)
	}
}
