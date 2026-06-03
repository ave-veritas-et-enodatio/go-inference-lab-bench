package inference

import (
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"inference-lab-bench/internal/ggml"
	"inference-lab-bench/internal/model"
)

// findRepoRoot walks up from cwd until it finds a directory containing
// models/arch. Mirrors the helper of the same name in the arch package
// (separate package means we can't share the symbol).
func findRepoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for i := 0; i < 8; i++ {
		if _, err := os.Stat(filepath.Join(dir, "models", "arch")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
	return ""
}

// TestEngineVisionPrefill_Gemma4E2B is the Phase 7 end-to-end engine
// smoke: load gemma-4-E2B-it.st, attach a real JPEG, call Generate, and
// verify the model produces tokens without error. The output isn't
// asserted for semantic correctness — numerical equivalence vs
// llama-mtmd is Phase 8. This test catches integration-level wiring
// regressions (apiserver shape → ChatMessage Parts → tokenizer →
// placeholder expansion → vision graph splice → decoder prefill →
// sampling loop).
//
// Skipped when gemma-4-E2B-it.st or the test image is absent.
func TestEngineVisionPrefill_Gemma4E2B(t *testing.T) {
	root := findRepoRoot(t)
	if root == "" {
		t.Skip("repo root not found relative to package dir — run inside the repo tree")
	}
	stDir := filepath.Join(root, "models", "gemma-4-E2B-it.st")
	archDir := filepath.Join(root, "models", "arch")
	imgPath := filepath.Join(root, "test_data", "vision_test.png")
	if _, err := os.Stat(filepath.Join(stDir, "config.json")); err != nil {
		t.Skipf("engine vision test skipped: %s missing. Drop a Gemma 4 multimodal "+
			"safetensors directory (gemma-4-E2B-it.st) at that path to enable — "+
			"see tools/hf_to_gguf.sh for the HF → bench conversion. (stat err: %v)",
			stDir, err)
	}
	if _, err := os.Stat(imgPath); err != nil {
		t.Skipf("engine vision test skipped: %s missing — this file should be "+
			"committed under test_data/. (stat err: %v)", imgPath, err)
	}

	meta, err := model.ParseSafetensorsDir(stDir, archDir)
	if err != nil {
		t.Fatalf("ParseSafetensorsDir: %v", err)
	}
	info := &model.ModelInfo{
		ID:            "gemma-4-E2B-it",
		Path:          stDir,
		Format:        model.FormatSafetensors,
		Metadata:      meta,
		MmprojEnabled: true,
	}

	// Generous memory budget so the loader's pre-check passes without
	// us querying the live backend.
	memStats := ggml.MemoryStats{
		VRAM: ggml.MemoryStat{Total: 1 << 40, Allocated: 0},
		RAM:  ggml.MemoryStat{Total: 1 << 40, Allocated: 0},
	}
	eng, err := NewEngine(memStats, info, archDir, "", 4096, false)
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	defer eng.Close()

	// Decode the committed test JPEG.
	f, err := os.Open(imgPath)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatalf("png.Decode: %v", err)
	}

	// Build a multi-part chat turn: text + image, mirroring what the
	// apiserver would synthesize from an OpenAI-multimodal request.
	messages := []ChatMessage{
		{
			Role: "user",
			Parts: []ChatContentPart{
				{Type: "text", Text: "Describe this image briefly."},
				{Type: "image"},
			},
		},
	}

	params := DefaultParams()
	params.MaxTokens = 48 // long enough to expose semantic image awareness past the boilerplate opening
	params.Temperature = 0
	params.Images = []ChatImage{{Image: img}}
	// ThinkingEnabled mirrors the apiserver default; either value is
	// fine here, but disabling keeps the prompt short and the
	// completion-token budget meaningful.
	params.ThinkingEnabled = false

	var collected strings.Builder
	metrics, err := eng.Generate(messages, params, func(tok string) bool {
		collected.WriteString(tok)
		return true
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if metrics == nil {
		t.Fatal("nil metrics")
	}
	t.Logf("vision engine smoke: prompt=%d completion=%d output=%q",
		metrics.PromptTokens, metrics.CompletionTokens, collected.String())

	// The model should ingest the expanded vision-soft-token positions.
	// PromptTokens reflects the expanded length, so it must exceed the
	// raw text token count by ~260 (the per-image soft-token count for
	// our 1200×796 test JPEG preprocessed to 960×624 → 60×39 → 20×13).
	const minVisionTokens = 200
	if metrics.PromptTokens < minVisionTokens {
		t.Errorf("prompt tokens = %d, expected ≥ %d after vision expansion",
			metrics.PromptTokens, minVisionTokens)
	}
	if metrics.CompletionTokens == 0 {
		t.Errorf("completion tokens = 0; generation produced nothing")
	}
}
