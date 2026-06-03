package arch

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// findRepoRoot walks up from the package's cwd until it finds a directory
// containing models/arch (the canonical repo-root marker for this project).
// Returns "" if not found. Used by integration tests that need to read real
// model files committed under repo-relative paths.
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

// TestVisionStmapLoadsGemma4E4B confirms that loading a real Gemma 4
// multimodal safetensors directory through NewModelReaderSafetensors
// surfaces the vision-tower and projector tensors alongside the decoder,
// and that the corresponding `vision.*` params resolve from
// `config.json.vision_config.*`.
//
// This is the integration smoke for the stmap vision surface described in
// ARCHITECTURE.md "Vision / Multimodal Subsystem → Construction Across Two
// Formats": stmap parsing + two-pattern matching in buildGGUFToHFMap +
// merged-namespace paramValues.
//
// Skipped when models/gemma-4-E4B-it.st is not present locally (e.g. CI
// without weights downloaded).
func TestVisionStmapLoadsGemma4E4B(t *testing.T) {
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

	names := reader.TensorNames()

	// ---- Vision tower per-layer tensors should appear under v.blk.<N>.* ----
	wantVisionPerLayer := []string{
		"v.blk.0.attn_q.weight",
		"v.blk.0.attn_k.weight",
		"v.blk.0.attn_v.weight",
		"v.blk.0.attn_out.weight", // upstream GGUF mmproj convention; arch.toml's vision section binds to "attn_out.weight"
		"v.blk.0.attn_q_norm.weight",
		"v.blk.0.attn_k_norm.weight",
		"v.blk.0.ln1.weight",
		"v.blk.0.ln2.weight",
		"v.blk.0.attn_post_norm.weight",
		"v.blk.0.ffn_post_norm.weight",
		"v.blk.0.ffn_gate.weight",
		"v.blk.0.ffn_up.weight",
		"v.blk.0.ffn_down.weight",
	}
	for _, want := range wantVisionPerLayer {
		if !contains(names, want) {
			t.Errorf("vision tensor %q missing from TensorNames()", want)
		}
	}

	// ---- Vision globals ----
	for _, want := range []string{"v.patch_embd.weight", "v.position_embd.weight"} {
		if !contains(names, want) {
			t.Errorf("vision global %q missing", want)
		}
	}

	// ---- Projector global (cross-tower bridge) ----
	if !contains(names, "mm.input_projection.weight") {
		t.Errorf("projector global mm.input_projection.weight missing")
	}

	// ---- Vision layer count tensors: confirm we routed every layer's
	// attn_q.weight, not just layer 0. Count must match
	// vision.block_count (28 for E4B per config.json, but read it from
	// the reader to keep the test resilient to weight-file updates).
	nLayers, ok := reader.GetU32("vision.block_count")
	if !ok || nLayers == 0 {
		t.Fatalf("vision.block_count not resolved (got %d, ok=%v)", nLayers, ok)
	}
	var attnQCount int
	for _, n := range names {
		if strings.HasPrefix(n, "v.blk.") && strings.HasSuffix(n, ".attn_q.weight") {
			attnQCount++
		}
	}
	if attnQCount != int(nLayers) {
		t.Errorf("v.blk.*.attn_q.weight count = %d, want %d (vision.block_count)", attnQCount, nLayers)
	}

	// ---- Multimodality capability flag (derived from config.json) ----
	if v, ok := reader.GetU32("vision.has_encoder"); !ok || v != 1 {
		t.Errorf("vision.has_encoder = (%d, ok=%v), want (1, true) — derived_metadata `config_key_present` op should set this from config.json.vision_config", v, ok)
	}

	// ---- Vision params resolved from vision_config.* ----
	if dim, ok := reader.GetU32("vision.embedding_length"); !ok || dim == 0 {
		t.Errorf("vision.embedding_length not resolved (got %d, ok=%v)", dim, ok)
	}
	if hc, ok := reader.GetU32("vision.attention.head_count"); !ok || hc == 0 {
		t.Errorf("vision.attention.head_count not resolved (got %d, ok=%v)", hc, ok)
	}
	if ff, ok := reader.GetU32("vision.feed_forward_length"); !ok || ff == 0 {
		t.Errorf("vision.feed_forward_length not resolved (got %d, ok=%v)", ff, ok)
	}
	if ps, ok := reader.GetU32("vision.patch_size"); !ok || ps == 0 {
		t.Errorf("vision.patch_size not resolved (got %d, ok=%v)", ps, ok)
	}

	// ---- Decoder tensors still resolve as before (no regression). ----
	if !contains(names, "blk.0.attn_q.weight") {
		t.Errorf("decoder tensor blk.0.attn_q.weight missing — decoder routing broken")
	}
	if !contains(names, "token_embd.weight") {
		t.Errorf("decoder global token_embd.weight missing")
	}
}

// contains is a tiny order-independent membership helper for []string.
func contains(haystack []string, needle string) bool {
	for _, s := range haystack {
		if s == needle {
			return true
		}
	}
	return false
}
