package model

import (
	"os"
	"path/filepath"
	"testing"
)

func TestStripModelDescriptors(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		// User's worked example.
		{"gemma-4-E4B-it-Q4_K_S", "gemma-4-E4B"},
		// Common Gemma 4 variants.
		{"gemma-4-E2B-it", "gemma-4-E2B"},
		{"gemma-4-26B-A4B-it", "gemma-4-26B-A4B"},
		{"gemma-4-26B-A4B-it-MXFP4_MOE", "gemma-4-26B-A4B"},
		// Instruct-style suffix (Llama, Qwen, LLaDA naming).
		{"Llama-3.2-3B-Instruct", "Llama-3.2-3B"},
		{"Llama-3.2-3B-Instruct-Q4_K_M", "Llama-3.2-3B"},
		{"LLaDA-8B-Instruct", "LLaDA-8B"},
		// Case insensitivity on descriptor tokens.
		{"foo-INSTRUCT", "foo"},
		{"foo-IT", "foo"},
		// Q<digit> truncation when no other descriptor present.
		{"llama-3.2-3b-instruct-q4_k_m", "llama-3.2-3b"},
		// Already-clean name passes through unchanged.
		{"gemma-4-E4B", "gemma-4-E4B"},
		{"qwen35-9b", "qwen35-9b"},
		// "it" embedded in a non-descriptor token does NOT truncate
		// (token-boundary discipline). Note: only whole-token "it"
		// matches; substrings don't.
		{"transit-3B", "transit-3B"},
		// Edge case: descriptor at index 0 — preserve whole name to
		// avoid emitting empty match key.
		{"it-foo-bar", "it-foo-bar"},
	}
	for _, c := range cases {
		got := stripModelDescriptors(c.in)
		if got != c.want {
			t.Errorf("stripModelDescriptors(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestMmprojMatchKey(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		// User's worked examples.
		{"qwen36-mmproj.gguf", "qwen36"},
		{"mmproj-gemma-4-E2B.gguf", "gemma-4-E2B"},
		{"gemma-4-E4B-mmproj-f16.gguf", "gemma-4-E4B-f16"},
		// Path components are stripped.
		{"/models/mmproj-gemma-4-E4B-it.gguf", "gemma-4-E4B-it"},
		// No mmproj decoration → returns the cleaned base.
		{"gemma-4-E4B.gguf", "gemma-4-E4B"},
		// Edge: mmproj as bare token.
		{"mmproj.gguf", ""},
	}
	for _, c := range cases {
		got := mmprojMatchKey(c.in)
		if got != c.want {
			t.Errorf("mmprojMatchKey(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestFindMmprojForGGUF exercises the full match pipeline against a
// temp dir of synthetic candidate files. Reproduces the user's spec
// example: gemma-4-E4B-it-Q4_K_S should match gemma-4-E4B-mmproj-f16.gguf
// among unrelated and same-family-but-different-size siblings.
func TestFindMmprojForGGUF(t *testing.T) {
	td := t.TempDir()
	// Use the same `touch` pattern as production GGUFs — we don't need
	// real GGUF content, just the filenames for the matcher.
	touch := func(name string) string {
		p := filepath.Join(td, name)
		if err := os.WriteFile(p, []byte{0}, 0o644); err != nil {
			t.Fatalf("touch %s: %v", name, err)
		}
		return p
	}
	touch("qwen36-mmproj.gguf")
	touch("mmproj-gemma-4-E2B.gguf")
	wantPath := touch("gemma-4-E4B-mmproj-f16.gguf")
	decoderPath := touch("gemma-4-E4B-it-Q4_K_S.gguf")

	m := &Manager{enableMmproj: true}
	got := m.findMmprojForGGUF(decoderPath)
	if got != wantPath {
		t.Errorf("findMmprojForGGUF(%q) = %q, want %q", decoderPath, got, wantPath)
	}

	// Disabled flag → no match even when one would resolve.
	m.enableMmproj = false
	if got := m.findMmprojForGGUF(decoderPath); got != "" {
		t.Errorf("findMmprojForGGUF with enableMmproj=false = %q, want \"\"", got)
	}
}

// TestFindMmproj_PrecisionSuffixPairing confirms that when the same model
// exists at two precisions (e.g. an F16/F32 equivalence matrix), each decoder
// binds the mmproj of its OWN precision rather than the sorted-first sidecar.
// Without the exact-stem first pass, both decoders strip to "gemma-4-E4B-it"
// (the precision token is past "it", which truncates first) and bind the f16
// sidecar.
func TestFindMmproj_PrecisionSuffixPairing(t *testing.T) {
	td := t.TempDir()
	touch := func(name string) string {
		p := filepath.Join(td, name)
		if err := os.WriteFile(p, []byte{0}, 0o644); err != nil {
			t.Fatalf("touch: %v", err)
		}
		return p
	}
	f16mm := touch("mmproj-gemma-4-E4B-it-f16.gguf")
	f32mm := touch("mmproj-gemma-4-E4B-it-f32.gguf")
	f16dec := touch("gemma-4-E4B-it-f16.gguf")
	f32dec := touch("gemma-4-E4B-it-f32.gguf")

	m := &Manager{enableMmproj: true}
	if got := m.findMmprojForGGUF(f16dec); got != f16mm {
		t.Errorf("f16 decoder bound %q, want %q", filepath.Base(got), filepath.Base(f16mm))
	}
	if got := m.findMmprojForGGUF(f32dec); got != f32mm {
		t.Errorf("f32 decoder bound %q, want %q", filepath.Base(got), filepath.Base(f32mm))
	}
}

// TestFindMmproj_NoCrossSizeMatch confirms the hyphen-boundary check —
// a decoder named gemma-4-E2 must NOT spuriously match a sidecar named
// gemma-4-E2B-* (different size class).
func TestFindMmproj_NoCrossSizeMatch(t *testing.T) {
	td := t.TempDir()
	touch := func(name string) string {
		p := filepath.Join(td, name)
		if err := os.WriteFile(p, []byte{0}, 0o644); err != nil {
			t.Fatalf("touch: %v", err)
		}
		return p
	}
	touch("mmproj-gemma-4-E2B.gguf")
	decoderPath := touch("gemma-4-E2-it.gguf") // hypothetical smaller-size decoder

	m := &Manager{enableMmproj: true}
	if got := m.findMmprojForGGUF(decoderPath); got != "" {
		t.Errorf("findMmprojForGGUF spurious match: gemma-4-E2 should NOT match gemma-4-E2B, got %q", got)
	}
}
