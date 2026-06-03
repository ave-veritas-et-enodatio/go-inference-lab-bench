package inference

import (
	"strings"
	"testing"
)

// TestExpandImagePlaceholders_SingleImage verifies the basic case:
// one placeholder, one image, N copies replace it.
func TestExpandImagePlaceholders_SingleImage(t *testing.T) {
	tokens := []int32{1, 2, 99, 3, 4} // 99 = placeholder
	expanded, runs, err := expandImagePlaceholders(tokens, 99, []int{4})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := []int32{1, 2, 99, 99, 99, 99, 3, 4}
	if !equalI32(expanded, want) {
		t.Errorf("expanded = %v, want %v", expanded, want)
	}
	if len(runs) != 1 {
		t.Fatalf("len(runs) = %d, want 1", len(runs))
	}
	if runs[0].Start != 2 || runs[0].Length != 4 || runs[0].ImageIdx != 0 {
		t.Errorf("runs[0] = %+v, want {Start:2, Length:4, ImageIdx:0}", runs[0])
	}
}

// TestExpandImagePlaceholders_NoImages — token stream contains no
// placeholders and no images attached; passthrough with empty runs.
func TestExpandImagePlaceholders_NoImages(t *testing.T) {
	tokens := []int32{1, 2, 3, 4, 5}
	expanded, runs, err := expandImagePlaceholders(tokens, 99, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !equalI32(expanded, tokens) {
		t.Errorf("expanded = %v, want %v", expanded, tokens)
	}
	if len(runs) != 0 {
		t.Errorf("expected no runs, got %v", runs)
	}
}

// TestExpandImagePlaceholders_MultipleImages — verifies that consecutive
// placeholders get distinct run-length budgets and the run-index → image-
// index correspondence holds.
func TestExpandImagePlaceholders_MultipleImages(t *testing.T) {
	tokens := []int32{99, 1, 99, 2, 3, 99}
	expanded, runs, err := expandImagePlaceholders(tokens, 99, []int{2, 3, 1})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// 99[2] + 1 + 99[3] + 2 + 3 + 99[1] = 2+1+3+1+1+1 = 9 tokens.
	want := []int32{99, 99, 1, 99, 99, 99, 2, 3, 99}
	if !equalI32(expanded, want) {
		t.Errorf("expanded = %v, want %v", expanded, want)
	}
	if len(runs) != 3 {
		t.Fatalf("len(runs) = %d, want 3", len(runs))
	}
	gotStarts := []int{runs[0].Start, runs[1].Start, runs[2].Start}
	wantStarts := []int{0, 3, 8}
	for i := range gotStarts {
		if gotStarts[i] != wantStarts[i] {
			t.Errorf("runs[%d].Start = %d, want %d", i, gotStarts[i], wantStarts[i])
		}
	}
	gotLengths := []int{runs[0].Length, runs[1].Length, runs[2].Length}
	wantLengths := []int{2, 3, 1}
	for i := range gotLengths {
		if gotLengths[i] != wantLengths[i] {
			t.Errorf("runs[%d].Length = %d, want %d", i, gotLengths[i], wantLengths[i])
		}
	}
	for i, r := range runs {
		if r.ImageIdx != i {
			t.Errorf("runs[%d].ImageIdx = %d, want %d", i, r.ImageIdx, i)
		}
	}
}

// TestExpandImagePlaceholders_CountMismatch — template emitted more or
// fewer placeholders than images attached. Must be a loud error.
func TestExpandImagePlaceholders_CountMismatch(t *testing.T) {
	cases := []struct {
		name        string
		tokens      []int32
		runLengths  []int
		wantMessage string
	}{
		{"more_placeholders_than_images", []int32{99, 1, 99}, []int{4}, "2 placeholder tokens in prompt vs 1 images attached"},
		{"fewer_placeholders_than_images", []int32{99}, []int{4, 4}, "1 placeholder tokens in prompt vs 2 images attached"},
		{"no_placeholders_but_images", []int32{1, 2, 3}, []int{4}, "0 placeholder tokens in prompt vs 1 images attached"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, _, err := expandImagePlaceholders(c.tokens, 99, c.runLengths)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !strings.Contains(err.Error(), c.wantMessage) {
				t.Errorf("err = %q, want substring %q", err.Error(), c.wantMessage)
			}
		})
	}
}

// TestExpandImagePlaceholders_BadRunLength — non-positive N is a config
// error (preprocessor should never produce zero tokens for a valid image).
func TestExpandImagePlaceholders_BadRunLength(t *testing.T) {
	_, _, err := expandImagePlaceholders([]int32{99}, 99, []int{0})
	if err == nil {
		t.Fatal("expected error for zero run length, got nil")
	}
	if !strings.Contains(err.Error(), "non-positive run length") {
		t.Errorf("err = %q, want non-positive run length error", err.Error())
	}
}

// TestExpandImagePlaceholders_BadPlaceholderID — caller passed an
// unresolved (-1) placeholder; should error rather than silently match
// nothing.
func TestExpandImagePlaceholders_BadPlaceholderID(t *testing.T) {
	_, _, err := expandImagePlaceholders([]int32{1, 2, 3}, -1, []int{4})
	if err == nil {
		t.Fatal("expected error for invalid placeholder ID, got nil")
	}
}

func equalI32(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
