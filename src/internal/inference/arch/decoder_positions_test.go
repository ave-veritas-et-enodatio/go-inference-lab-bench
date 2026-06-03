package arch

import (
	"reflect"
	"testing"
)

// blocks splits a [4*n] IMROPE position buffer into its 4 channel blocks
// [t, h, w, e] for readable assertions.
func blocks(pos []int32, n int) (t, h, w, e []int32) {
	return pos[0:n], pos[n : 2*n], pos[2*n : 3*n], pos[3*n : 4*n]
}

// TestImropeDecoderPositionsTextOnly pins the IMROPE text reduction: with no
// image spans every token's four blocks must all equal the sequential position
// startCtr+i. This is the property that makes universal IMROPE byte-identical
// to NEOX for text (NEOX reads only block 0; the others are inert but must still
// be self-consistent), so the equiv regression cannot drift on text.
func TestImropeDecoderPositionsTextOnly(t *testing.T) {
	n := 5
	pos, end := buildImropeDecoderPositions(nil, n, 0, 2)
	if len(pos) != 4*n {
		t.Fatalf("len = %d, want %d", len(pos), 4*n)
	}
	bt, bh, bw, be := blocks(pos, n)
	want := []int32{0, 1, 2, 3, 4}
	for _, got := range [][]int32{bt, bh, bw, be} {
		if !reflect.DeepEqual(got, want) {
			t.Errorf("text block = %v, want %v (all 4 blocks must equal sequential pos)", got, want)
		}
	}
	if end != 5 {
		t.Errorf("end counter = %d, want 5", end)
	}
}

// TestImropeDecoderPositionsNonZeroStart confirms a decode continuation: a
// single text token at running counter 7 gets all blocks = 7 and advances to 8.
func TestImropeDecoderPositionsNonZeroStart(t *testing.T) {
	pos, end := buildImropeDecoderPositions(nil, 1, 7, 2)
	for c, got := range pos {
		if got != 7 {
			t.Errorf("block %d = %d, want 7", c, got)
		}
	}
	if end != 8 {
		t.Errorf("end = %d, want 8", end)
	}
}

// TestImropeDecoderPositionsWithImage pins the get_rope_index image-span math
// (mtmd.cpp mtmd_image_tokens_get_decoder_pos, MROPE case) and the counter
// advance (mtmd_image_tokens_get_n_pos = max(nx,ny)). Layout: 2 text tokens,
// then an image whose merged grid is nx=3, ny=2 (6 tokens), then 1 trailing text
// token. NPatchesX/Y = nMerge*nx, nMerge*ny so the merged grid is 3x2.
//
// Expected, hand-derived:
//   pos_0 entering the image span = 2 (after 2 text tokens at counter 0,1).
//   image token i (row=i/nx, col=i%nx, nx=3):
//     i: 0 1 2 3 4 5
//     row(h): 0 0 0 1 1 1  → h = 2 + row
//     col(w): 0 1 2 0 1 2  → w = 2 + col
//     t = 2 (const), e = 0.
//   counter advances by max(3,2)=3 → next text token at 5.
func TestImropeDecoderPositionsWithImage(t *testing.T) {
	const nMerge = 2
	nx, ny := 3, 2
	img := VisionSpliceInput{
		Start:  2,
		Length: nx * ny,
		Preprocessed: &PreprocessedImage{
			NPatchesX: nx * nMerge,
			NPatchesY: ny * nMerge,
		},
	}
	n := 2 + nx*ny + 1 // 2 text + 6 image + 1 text = 9
	pos, end := buildImropeDecoderPositions([]VisionSpliceInput{img}, n, 0, nMerge)

	bt, bh, bw, be := blocks(pos, n)

	wantT := []int32{0, 1, 2, 2, 2, 2, 2, 2, 5}
	wantH := []int32{0, 1, 2, 2, 2, 3, 3, 3, 5}
	wantW := []int32{0, 1, 2, 3, 4, 2, 3, 4, 5}
	wantE := []int32{0, 1, 0, 0, 0, 0, 0, 0, 5}

	if !reflect.DeepEqual(bt, wantT) {
		t.Errorf("t block = %v, want %v", bt, wantT)
	}
	if !reflect.DeepEqual(bh, wantH) {
		t.Errorf("h block = %v, want %v", bh, wantH)
	}
	if !reflect.DeepEqual(bw, wantW) {
		t.Errorf("w block = %v, want %v", bw, wantW)
	}
	if !reflect.DeepEqual(be, wantE) {
		t.Errorf("e block = %v, want %v", be, wantE)
	}
	// Trailing text token at counter 5 (2 + max(3,2)=5), so end = 6.
	if end != 6 {
		t.Errorf("end counter = %d, want 6 (image advanced by max(nx,ny)=3)", end)
	}
}
