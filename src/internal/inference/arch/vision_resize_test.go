package arch

import (
	"image"
	"testing"
)

// TestResizeBilinearLlama pins resizeBilinearLlama to llama.cpp's
// RESIZE_ALGO_BILINEAR convention (tools/mtmd/mtmd-image.cpp resize_bilinear):
// align-corners coordinate mapping px = x*(srcW-1)/(dstW-1), 2-tap lerp, result
// truncated to uint8. Expected values are HAND-DERIVED from that formula (see
// the per-case comments), not produced by the implementation — so this guards
// against silent regressions to x/image/draw's half-pixel/antialiased BiLinear.
func TestResizeBilinearLlama(t *testing.T) {
	// mkGray builds a 1-channel test image as RGBA with R=G=B=v.
	mkGray := func(w, h int, vals []uint8) *image.RGBA {
		im := image.NewRGBA(image.Rect(0, 0, w, h))
		for i, v := range vals {
			x, y := i%w, i/w
			o := y*im.Stride + x*4
			im.Pix[o] = v
			im.Pix[o+1] = v
			im.Pix[o+2] = v
			im.Pix[o+3] = 255
		}
		return im
	}
	red := func(im *image.RGBA, w, h int) []uint8 {
		out := make([]uint8, w*h)
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				out[y*w+x] = im.Pix[y*im.Stride+x*4]
			}
		}
		return out
	}

	cases := []struct {
		name             string
		sw, sh           int
		src              []uint8
		dw, dh           int
		want             []uint8
	}{
		{
			// 1D upscale 3->5, x_ratio=(3-1)/(5-1)=0.5, px=0,0.5,1,1.5,2.
			name: "upscale_3to5", sw: 3, sh: 1, src: []uint8{0, 100, 200},
			dw: 5, dh: 1, want: []uint8{0, 50, 100, 150, 200},
		},
		{
			// 1D downscale 4->2, x_ratio=3, px=0,3 -> exact endpoints (no AA).
			name: "downscale_4to2", sw: 4, sh: 1, src: []uint8{0, 90, 180, 255},
			dw: 2, dh: 1, want: []uint8{0, 255},
		},
		{
			// 2x2 -> 3x3, ratio=0.5 both axes; center is the 4-tap average.
			name: "grid_2x2to3x3", sw: 2, sh: 2, src: []uint8{0, 40, 80, 120},
			dw: 3, dh: 3, want: []uint8{0, 20, 40, 40, 60, 80, 80, 100, 120},
		},
		{
			// 1->N: dw=1 leaves x_ratio=0, every dst pixel samples src[0].
			name: "single_col", sw: 1, sh: 1, src: []uint8{170},
			dw: 1, dh: 1, want: []uint8{170},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			src := mkGray(tc.sw, tc.sh, tc.src)
			dst := image.NewRGBA(image.Rect(0, 0, tc.dw, tc.dh))
			resizeBilinearLlama(dst, dst.Bounds(), src)
			got := red(dst, tc.dw, tc.dh)
			for i := range tc.want {
				if got[i] != tc.want[i] {
					t.Errorf("pixel %d: got %d want %d (full got=%v want=%v)",
						i, got[i], tc.want[i], got, tc.want)
				}
			}
		})
	}
}

// TestResizeBilinearLlama_Offset verifies the function honors a nonzero
// destination rect origin (used for center-pad compositing) and leaves the
// surrounding pad pixels untouched.
func TestResizeBilinearLlama_Offset(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 2, 1)) // [10, 250]
	src.Pix[0], src.Pix[1], src.Pix[2], src.Pix[3] = 10, 10, 10, 255
	src.Pix[4], src.Pix[5], src.Pix[6], src.Pix[7] = 250, 250, 250, 255

	dst := image.NewRGBA(image.Rect(0, 0, 4, 1)) // pad with 0
	// write the 2px source into [1,3): align-corners 2->2 is an identity copy.
	resizeBilinearLlama(dst, image.Rect(1, 0, 3, 1), src)

	want := []uint8{0, 10, 250, 0} // pad, src0, src1, pad
	for x := 0; x < 4; x++ {
		if got := dst.Pix[x*4]; got != want[x] {
			t.Errorf("x=%d: got %d want %d", x, got, want[x])
		}
	}
}
