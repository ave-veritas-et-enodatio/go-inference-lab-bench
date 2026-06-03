package arch

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"math"
	"runtime"
	"sync"
)

// visionPadColor is the fill color for the aspect-preserving pad border
// (llama.cpp PAD_CEIL). Both bench towers leave hparams.image_pad_color at the
// {0,0,0} default, which normalizes (mean/std = 0.5) to the encoder's -1.0
// background — the value the reference dumps for padded rows.
var visionPadColor = color.RGBA{R: 0, G: 0, B: 0, A: 255}

// Phase 6 of the vision-input plan: image preprocessing for the Gemma 4
// vision encoder. Mirrors the upstream HF "smart_resize" / llama.cpp
// `mtmd_image_preprocessor_dyn_size` pipeline: a single bilinear resize
// to an aspect-preserved target whose pixel count falls in the per-arch
// [min, max] band and whose dimensions are aligned to patch_size × n_merge.
//
// What this does NOT do (intentionally):
//   - Pan-and-scan / multi-tile sub-cropping (the upstream Gemma 4 vision
//     preprocessor is single-image; the multi-tile path applies to other
//     VLMs and would land alongside their preprocess strategies in
//     Phase 10).
//   - Per-channel mean/std normalization (Gemma 4 vision uses identity
//     mean = [0,0,0], std = [1,1,1]; do_normalize = false in the HF
//     processor_config). Other models supplying non-identity stats can
//     extend PreprocConfig with those fields and apply them inside
//     packPixelsF32.

// PreprocConfig is the per-arch image-preprocessing parameter set. Sourced
// from a model-specific helper today; eventually populated from arch.toml
// + stmap metadata so multimodal VLMs are data-driven the way other
// arch params already are.
type PreprocConfig struct {
	PatchSize      int // pixel side of one patch (Gemma 4: 16)
	NMerge         int // pooling kernel = post-encoder spatial merge (Gemma 4: 3)
	ImageMinTokens int // minimum output-token count after pooling (Gemma 4: 252)
	ImageMaxTokens int // maximum output-token count after pooling (Gemma 4: 280)
}

// AlignSize is the pixel multiple that both target dimensions must be
// rounded to: patch_size × n_merge. Aligning here guarantees that the
// post-Conv2D patch grid divides evenly into n_merge × n_merge pool windows,
// so the pooler doesn't need to truncate edge patches.
func (c PreprocConfig) AlignSize() int { return c.PatchSize * c.NMerge }

// MinPixels / MaxPixels translate the token-count band into the pixel-count
// band used by smartResize. Matches the llama.cpp set_limit_image_tokens
// arithmetic: each output token consumes (patch_size × n_merge)² pixels.
func (c PreprocConfig) MinPixels() int {
	a := c.AlignSize()
	return c.ImageMinTokens * a * a
}
func (c PreprocConfig) MaxPixels() int {
	a := c.AlignSize()
	return c.ImageMaxTokens * a * a
}

// validate verifies the config is internally consistent. Cheap; called at
// the top of PreprocessImage so a bad config produces a named error rather
// than NaN-shape failures deep inside resize math.
func (c PreprocConfig) validate() error {
	if c.PatchSize <= 0 || c.NMerge <= 0 {
		return fmt.Errorf("PreprocConfig: patch_size and n_merge must be > 0 (got %d, %d)", c.PatchSize, c.NMerge)
	}
	if c.ImageMinTokens <= 0 || c.ImageMaxTokens <= 0 || c.ImageMinTokens > c.ImageMaxTokens {
		return fmt.Errorf("PreprocConfig: image_min_tokens (%d) and image_max_tokens (%d) must be positive and min ≤ max",
			c.ImageMinTokens, c.ImageMaxTokens)
	}
	return nil
}

// PreprocConfigFromArchDef projects a parsed arch.toml's [vision]
// section into the smaller PreprocConfig that the preprocessing
// pipeline consumes. arch.toml is the single source of truth — both
// production callers (engine.Generate has access to *ArchDef via
// GenericModel.Def) and tests / tooling go through this function.
func PreprocConfigFromArchDef(def *ArchDef) (PreprocConfig, error) {
	if def == nil || def.Vision == nil {
		return PreprocConfig{}, fmt.Errorf("PreprocConfigFromArchDef: arch has no [vision] block")
	}
	cfg := PreprocConfig{
		PatchSize:      def.Vision.PatchSize,
		NMerge:         def.Vision.NMerge,
		ImageMinTokens: def.Vision.ImageMinTokens,
		ImageMaxTokens: def.Vision.ImageMaxTokens,
	}
	return cfg, cfg.validate()
}

// PreprocessedImage is the result of running PreprocessImage. Pixels is
// laid out for direct copy into a ggml tensor with ne = [W, H, 3, 1]:
// channel-major (ne[2] is the channel axis), with H × W rows per channel
// stored y-major then x-minor (so ne[0] = W is innermost as ggml expects).
// PosX / PosY are the per-patch column / row indices into the 2D position
// embedding table, in patch-grid order (col fastest).
type PreprocessedImage struct {
	Pixels    []float32 // length = 3 * Width * Height, channel-major
	Width     int       // target pixel width, multiple of AlignSize
	Height    int       // target pixel height, multiple of AlignSize
	NPatchesX int       // Width / PatchSize
	NPatchesY int       // Height / PatchSize
	PosX      []int32   // length = NPatchesX*NPatchesY, col-major patch indices
	PosY      []int32   // length = NPatchesX*NPatchesY, row-major patch indices
}

// NTokens reports how many decoder soft tokens this preprocessed image
// will produce, given the n_merge pool kernel. For Gemma 4 this matches
// the GenericModel.VisionParams.NImageTokens budget for one image.
func (p *PreprocessedImage) NTokens(nMerge int) int {
	if nMerge <= 0 {
		return 0
	}
	return (p.NPatchesX / nMerge) * (p.NPatchesY / nMerge)
}

// PreprocessImage applies the smart-resize pipeline: convert source to
// 8-bit RGB, compute aspect-preserved target dimensions inside the
// configured pixel band, fit the source inside that target preserving aspect
// (center-padding the border with the pad color, llama.cpp PAD_CEIL), and pack
// the result into a channel-major F32 tensor in the [0, 1] range.
//
// The resize pass uses resizeBilinearLlama, a literal port of llama.cpp's
// RESIZE_ALGO_BILINEAR (align-corners, 2-tap, no antialias — both Gemma4V and
// Qwen3-VL). golang.org/x/image/draw's BiLinear was NOT used: it is half-pixel
// and antialiases on downscale, shifting every resized pixel relative to
// llama-mtmd's inp_raw_scaled. The pad branch
// mirrors img_tool::resize's default PAD_CEIL: without it the source is
// stretched to fill the target, which distorts any image whose aspect ratio
// doesn't already match the aligned target (e.g. Qwen3-VL's 960×624 → 960×640),
// shifting every patch and diverging the encoder from llama-mtmd. Gemma's usual
// target equals the aspect-correct size, so its path reduces to a zero-pad
// full-frame resize — byte-identical to the prior code.
func PreprocessImage(img image.Image, cfg PreprocConfig) (*PreprocessedImage, error) {
	if img == nil {
		return nil, fmt.Errorf("PreprocessImage: nil image")
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	srcBounds := img.Bounds()
	srcW := srcBounds.Dx()
	srcH := srcBounds.Dy()
	if srcW <= 0 || srcH <= 0 {
		return nil, fmt.Errorf("PreprocessImage: degenerate source size %dx%d", srcW, srcH)
	}

	tgtW, tgtH := smartResize(srcW, srcH, cfg.AlignSize(), cfg.MinPixels(), cfg.MaxPixels())
	if tgtW <= 0 || tgtH <= 0 {
		return nil, fmt.Errorf("PreprocessImage: smartResize produced degenerate target %dx%d", tgtW, tgtH)
	}

	// Promote the source to RGBA so the resize pass operates on a known
	// 8-bit packed layout. draw.Draw handles whatever pixel type the
	// source decoder returned (NRGBA, YCbCr, Gray, Paletted, etc.).
	srcRGBA, ok := img.(*image.RGBA)
	if !ok || !srcBounds.Eq(srcRGBA.Rect) {
		conv := image.NewRGBA(srcBounds)
		draw.Draw(conv, srcBounds, img, srcBounds.Min, draw.Src)
		srcRGBA = conv
	}

	// Aspect-preserving fit + center pad (PAD_CEIL), mirroring llama.cpp
	// img_tool::resize's default padding branch (tools/mtmd/mtmd-image.cpp):
	// scale the source by min(tgtW/srcW, tgtH/srcH) so it fits inside the
	// target without distortion, then composite it centered into a target-
	// sized canvas pre-filled with the pad color (black). smartResize already
	// chose an aspect-aligned target, so the unfilled border is at most one
	// align step on the shorter axis. Both bench towers (Qwen3-VL align=32,
	// Gemma 4 align=48) leave image_resize_pad at the PAD_CEIL default, so
	// this single path is correct for both; an exact-aspect target (Gemma's
	// usual case) reduces to a full-frame resize with zero padding.
	scale := math.Min(float64(tgtW)/float64(srcW), float64(tgtH)/float64(srcH))
	innerW := min(int(math.Ceil(float64(srcW)*scale)), tgtW)
	innerH := min(int(math.Ceil(float64(srcH)*scale)), tgtH)
	offX := (tgtW - innerW) / 2
	offY := (tgtH - innerH) / 2

	dst := image.NewRGBA(image.Rect(0, 0, tgtW, tgtH))
	// Pre-fill with the pad color (opaque black). resizeBilinearLlama writes
	// the fitted inner rect; the border keeps the pad color.
	draw.Draw(dst, dst.Bounds(), image.NewUniform(visionPadColor), image.Point{}, draw.Src)
	resizeBilinearLlama(dst, image.Rect(offX, offY, offX+innerW, offY+innerH), srcRGBA)

	pixels := packPixelsF32(dst, tgtW, tgtH)
	posX, posY := buildPatchPositions(tgtW/cfg.PatchSize, tgtH/cfg.PatchSize)

	return &PreprocessedImage{
		Pixels:    pixels,
		Width:     tgtW,
		Height:    tgtH,
		NPatchesX: tgtW / cfg.PatchSize,
		NPatchesY: tgtH / cfg.PatchSize,
		PosX:      posX,
		PosY:      posY,
	}, nil
}

// smartResize is a literal port of llama.cpp's calc_size_preserved_ratio
// (the (orig, align, min_px, max_px) overload) in
// tools/mtmd/mtmd-image.cpp. Maintains aspect ratio while clamping the
// total pixel count to the configured band and snapping both axes to a
// multiple of align.
//
// Algorithm:
//  1. Round each axis to align (with a floor of align).
//  2. If the rounded area exceeds max_px, downscale by sqrt(area/max_px)
//     and floor-align (so the result stays under max).
//  3. If under min_px, upscale by sqrt(min_px/area) and ceil-align (so the
//     result stays at or above min).
//
// Pure math: kept package-private + unit-tested in isolation so a future
// numerical-diff against the HF reference is grounded.
func smartResize(srcW, srcH, align, minPx, maxPx int) (w, h int) {
	a := float64(align)
	roundBy := func(x float64) int { return int(math.Round(x/a)) * align }
	ceilBy := func(x float64) int { return int(math.Ceil(x/a)) * align }
	floorBy := func(x float64) int { return int(math.Floor(x/a)) * align }

	hBar := max(align, roundBy(float64(srcH)))
	wBar := max(align, roundBy(float64(srcW)))

	switch area := hBar * wBar; {
	case area > maxPx:
		beta := math.Sqrt(float64(srcH*srcW) / float64(maxPx))
		hBar = max(align, floorBy(float64(srcH)/beta))
		wBar = max(align, floorBy(float64(srcW)/beta))
	case area < minPx:
		beta := math.Sqrt(float64(minPx) / float64(srcH*srcW))
		hBar = ceilBy(float64(srcH) * beta)
		wBar = ceilBy(float64(srcW) * beta)
	}

	// Post-clamp area guard. For extreme aspect ratios the downscale branch
	// above can't bring the area under maxPx: its per-axis floor is max(align,
	// …), so once the short axis pins to `align`, dividing by beta only shrinks
	// the long axis proportionally — and align × longAxis can still exceed
	// maxPx. Trim the longer axis in align steps until the area fits. This
	// distorts aspect for pathological inputs (e.g. 200000×300), which is
	// unavoidable once one axis is already at the align floor.
	for wBar*hBar > maxPx {
		switch {
		case wBar >= hBar && wBar > align:
			wBar -= align
		case hBar > align:
			hBar -= align
		default:
			// Both axes at the align floor: align² may exceed a (tiny) maxPx,
			// but nothing further can shrink without producing a zero axis.
			return wBar, hBar
		}
	}

	return wBar, hBar
}

// packPixelsF32 walks the RGBA buffer once and emits a channel-major F32
// view: pixels[c*W*H + y*W + x] = src[y*Stride + x*4 + c] / 255.
//
// Parallelism note: this is the one preprocessing step that benefits from
// goroutine scatter — channel and row strips are independent and the work
// is purely memory-bandwidth-bound. Spawns up to GOMAXPROCS workers, but
// only when the work justifies the dispatch overhead.
func packPixelsF32(rgba *image.RGBA, w, h int) []float32 {
	pixels := make([]float32, 3*w*h)
	stride := rgba.Stride
	pix := rgba.Pix

	const inv255 = 1.0 / 255.0
	const parallelThreshold = 1 << 18 // ~256K pixels — below this, single-threaded wins

	// packRows converts rows [y0, y1) of the RGBA buffer into the channel-major
	// F32 view. The seq/parallel split below is purely about dispatch (whether
	// to spawn goroutines); the pixel math lives here, once.
	packRows := func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			row := pix[y*stride : y*stride+w*4]
			for x := 0; x < w; x++ {
				p := x * 4
				pixels[0*w*h+y*w+x] = float32(row[p+0]) * inv255
				pixels[1*w*h+y*w+x] = float32(row[p+1]) * inv255
				pixels[2*w*h+y*w+x] = float32(row[p+2]) * inv255
			}
		}
	}

	if w*h < parallelThreshold {
		packRows(0, h)
		return pixels
	}

	nWorkers := runtime.GOMAXPROCS(0)
	if nWorkers > h {
		nWorkers = h
	}
	rowsPerWorker := (h + nWorkers - 1) / nWorkers
	var wg sync.WaitGroup
	for i := 0; i < nWorkers; i++ {
		y0 := i * rowsPerWorker
		y1 := y0 + rowsPerWorker
		if y1 > h {
			y1 = h
		}
		if y0 >= y1 {
			break
		}
		wg.Add(1)
		go func(y0, y1 int) {
			defer wg.Done()
			packRows(y0, y1)
		}(y0, y1)
	}
	wg.Wait()
	return pixels
}

// resizeBilinearLlama scales srcRGBA into dst's dstRect using llama.cpp's
// img_tool::resize_bilinear convention (tools/mtmd/mtmd-image.cpp). This is a
// literal port, NOT golang.org/x/image/draw's BiLinear, which differs in two
// ways that shift every resized pixel:
//
//   - Coordinate transform: llama is ALIGN-CORNERS. For a destination of
//     width Wd it maps dst x → src x via x_ratio = (Ws-1)/(Wd-1), px = x*x_ratio
//     (dst 0 → src 0, dst Wd-1 → src Ws-1). x/image/draw uses the HALF-PIXEL
//     center convention src = (x+0.5)*scale - 0.5.
//   - Antialiasing: llama samples exactly the 2 nearest taps per axis with no
//     prefilter, even on downscale; x/image/draw broadens its tent kernel by
//     1/scale when shrinking (a proper resampling prefilter). On a downscale
//     these produce materially different pixels.
//
// Matching op-for-op (uint8 source, lerp, truncate-to-uint8) keeps bench's
// preprocessed image bit-comparable with llama-mtmd's inp_raw_scaled so the
// vision encoder sees the same input. Operates per RGB channel; alpha is set
// opaque (the encoder ignores it and packPixelsF32 reads only RGB).
func resizeBilinearLlama(dst *image.RGBA, dstRect image.Rectangle, src *image.RGBA) {
	sw := src.Rect.Dx()
	sh := src.Rect.Dy()
	dw := dstRect.Dx()
	dh := dstRect.Dy()
	if sw == 0 || sh == 0 || dw <= 0 || dh <= 0 {
		return
	}

	var xRatio, yRatio float64
	if dw > 1 {
		xRatio = float64(sw-1) / float64(dw-1)
	}
	if dh > 1 {
		yRatio = float64(sh-1) / float64(dh-1)
	}

	lerp := func(s, e, t float64) float64 { return s + (e-s)*t }
	// src is a *image.RGBA whose Rect.Min may be nonzero; index relative to it.
	sp := func(x, y, c int) float64 {
		return float64(src.Pix[(src.Rect.Min.Y+y)*src.Stride+(src.Rect.Min.X+x)*4+c])
	}

	for y := 0; y < dh; y++ {
		py := float64(y) * yRatio
		y0 := min(int(py), sh-1)
		y1 := min(y0+1, sh-1)
		yf := py - float64(y0)
		dstRow := (dstRect.Min.Y + y) * dst.Stride
		for x := 0; x < dw; x++ {
			px := float64(x) * xRatio
			x0 := min(int(px), sw-1)
			x1 := min(x0+1, sw-1)
			xf := px - float64(x0)
			o := dstRow + (dstRect.Min.X+x)*4
			for c := 0; c < 3; c++ {
				top := lerp(sp(x0, y0, c), sp(x1, y0, c), xf)
				bottom := lerp(sp(x0, y1, c), sp(x1, y1, c), xf)
				dst.Pix[o+c] = uint8(lerp(top, bottom, yf))
			}
			dst.Pix[o+3] = 255
		}
	}
}

// buildPatchPositions returns the per-patch (x, y) indices into the 2D
// position embedding table, in patch-grid order (col fastest, row outer).
// Output length = nPatchesX * nPatchesY.
func buildPatchPositions(nPatchesX, nPatchesY int) (posX, posY []int32) {
	n := nPatchesX * nPatchesY
	posX = make([]int32, n)
	posY = make([]int32, n)
	i := 0
	for py := 0; py < nPatchesY; py++ {
		for px := 0; px < nPatchesX; px++ {
			posX[i] = int32(px)
			posY[i] = int32(py)
			i++
		}
	}
	return posX, posY
}

