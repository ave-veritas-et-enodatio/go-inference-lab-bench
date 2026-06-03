package arch

// Decoder position-buffer construction.
//
// Most architectures feed a 1-D position buffer ([n_tokens] I32, pos[i]=seqPos+i)
// to RoPE. Qwen3-VL's decoder instead uses interleaved multi-section M-RoPE
// (GGML_ROPE_TYPE_IMROPE), which requires a 4-channel [4·n_tokens] buffer laid
// out channel-major [t.. | h.. | w.. | e..] and assigns image tokens 2-D grid
// positions. This file builds that buffer, mirroring llama.cpp's get_rope_index
// (tools/mtmd/mtmd.cpp `mtmd_image_tokens_get_decoder_pos` +
// mtmd-helper.cpp `set_position_mrope_2d`) op-for-op.
//
// The choice between the two is data-driven: usesImropeDecoder() inspects the
// arch's decoder block configs for rope="imrope" — never the architecture name.

import "inference-lab-bench/internal/log"

// usesImropeDecoder reports whether any decoder block declares rope="imrope".
// When true the decoder InpPos is the wide [4·n_tokens] IMROPE buffer; otherwise
// it is the historical 1-D [n_tokens] buffer. Cheap (few blocks); called per
// forward. Drives the layout off arch config, not arch name.
func (m *GenericModel) usesImropeDecoder() bool {
	for _, blk := range m.Def.Blocks {
		if configStrOr(blk.Config, ConfigRope, "") == RopeImrope {
			return true
		}
	}
	return false
}

// decoderPositionCount returns the InpPos element count for a forward pass of
// nTokens: 4·nTokens for an IMROPE decoder, nTokens otherwise. (IMROPE asserts
// `a->ne[2]*4 == pos->ne[0]` inside ggml; a 1-D buffer would abort.)
func (m *GenericModel) decoderPositionCount(nTokens int64) int64 {
	if m.usesImropeDecoder() {
		return 4 * nTokens
	}
	return nTokens
}

// buildDecoderPositions builds the InpPos data for a forward pass starting at
// absolute position seqPos covering nTokens tokens.
//
//   - Non-IMROPE: returns the 1-D sequential buffer [seqPos .. seqPos+nTokens-1].
//   - IMROPE: returns the channel-major [4·nTokens] buffer. Text tokens get all
//     four blocks = pos_0 (= seqPos running counter, +i within text runs); image
//     spans (from visionImages) get block0=t=pos_0 (const over the image),
//     block1=h=pos_0+row, block2=w=pos_0+col, block3=e=0, over the merged grid
//     nx×ny (nx=NPatchesX/NMerge, ny=NPatchesY/NMerge). After an image the running
//     counter advances by max(nx,ny); the image's splice run length stays nx*ny.
//
// nMerge is the spatial merge size (0 when the model has no vision tower —
// visionImages is then empty so it is never consulted). posCtr is the running
// RoPE position the pass starts at (== seqPos for non-imrope and for the
// stateless single-shot path; == cache.PosCtr for cached imrope decode, which
// can trail the KV slot count after an image). Returns the position buffer and
// the counter value AFTER this pass, which the cached path persists into
// cache.PosCtr for the next decode token.
func (m *GenericModel) buildDecoderPositions(visionImages []VisionSpliceInput, nTokens int64, posCtr int) ([]int32, int) {
	n := int(nTokens)
	if !m.usesImropeDecoder() {
		positions := make([]int32, n)
		for i := range positions {
			positions[i] = int32(posCtr + i)
		}
		return positions, posCtr + n
	}

	nMerge := 0
	if m.VisionParams != nil {
		nMerge = m.VisionParams.NMerge
	}
	return buildImropeDecoderPositions(visionImages, n, posCtr, nMerge)
}

// buildImropeDecoderPositions is the get_rope_index analogue. Returns a
// channel-major [4*n] I32 buffer: blocks [t | h | w | e], each length n.
//
// The running counter `posCtr` starts at seqPos and tracks the next sequential
// position. Text tokens consume one counter value each (all 4 blocks = that
// value). An image span at [start, start+nx*ny) writes 2-D grid positions
// anchored at the counter value entering the span, then advances the counter by
// max(nx,ny) — exactly llama.cpp's `mtmd_image_tokens_get_n_pos` for MROPE.
func buildImropeDecoderPositions(visionImages []VisionSpliceInput, n, startCtr, nMerge int) (positions []int32, endCtr int) {
	pos := make([]int32, 4*n)
	positions = pos
	t := pos[0:n]
	h := pos[n : 2*n]
	w := pos[2*n : 3*n]
	e := pos[3*n : 4*n] // image z-channel = 0; text = counter (set in loop)

	// Index image spans by their start token for O(1) lookup as we sweep tokens.
	type imgSpan struct {
		end, nx, ny int
	}
	spans := make(map[int]imgSpan, len(visionImages))
	for _, img := range visionImages {
		nx, ny := 1, 1
		if nMerge > 0 && img.Preprocessed != nil {
			nx = img.Preprocessed.NPatchesX / nMerge
			ny = img.Preprocessed.NPatchesY / nMerge
		}
		if nx*ny != img.Length {
			// Splice length must equal the merged grid token count; a mismatch
			// means preprocess↔arch drift. Fall back to treating the span as
			// 1×Length so positions stay finite and contiguous rather than
			// scribbling out of range — and log loudly.
			log.Error("buildImropeDecoderPositions: image at start=%d grid %dx%d (=%d tokens) != splice length %d — using 1xLength fallback",
				img.Start, nx, ny, nx*ny, img.Length)
			nx, ny = img.Length, 1
		}
		spans[img.Start] = imgSpan{end: img.Start + img.Length, nx: nx, ny: ny}
	}

	posCtr := startCtr
	i := 0
	for i < n {
		if span, ok := spans[i]; ok {
			pos0 := int32(posCtr)
			nx := span.nx
			for j := 0; i < span.end && i < n; j, i = j+1, i+1 {
				t[i] = pos0
				h[i] = pos0 + int32(j/nx)
				w[i] = pos0 + int32(j%nx)
				e[i] = 0
			}
			advance := span.nx
			if span.ny > advance {
				advance = span.ny
			}
			posCtr += advance
			continue
		}
		// Text token: all four blocks share the sequential counter value.
		v := int32(posCtr)
		t[i] = v
		h[i] = v
		w[i] = v
		e[i] = v
		posCtr++
		i++
	}
	return positions, posCtr
}
