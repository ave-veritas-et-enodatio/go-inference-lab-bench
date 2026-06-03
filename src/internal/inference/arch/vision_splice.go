package arch

import (
	"fmt"
	"unsafe"

	"inference-lab-bench/internal/ggml"
)

// See ARCHITECTURE.md "Vision / Multimodal Subsystem → Splice and Decoder
// Integration". This wires the
// vision encoder + projector (vision.go::BuildVisionGraph) into the
// decoder's prefill graph: each preprocessed image runs through the
// encoder, producing a [n_decoder_embd × N] tensor of projected
// embeddings, which is then spliced into the input-embedding stream at
// the position run claimed by that image's `<|image|>` placeholders.

// VisionSpliceInput is the per-image input for the prefill splice. The
// caller (inference engine) constructs one of these per image attached
// to the request, in template-render order.
type VisionSpliceInput struct {
	Preprocessed *PreprocessedImage // pixel data + dims + per-patch positions
	Start        int                // first index in inpEmbd this image occupies
	Length       int                // = Preprocessed.NTokens(NMerge); placeholder run length
}

// visionDecoderSpans returns the bidirectional image spans for the decoder
// mask, but ONLY when the arch opts into non-causal image-token attention
// (Gemma 4's [vision].decoder_non_causal). Towers that keep image tokens
// causal in the decoder (Qwen3-VL and the llama-mtmd default — see
// mtmd_decode_use_non_causal, which is true only for GEMMA3/GEMMA4V) get nil,
// so their image spans stay causal like the surrounding text. Data-driven: no
// arch branch.
func (m *GenericModel) visionDecoderSpans(images []VisionSpliceInput) []MaskSpan {
	if m.Def.Vision == nil || !m.Def.Vision.DecoderNonCausal {
		return nil
	}
	return VisionMaskSpans(images)
}

// VisionMaskSpans projects the per-image splice runs into the
// MaskSpan slice consumed by buildCausalMaskData. Each image's
// position run becomes one non-causal span so vision tokens attend
// bidirectionally to each other — matching llama-mtmd's
// non-causal-during-image-decode behavior. Returns nil when there
// are no images.
func VisionMaskSpans(images []VisionSpliceInput) []MaskSpan {
	if len(images) == 0 {
		return nil
	}
	out := make([]MaskSpan, len(images))
	for i, img := range images {
		out[i] = MaskSpan{Start: img.Start, Length: img.Length}
	}
	return out
}

// RewriteTokenIDsForVision returns a copy of tokenIDs with vision-span
// positions overwritten with token ID 0 (the padding token by Gemma
// convention). Used for both decoder-embedding and per-layer-embedding
// GetRows lookups:
//
//   - Decoder embedding (token_embd): vision positions get the padding
//     token's row, but those rows are immediately overwritten by the
//     vision splice via SetRows. Side-effect-free.
//   - Per-layer embedding (per_layer_tok_embd): vision positions get
//     the padding token's per-layer embedding row. This matches
//     llama-mtmd's gemma4.cpp:415-428 multimodal path, which views
//     row 0 (padding) and broadcasts it across the whole image batch
//     instead of GetRows'ing per token.
//
// Returns tokenIDs unchanged when there are no vision images, so the
// non-vision path pays nothing.
func RewriteTokenIDsForVision(tokenIDs []int32, images []VisionSpliceInput) []int32 {
	if len(images) == 0 {
		return tokenIDs
	}
	out := make([]int32, len(tokenIDs))
	copy(out, tokenIDs)
	for _, img := range images {
		end := img.Start + img.Length
		if end > len(out) {
			end = len(out)
		}
		for j := img.Start; j < end; j++ {
			out[j] = 0
		}
	}
	return out
}

// visionSpliceContext carries the input tensors that the per-image
// vision encoder reads. They're created during graph build (descriptors
// only, AllocPermDisallow) and filled with preprocessed data after
// sched.AllocGraph runs — same pattern as inpTokens / inpPos / inpMask.
type visionSpliceContext struct {
	inputs []visionImageInputs
}

type visionImageInputs struct {
	pp        *PreprocessedImage
	inpRaw    ggml.Tensor // F32 [W, H, 3, 1]
	posX      ggml.Tensor // I32 [n_patches]
	posY      ggml.Tensor // I32 [n_patches]
	posMRope  ggml.Tensor // I32 [4*n_patches] M-RoPE buffer; NilTensor for axial towers
	mropeData []int32     // precomputed M-RoPE buffer; nil for axial towers
	rowsIdx   ggml.Tensor // I64 [Length]; SetRows indices for this image's splice
	start     int         // copy of Start (for filling rowsIdx after AllocGraph)
	length    int
}

// buildVisionSplice constructs the encoder graphs for each image and the
// chained SetRows ops that splice their projected embeddings into
// inpEmbd. Returns the post-splice input-embedding tensor and a context
// carrying the input tensors that need data fill after AllocGraph.
//
// Called from forwardStatelessCore and ForwardCached's analog between
// the initial GetRows+EmbedScale and runLayers — the layer loop consumes
// the returned (possibly-modified) embedding tensor.
func (m *GenericModel) buildVisionSplice(
	ctx *ggml.GraphContext,
	gf *ggml.Graph,
	inpEmbd ggml.Tensor,
	images []VisionSpliceInput,
	caps *ForwardCaptures,
) (ggml.Tensor, *visionSpliceContext, error) {
	if len(images) == 0 {
		return inpEmbd, nil, nil
	}
	if m.VisionTensors == nil || m.VisionParams == nil {
		return ggml.NilTensor(), nil, fmt.Errorf("vision splice: model has no vision tower (request attached images to a unimodal model)")
	}

	sc := &visionSpliceContext{inputs: make([]visionImageInputs, len(images))}
	x := inpEmbd
	for i, img := range images {
		if img.Preprocessed == nil {
			return ggml.NilTensor(), nil, fmt.Errorf("vision splice: image %d has nil Preprocessed", i)
		}
		if img.Length <= 0 || img.Start < 0 {
			return ggml.NilTensor(), nil, fmt.Errorf("vision splice: image %d has invalid run start=%d length=%d", i, img.Start, img.Length)
		}

		pp := img.Preprocessed
		nPatches := int64(pp.NPatchesX * pp.NPatchesY)

		inpRaw := ggml.NewTensor4D(ctx, ggml.TypeF32, int64(pp.Width), int64(pp.Height), 3, 1)
		ggml.SetInput(inpRaw)
		if inpRaw.IsNil() {
			return ggml.NilTensor(), nil, fmt.Errorf("vision splice: failed to allocate input tensors for image %d", i)
		}

		// Axial-2D towers (Gemma) consume per-patch PosX/PosY via GetRows; M-RoPE
		// towers (Qwen3-VL) do not — they read the 4-channel InpPosVision buffer
		// instead. Allocating PosX/PosY for the M-RoPE path would leave them
		// unconsumed by the graph, so the scheduler never backs them with a
		// buffer and the post-AllocGraph fill would write into a NULL tensor.
		// So we only build the buffers the active tower's graph actually reads.
		usesMRope := m.VisionBuilders != nil && m.VisionBuilders.usesMRopeVision()
		var posX, posY ggml.Tensor = ggml.NilTensor(), ggml.NilTensor()
		if !usesMRope {
			posX = ggml.NewTensor1D(ctx, ggml.TypeI32, nPatches)
			ggml.SetInput(posX)
			posY = ggml.NewTensor1D(ctx, ggml.TypeI32, nPatches)
			ggml.SetInput(posY)
			if posX.IsNil() || posY.IsNil() {
				return ggml.NilTensor(), nil, fmt.Errorf("vision splice: failed to allocate position tensors for image %d", i)
			}
		}

		// M-RoPE position buffer for the Qwen3-VL tower (rope=mrope_vision).
		// NilTensor + nil data for the Gemma axial tower, so the existing path
		// is untouched. Computed CPU-side here; filled after AllocGraph.
		var posMRope ggml.Tensor = ggml.NilTensor()
		var mropeData []int32
		if usesMRope {
			mropeData = VisionMRopePositions(pp.NPatchesX, pp.NPatchesY, m.VisionParams.NMerge)
			if mropeData == nil {
				return ggml.NilTensor(), nil, fmt.Errorf("vision splice: image %d patch grid %dx%d not divisible by spatial merge %d",
					i, pp.NPatchesX, pp.NPatchesY, m.VisionParams.NMerge)
			}
			posMRope = ggml.NewTensor1D(ctx, ggml.TypeI32, int64(len(mropeData)))
			ggml.SetInput(posMRope)
			if posMRope.IsNil() {
				return ggml.NilTensor(), nil, fmt.Errorf("vision splice: failed to allocate M-RoPE buffer for image %d", i)
			}
		}

		projected, err := BuildVisionGraph(ctx, gf, &VisionInputs{
			InpRaw:       inpRaw,
			PosX:         posX,
			PosY:         posY,
			InpPosVision: posMRope,
			NPatchesX:    pp.NPatchesX,
			NPatchesY:    pp.NPatchesY,
		}, m.VisionParams, m.VisionTensors, m.VisionBuilders, caps)
		if err != nil {
			return ggml.NilTensor(), nil, fmt.Errorf("vision splice: encode image %d: %w", i, err)
		}
		if projected.IsNil() {
			return ggml.NilTensor(), nil, fmt.Errorf("vision splice: encoder returned nil for image %d", i)
		}
		// Sanity check projector output token count vs declared splice length —
		// catches a preprocess↔arch.toml drift before SetRows fails opaquely.
		if got := projected.Ne(1); got != int64(img.Length) {
			return ggml.NilTensor(), nil, fmt.Errorf("vision splice: image %d projector emitted %d tokens, splice expects %d",
				i, got, img.Length)
		}

		rowsIdx := ggml.NewTensor1D(ctx, ggml.TypeI64, int64(img.Length))
		ggml.SetInput(rowsIdx)
		if rowsIdx.IsNil() {
			return ggml.NilTensor(), nil, fmt.Errorf("vision splice: failed to allocate row-index tensor for image %d", i)
		}

		x = ggml.SetRows(ctx, x, projected, rowsIdx)
		if x.IsNil() {
			return ggml.NilTensor(), nil, fmt.Errorf("vision splice: SetRows returned nil for image %d", i)
		}

		sc.inputs[i] = visionImageInputs{
			pp:        pp,
			inpRaw:    inpRaw,
			posX:      posX,
			posY:      posY,
			posMRope:  posMRope,
			mropeData: mropeData,
			rowsIdx:   rowsIdx,
			start:     img.Start,
			length:    img.Length,
		}
	}
	return x, sc, nil
}

// fillVisionInputs writes the preprocessed pixel / position data and
// computed row-index buffers into the input tensors created by
// buildVisionSplice. Must be called after sched.AllocGraph (which is
// what backs the tensors with real memory).
func (sc *visionSpliceContext) fillVisionInputs() {
	for _, in := range sc.inputs {
		// Pixels: F32, already channel-major in PreprocessedImage.Pixels.
		// Length = 3 * W * H. ggml's tensor layout matches (ne[0]=W innermost).
		setF32(in.inpRaw, in.pp.Pixels)
		setI32(in.posX, in.pp.PosX)
		setI32(in.posY, in.pp.PosY)
		if in.mropeData != nil {
			setI32(in.posMRope, in.mropeData)
		}

		// Row indices: [start, start+1, ..., start+length-1].
		idx := make([]int64, in.length)
		for j := 0; j < in.length; j++ {
			idx[j] = int64(in.start + j)
		}
		setI64(in.rowsIdx, idx)
	}
}

// Helpers — small enough to keep local; mirror the unsafe.Pointer +
// element-size pattern used by graph.go::ForwardStateless when filling
// inpTokens/inpPos/inpMask.
func setF32(t ggml.Tensor, data []float32) {
	if len(data) == 0 || t.IsNil() {
		return
	}
	ggml.TensorSet(t, unsafe.Pointer(&data[0]), 0, len(data)*4)
}

func setI32(t ggml.Tensor, data []int32) {
	if len(data) == 0 || t.IsNil() {
		return
	}
	ggml.TensorSet(t, unsafe.Pointer(&data[0]), 0, len(data)*4)
}

func setI64(t ggml.Tensor, data []int64) {
	if len(data) == 0 || t.IsNil() {
		return
	}
	ggml.TensorSet(t, unsafe.Pointer(&data[0]), 0, len(data)*8)
}
