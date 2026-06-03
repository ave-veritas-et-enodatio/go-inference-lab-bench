package inference

import (
	"fmt"
	"image"

	"inference-lab-bench/internal/inference/arch"
)

// See ARCHITECTURE.md "Vision / Multimodal Subsystem → Splice and Decoder
// Integration". This connects three pieces that previously lived in isolation:
//
//   1. ChatMessage.Parts → chat template emits one `<|image|>` token per
//      image (verified in chat_template.jinja:322-324 for Gemma 4).
//   2. arch.PreprocessImage → an [n_embd × N] projected embedding tensor
//      per image, where N depends on the preprocessed dimensions.
//   3. arch.BuildVisionGraph + arch's prefill path → the embedding stream
//      that flows into the decoder.
//
// This file bridges (1) and (2): it expands each single-token `<|image|>`
// placeholder in the tokenized prompt into N copies of the placeholder
// ID, where N is the per-image token count after preprocessing. The
// resulting position runs (one per image) tell arch's splice code exactly
// where to overwrite throwaway rows with projected vision embeddings.

// ImagePlaceholderRun records a contiguous range in the (expanded)
// tokenized prompt where one image's projected embeddings will be
// spliced in. Tokens at [Start, Start+Length) are all copies of the
// image-placeholder ID — placeholders for the splice to overwrite.
type ImagePlaceholderRun struct {
	Start    int // index into the expanded token list
	Length   int // = N tokens per image (post-preprocessing pool)
	ImageIdx int // index into GenerateParams.Images
}

// VisionPrefill bundles the data the prefill path needs to splice
// projected vision embeddings into the input-embedding stream.
type VisionPrefill struct {
	// ExpandedTokens is the tokenized prompt with each `<|image|>`
	// placeholder replaced by N copies (one per soft token for that
	// image). Total length = original_len + Σ(N-1) per image.
	ExpandedTokens []int32
	// Runs lists the splice ranges, one per image, in order of appearance
	// in the expanded token stream.
	Runs []ImagePlaceholderRun
	// Preprocessed is the per-image preprocessor output, indexed
	// 1:1 with Runs.
	Preprocessed []*arch.PreprocessedImage
}

// prepareVisionPrefill preprocesses each attached image, then expands
// the tokenized prompt so each `<|image|>` placeholder occupies the
// right number of positions for the splice. Returns (nil, nil) for
// requests with no images attached.
//
// Errors:
//   - The chat template's placeholder count doesn't match the number
//     of attached images (template emitted too few/too many `<|image|>`
//     tokens for the given image set).
//   - PreprocessImage fails for any image (degenerate dimensions,
//     invalid config, etc.).
//   - The model's arch has no `[vision]` block (request attached
//     images to a unimodal model).
func prepareVisionPrefill(
	def *arch.ArchDef,
	tokens []int32,
	placeholderID int32,
	images []ChatImage,
) (*VisionPrefill, error) {
	if len(images) == 0 {
		return nil, nil
	}
	if def == nil || def.Vision == nil {
		return nil, fmt.Errorf("vision prefill: model has no [vision] block — image inputs not supported")
	}
	cfg, err := arch.PreprocConfigFromArchDef(def)
	if err != nil {
		return nil, fmt.Errorf("vision prefill: %w", err)
	}

	preprocessed := make([]*arch.PreprocessedImage, len(images))
	runLengths := make([]int, len(images))
	for i, img := range images {
		pp, err := arch.PreprocessImage(img.Image, cfg)
		if err != nil {
			return nil, fmt.Errorf("vision prefill: preprocess image %d: %w", i, err)
		}
		preprocessed[i] = pp
		runLengths[i] = pp.NTokens(cfg.NMerge)
	}

	expanded, runs, err := expandImagePlaceholders(tokens, placeholderID, runLengths)
	if err != nil {
		return nil, fmt.Errorf("vision prefill: %w", err)
	}
	return &VisionPrefill{
		ExpandedTokens: expanded,
		Runs:           runs,
		Preprocessed:   preprocessed,
	}, nil
}

// expandImagePlaceholders rewrites a token stream so each occurrence of
// placeholderID is replaced by runLengths[k] copies of itself, where k
// is the placeholder's positional index. Returns the expanded token list
// and the runs (start positions and lengths) for the splice.
//
// The number of placeholder occurrences in `tokens` must equal len(runLengths) —
// a mismatch is a hard error (template emitted the wrong placeholder
// count vs. images attached, or vice versa).
func expandImagePlaceholders(
	tokens []int32,
	placeholderID int32,
	runLengths []int,
) ([]int32, []ImagePlaceholderRun, error) {
	if placeholderID < 0 {
		return nil, nil, fmt.Errorf("expandImagePlaceholders: invalid placeholder ID %d", placeholderID)
	}
	// Count occurrences first so we can size the output exactly.
	var nPlaceholders int
	for _, t := range tokens {
		if t == placeholderID {
			nPlaceholders++
		}
	}
	if nPlaceholders != len(runLengths) {
		return nil, nil, fmt.Errorf(
			"expandImagePlaceholders: %d placeholder tokens in prompt vs %d images attached",
			nPlaceholders, len(runLengths))
	}
	if nPlaceholders == 0 {
		return tokens, nil, nil
	}

	totalLen := len(tokens) - nPlaceholders
	for _, n := range runLengths {
		if n <= 0 {
			return nil, nil, fmt.Errorf("expandImagePlaceholders: non-positive run length %d", n)
		}
		totalLen += n
	}
	expanded := make([]int32, 0, totalLen)
	runs := make([]ImagePlaceholderRun, 0, nPlaceholders)

	idx := 0 // index into runLengths
	for _, t := range tokens {
		if t != placeholderID {
			expanded = append(expanded, t)
			continue
		}
		runs = append(runs, ImagePlaceholderRun{
			Start:    len(expanded),
			Length:   runLengths[idx],
			ImageIdx: idx,
		})
		for j := 0; j < runLengths[idx]; j++ {
			expanded = append(expanded, placeholderID)
		}
		idx++
	}
	return expanded, runs, nil
}

// chatImagesFromGoImages is a tiny convenience for tests / callers that
// have []image.Image and want []ChatImage.
func chatImagesFromGoImages(imgs []image.Image) []ChatImage {
	out := make([]ChatImage, len(imgs))
	for i, img := range imgs {
		out[i] = ChatImage{Image: img}
	}
	return out
}

// toArchSpliceInputs converts an inference-layer VisionPrefill into the
// arch-layer slice that the forward path's splice helper consumes.
// Returns nil for nil/empty prefill.
func (vp *VisionPrefill) toArchSpliceInputs() []arch.VisionSpliceInput {
	if vp == nil || len(vp.Runs) == 0 {
		return nil
	}
	out := make([]arch.VisionSpliceInput, len(vp.Runs))
	for i, r := range vp.Runs {
		out[i] = arch.VisionSpliceInput{
			Preprocessed: vp.Preprocessed[r.ImageIdx],
			Start:        r.Start,
			Length:       r.Length,
		}
	}
	return out
}
