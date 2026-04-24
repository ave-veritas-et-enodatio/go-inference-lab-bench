package inference

import (
	"fmt"
	"math"
	"sort"
	"time"

	log "inference-lab-bench/internal/log"
)

// gumbelPerturb adds Gumbel(0,1) noise scaled by temperature to a confidence
// score, enabling stochastic position selection via the Gumbel-top-K trick.
// This is statistically equivalent to sampling positions without replacement
// from a distribution weighted by softmax(confidence/temperature).
// u must be uniform(0,1); pseudoRand() satisfies this.
func gumbelPerturb(confidence, temperature float32, u float64) float32 {
	// Gumbel(0,1) sample: -log(-log(u))
	gumbel := float32(-math.Log(-math.Log(u)))
	return confidence/temperature + gumbel
}

// maxSoftmax computes a numerically stable softmax over logits and returns the
// maximum probability. Used as the confidence score in diffusion generation.
// Callers must ensure len(logits) > 0; the nVocab guard in generateDiffusion
// enforces this invariant before any call to maxSoftmax.
func maxSoftmax(logits []float32) float32 {
	maxL := logits[0]
	for _, v := range logits {
		if v > maxL {
			maxL = v
		}
	}
	sum := float32(0)
	for _, v := range logits {
		sum += float32(math.Exp(float64(v - maxL)))
	}
	best := float32(0)
	for _, v := range logits {
		p := float32(math.Exp(float64(v-maxL))) / sum
		if p > best {
			best = p
		}
	}
	return best
}

// linearUnmaskSchedule distributes count unmaskings across steps steps.
// Returns a []int of length steps summing to count, with remainder spread
// over the first (count % steps) entries.
func linearUnmaskSchedule(count, steps int) []int {
	base := count / steps
	extra := count % steps
	s := make([]int, steps)
	for i := range steps {
		if i < extra {
			s[i] = base + 1
		} else {
			s[i] = base
		}
	}
	return s
}

// generateDiffusion implements block-based iterative masked denoising for diffusion
// models (e.g. LLaDA). The output is divided into contiguous blocks which are
// denoised left-to-right; completed earlier blocks condition the model for later
// ones. When DiffusionBlockLength <= 0 or >= outputLen, a single global block is
// used (equivalent to the original simultaneous denoising approach).
func (e *Engine) generateDiffusion(
	promptIDs []int32, maxTokens int, stopSet map[int32]bool,
	params GenerateParams,
	onToken func(string) bool, metrics *InferenceMetrics,
) error {
	maskID := e.tokenizer.MaskTokenID()
	if maskID == -1 {
		return fmt.Errorf("diffusion model has no mask_token_id in GGUF")
	}

	outputLen := maxTokens
	promptLen := len(promptIDs)

	log.Info("diffusion: promptLen=%d outputLen=%d maskID=%d", promptLen, outputLen, maskID)

	seq := make([]int32, promptLen+outputLen)
	copy(seq, promptIDs)
	for i := range outputLen {
		seq[promptLen+i] = maskID
	}

	const defaultDiffusionSteps = 64
	var T int
	var blockLength int
	if params.Diffusion != nil {
		T = params.Diffusion.Steps
		blockLength = params.Diffusion.BlockLength
	}
	if T <= 0 {
		T = defaultDiffusionSteps
	}

	// Determine block layout. Single block when blockLength is unset or covers all output.
	if blockLength <= 0 || blockLength >= outputLen {
		blockLength = outputLen
	}
	numBlocks := (outputLen + blockLength - 1) / blockLength

	// Build per-block step budget, distributing remainder to first blocks.
	// e.g. T=64, numBlocks=3 → [22, 21, 21] (sums to 64, no steps dropped).
	stepsPerBlock := make([]int, numBlocks)
	baseSteps := T / numBlocks
	extraSteps := T % numBlocks
	for i := range stepsPerBlock {
		stepsPerBlock[i] = baseSteps
		if i < extraSteps {
			stepsPerBlock[i]++
		}
		if stepsPerBlock[i] < 1 {
			stepsPerBlock[i] = 1
		}
	}

	type scored struct {
		pos        int
		tokenID    int32
		confidence float32
	}

	decodeStart := time.Now()

	var lastAllLogits []float32

	for blockNum := range numBlocks {
		blockStart := promptLen + blockNum*blockLength
		blockEnd := blockStart + blockLength
		if blockEnd > promptLen+outputLen {
			blockEnd = promptLen + outputLen
		}

		// Collect masked positions in this block.
		blockMasked := make(map[int]struct{}, blockEnd-blockStart)
		for p := blockStart; p < blockEnd; p++ {
			blockMasked[p] = struct{}{}
		}

		blockMaskCount := len(blockMasked)
		schedule := linearUnmaskSchedule(blockMaskCount, stepsPerBlock[blockNum])

		for step := range stepsPerBlock[blockNum] {
			if len(blockMasked) == 0 {
				break
			}

			// Full sequence every time — completed earlier blocks condition the model.
			allLogits, err := e.model.ForwardStatelessAllLogits(seq, *params.FlashAttention)
			if err != nil {
				return fmt.Errorf("diffusion block %d step %d forward: %w", blockNum, step, err)
			}
			if err := ValidateLogits(allLogits); err != nil {
				return fmt.Errorf("diffusion block %d step %d: %w", blockNum, step, err)
			}
			lastAllLogits = allLogits
			nVocab := len(allLogits) / len(seq)
			if nVocab <= 0 {
				return fmt.Errorf("generateDiffusion: ForwardStatelessAllLogits returned %d floats for %d tokens (expected nVocab > 0)", len(allLogits), len(seq))
			}

			// Score only the masked positions in this block.
			scores := make([]scored, 0, len(blockMasked))
			for p := range blockMasked {
				logitIdx := p
				if e.model.Def.Architecture.ShiftLogits {
					// Output positions start at promptLen; p is always >= promptLen >= 1,
					// so p-1 is always a valid index.
					logitIdx = p - 1
				}
				logitSlice := allLogits[logitIdx*nVocab : (logitIdx+1)*nVocab]
				tokenID := TopP(logitSlice, params.Temperature, 1.0)
				confidence := maxSoftmax(logitSlice)
				scores = append(scores, scored{pos: p, tokenID: tokenID, confidence: confidence})
			}

			toUnmask := schedule[step]
			if toUnmask > len(scores) {
				toUnmask = len(scores)
			}

			// Select toUnmask positions by confidence.
			// Temperature > 0: Gumbel-top-K (stochastic, weighted by confidence).
			//   Perturbed scores are pre-computed once per position so that
			//   sort.Slice sees a stable ordering (comparator called many times).
			// Temperature == 0: deterministic sort by confidence descending.
			if params.Temperature > 0 {
				perturbed := make([]float32, len(scores))
				for i, s := range scores {
					perturbed[i] = gumbelPerturb(s.confidence, params.Temperature, pseudoRand())
				}
				sort.Slice(scores, func(a, b int) bool {
					return perturbed[a] > perturbed[b]
				})
			} else {
				sort.Slice(scores, func(a, b int) bool {
					return scores[a].confidence > scores[b].confidence
				})
			}
			for _, s := range scores[:toUnmask] {
				seq[s.pos] = s.tokenID
				delete(blockMasked, s.pos)
			}

			log.Debug("diffusion block %d/%d step %d/%d: unmasked %d, %d remaining",
				blockNum+1, numBlocks, step+1, stepsPerBlock[blockNum], toUnmask, len(blockMasked))
		}

		// Resolve any positions still masked after all steps for this block.
		if len(blockMasked) > 0 && lastAllLogits != nil {
			nVocab := len(lastAllLogits) / len(seq)
			for p := range blockMasked {
				logitIdx := p
				if e.model.Def.Architecture.ShiftLogits {
					// Output positions start at promptLen; p is always >= promptLen >= 1,
					// so p-1 is always a valid index.
					logitIdx = p - 1
				}
				seq[p] = TopP(lastAllLogits[logitIdx*nVocab:(logitIdx+1)*nVocab], params.Temperature, 1.0)
			}
		}
	}

	metrics.DecodeDuration = time.Since(decodeStart)
	// PrefillDuration is intentionally left at zero. Diffusion has no separate
	// prefill phase — every denoising step is a full-sequence forward pass, so
	// the entire wall time is decode time. Do not add a prefill measurement here.

	// Final forward pass for logprob extraction. Run once on the completed
	// sequence only when LogProbs is requested; normal generation is unaffected.
	var logprobLogits []float32
	var logprobNVocab int
	if params.LogProbs {
		lpl, err := e.model.ForwardStatelessAllLogits(seq, *params.FlashAttention)
		if err != nil {
			return fmt.Errorf("diffusion logprob forward: %w", err)
		}
		if err := ValidateLogits(lpl); err != nil {
			return fmt.Errorf("diffusion logprob: %w", err)
		}
		logprobLogits = lpl
		logprobNVocab = len(lpl) / len(seq)
	}

	// Emit output tokens in sequence order, stopping at the first stop token.
	// Diffusion models predict EOS for positions beyond the response length;
	// truncating here matches llama.cpp behavior and user expectations.
	hitStop := false
	for i := promptLen; i < len(seq); i++ {
		if stopSet[seq[i]] {
			hitStop = true
			break
		}
		if params.LogProbs && logprobLogits != nil {
			logitIdx := i
			if e.model.Def.Architecture.ShiftLogits {
				logitIdx = i - 1
			}
			logitSlice := logprobLogits[logitIdx*logprobNVocab : (logitIdx+1)*logprobNVocab]
			metrics.TokenLogProbs = append(metrics.TokenLogProbs,
				ComputeTopLogProbs(logitSlice, seq[i], params.TopLogProbs, e.tokenizer.TokenString))
		}
		if !onToken(e.tokenizer.TokenString(seq[i])) {
			break
		}
		metrics.CompletionTokens++
	}
	if hitStop {
		metrics.FinishReason = FinishReasonStop
	} else {
		metrics.FinishReason = FinishReasonLength
	}
	return nil
}
