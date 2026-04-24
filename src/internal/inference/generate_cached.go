package inference

import (
	"fmt"
	"time"
)

func (e *Engine) generateCached(
	promptIDs []int32, maxTokens int, stopSet map[int32]bool,
	params GenerateParams,
	onToken func(string) bool, metrics *InferenceMetrics,
) error {
	cache, err := e.model.NewCache(e.maxSeqLen)
	if err != nil {
		return fmt.Errorf("creating cache: %w", err)
	}
	defer cache.Free()

	prefillStart := time.Now()
	logits, err := e.model.ForwardCached(cache, promptIDs, *params.FlashAttention)
	if err != nil {
		return fmt.Errorf("prefill forward: %w", err)
	}
	metrics.PrefillDuration = time.Since(prefillStart)

	decodeStart := time.Now()
	hitStop := false
	for range maxTokens {
		nextID, err := e.sample(logits, params)
		if err != nil {
			return err
		}
		if stopSet[nextID] {
			hitStop = true
			break
		}
		if params.LogProbs {
			metrics.TokenLogProbs = append(metrics.TokenLogProbs,
				ComputeTopLogProbs(logits, nextID, params.TopLogProbs, e.tokenizer.TokenString))
		}
		if !onToken(e.tokenizer.TokenString(nextID)) {
			hitStop = true
			break
		}
		metrics.CompletionTokens++
		logits, err = e.model.ForwardCached(cache, []int32{nextID}, *params.FlashAttention)
		if err != nil {
			return fmt.Errorf("decode forward: %w", err)
		}
	}
	metrics.DecodeDuration = time.Since(decodeStart)
	if hitStop {
		metrics.FinishReason = FinishReasonStop
	} else {
		metrics.FinishReason = FinishReasonLength
	}
	return nil
}
