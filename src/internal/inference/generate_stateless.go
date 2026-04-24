package inference

import (
	"fmt"
	"time"
)

func (e *Engine) generateStateless(
	promptIDs []int32, maxTokens int, stopSet map[int32]bool,
	params GenerateParams,
	onToken func(string) bool, metrics *InferenceMetrics,
) error {
	tokenIDs := make([]int32, len(promptIDs))
	copy(tokenIDs, promptIDs)

	hitStop := false
	for range maxTokens {
		stepStart := time.Now()
		logits, err := e.model.ForwardStateless(tokenIDs, nil, *params.FlashAttention)
		if err != nil {
			return fmt.Errorf("forward pass: %w", err)
		}
		if metrics.CompletionTokens == 0 {
			metrics.PrefillDuration = time.Since(stepStart)
		} else {
			metrics.DecodeDuration += time.Since(stepStart)
		}
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
		tokenIDs = append(tokenIDs, nextID)
	}
	if hitStop {
		metrics.FinishReason = FinishReasonStop
	} else {
		metrics.FinishReason = FinishReasonLength
	}
	return nil
}
