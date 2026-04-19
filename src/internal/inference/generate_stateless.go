package inference

import (
	"fmt"
	"strings"
	"time"

	"inference-lab-bench/internal/inference/arch"
	"inference-lab-bench/internal/inference/culling"
	log "inference-lab-bench/internal/log"
)

func (e *Engine) generateStateless(
	promptIDs []int32, maxTokens int, stopSet map[int32]bool,
	params GenerateParams, mask *arch.CullingMask,
	onToken func(string) bool, metrics *InferenceMetrics,
) error {
	tokenIDs := make([]int32, len(promptIDs))
	copy(tokenIDs, promptIDs)

	hitStop := false
	for range maxTokens {
		stepStart := time.Now()
		logits, engagement, err := e.model.ForwardStateless(tokenIDs, mask, nil, *params.FlashAttention)
		if err != nil {
			return fmt.Errorf("forward pass: %w", err)
		}
		if metrics.CompletionTokens == 0 {
			metrics.PrefillDuration = time.Since(stepStart)
			metrics.Engagement = engagement
			logEngagement(engagement)
		} else {
			metrics.DecodeDuration += time.Since(stepStart)
			metrics.Engagement = engagement // update with latest for live SVG
		}
		// Live engagement SVG update (every forward pass in stateless mode).
		if engagement != nil {
			culling.WriteEngagementDiag(e.diagDir, e.model.Def, e.model.CanonicalModuleMap, e.model.ModelPath, e.model.TensorDims, engagement, e.model.Def.Architecture.NonCausal, e.model.Def.Architecture.Generation)
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
		metrics.FinishReason = "stop"
	} else {
		metrics.FinishReason = "length"
	}
	return nil
}

// logEngagement logs per-layer engagement (1 - cosSim) at DEBUG level.
func logEngagement(ed *arch.EngagementData) {
	if ed == nil {
		return
	}
	var parts []string
	for il := range ed.BlockCosSim {
		bc := ed.BlockCosSim[il]
		fc := ed.FFNCosSim[il]
		parts = append(parts, fmt.Sprintf("L%d blk=%.4f ffn=%.4f", il, 1-bc, 1-fc))
	}
	log.Debug("engagement (1-cos): %s", strings.Join(parts, " | "))
}
