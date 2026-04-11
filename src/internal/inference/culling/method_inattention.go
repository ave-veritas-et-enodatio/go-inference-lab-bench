package culling

import (
	"inference-lab-bench/internal/inference/arch"
)

// InattentionPromptMeta holds prompt-derived spectral metrics from the generated metadata.
// Built from prompt token IDs via computeInattentionPromptMeta.
type InattentionPromptMeta struct {
	Dummy float32
}


func computeInattentionModelMeta(modelPath, outPath string, gpu bool) bool {
	return true
}


func computeInattentionPromptMeta(tokenIDs []int32, meta *CullingMeta) *InattentionPromptMeta {
	return &InattentionPromptMeta{Dummy: 0.0}
}


func applyInattentionCulling(mm *arch.ModuleMap, tokenIDs []int32, meta *CullingMeta) {
	_ = computeInattentionPromptMeta(tokenIDs, meta)
}
