package culling

import (
	"fmt"

	log "inference-lab-bench/internal/log"
	"inference-lab-bench/internal/inference/arch"
	"inference-lab-bench/internal/util"
)

const (
	InattentionCullMethod = "inattention"
	RandomCullMethod      = "random"
)

// ComputeCullingMeta generates and writes a .cullmeta sidecar file for the given model and method.
// Each method handles its own serialization format. gpu=true uses GPU for generation.
func ComputeCullingMeta(modelPath string, method string, gpu bool) *string {
	metaPath := cullMetaPath(modelPath, method)

	switch method {
	case InattentionCullMethod:
		if !computeInattentionModelMeta(modelPath, metaPath, gpu) {
			return nil
		}
	case RandomCullMethod:
		util.WriteTOML(metaPath, computeRandomModelMeta(modelPath))
	default:
		log.Warn("unknown culling method: %s", method)
		return nil
	}

	return &metaPath
}

// ApplyCulling clones the canonical module map and applies the named culling method.
// Returns the culled ModuleMap (for diagnostic serialization) and compiled CullingMask (for inference).
// tokenIDs is the full encoded prompt (for prompt-aware culling methods).
// prompt is the user-facing prompt string (for diagnostic serialization only).
// meta is optional algorithm-specific metadata loaded from a sidecar file (nil = no metadata).
func ApplyCulling(canonical *arch.ModuleMap, method string, tokenIDs []int32, prompt string, meta *CullingMeta) (*arch.ModuleMap, *arch.CullingMask) {
	mm := canonical.Clone()
	mm.Method = method
	mm.Prompt = prompt

	if meta == nil {
		mm.CullLog = append(mm.CullLog, "[WRN] no culling metadata provided (culling not applied)")
	} else {
		mm.CullLog = append(mm.CullLog, fmt.Sprintf("culling_meta=%s", meta.Method))
		switch method {
		case InattentionCullMethod:
			applyInattentionCulling(mm, tokenIDs, meta)
		case RandomCullMethod:
			applyRandomCulling(mm, tokenIDs, meta)
		default:
			msg := fmt.Sprintf("[WRN] unknown culling method: %s (no culling applied)", method)
			log.Warn("unknown culling method: %s (no culling applied)", method)
			mm.CullLog = append(mm.CullLog, msg)
		}
	}

	mask := mm.Compile()
	return mm, mask
}
