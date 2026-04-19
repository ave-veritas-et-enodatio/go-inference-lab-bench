package culling

import (
	"strings"

	"inference-lab-bench/internal/inference/arch"
)

type RandomPromptMeta struct {
	Dummy uint32
}

func computeRandomModelMeta(modelPath string) *CullingMeta {
	_ = modelPath
	return &CullingMeta{Method: "random"}
}


func computeRandomPromptMeta(tokenIDs []int32, meta *CullingMeta) *RandomPromptMeta {
	return &RandomPromptMeta{Dummy: 0}
}

// applyRandomCulling applies a test pattern: every 4th block (per block type) and every
// 4th FFN module is fully culled via CulledIDs. Placeholder for real culling algorithms.
func applyRandomCulling(mm *arch.ModuleMap, tokenIDs []int32, meta *CullingMeta) {
	_ = meta
	_ = computeRandomPromptMeta(tokenIDs, meta)
	blockCount := make(map[string]int)
	ffnIdx := 0
	for _, m := range mm.Modules {
		switch {
		case strings.HasPrefix(m.Name, arch.PrefixBlock):
			idx := blockCount[m.BlockName]
			blockCount[m.BlockName] = idx + 1
			if idx%4 == 0 {
				mm.CulledIDs = append(mm.CulledIDs, m.ID)
			}
		case strings.HasPrefix(m.Name, arch.PrefixFFN):
			if ffnIdx%4 == 0 {
				mm.CulledIDs = append(mm.CulledIDs, m.ID)
			}
			ffnIdx++
		}
	}
}
