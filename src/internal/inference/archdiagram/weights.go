package archdiagram

import (
	"inference-lab-bench/internal/inference/arch"
)

// ResolveWeightsForDiagram builds a ResolvedWeights from a parsed ArchDef and a known
// layer count, without requiring a GGUF file. Only used for example diagram generation
// (gen-arch-diagram), never for live model processing. Routing uses fallback param
// values (e.g. full_attn_interval=4); if the rule still cannot be evaluated, all
// layers use the first available block type.
func ResolveWeightsForDiagram(def *arch.ArchDef, nLayers int) *arch.ResolvedWeights {
	return arch.ResolveWeightsLenient(def, nLayers, diagramFallbackParams(def, nLayers))
}

// diagramFallbackParams builds ResolvedParams for diagram rendering without GGUF.
// Covers both rule-based routing (Qwen3.5: full_attn_interval) and pattern-based
// routing (Gemma4: swa_pattern as every-Nth-layer example).
func diagramFallbackParams(def *arch.ArchDef, nLayers int) *arch.ResolvedParams {
	rp := &arch.ResolvedParams{
		Ints:    map[string]int{},
		Floats:  map[string]float32{},
		Strings: map[string]string{},
		IntArr:  map[string][]int{},
	}

	if def.Example.FullAttnEvery > 0 {
		rp.Ints[arch.ParamFullAttnInterval] = def.Example.FullAttnEvery
	}

	// For pattern routing (fully recorded bool pattern), generate a pattern
	// at intervals
	if def.Example.AttnPatternTrueEvery > 0 || def.Example.AttnPatternFalseEvery > 0 {
		var interval int
		var baseValue int
		var intervalValue int
		if def.Example.AttnPatternTrueEvery > 0 {
			interval = def.Example.AttnPatternTrueEvery
			intervalValue = 1
			baseValue = 0
		} else {
			interval = def.Example.AttnPatternFalseEvery
			intervalValue = 0
			baseValue = 1
		}
		pattern := make([]int, nLayers)
		for i := range pattern {
			if (i+1)%interval == 0 {
				pattern[i] = intervalValue
			} else {
				pattern[i] = baseValue
			}
		}
		rp.IntArr[def.Layers.Routing.Pattern] = pattern
	}
	return rp
}

// resolveBlockForDiagram determines the block name for a layer in diagram context.
// Uses arch.ResolveBlockName which handles both pattern and rule routing.
func resolveBlockForDiagram(def *arch.ArchDef, layerIdx int, rp *arch.ResolvedParams) string {
	blockName, err := arch.ResolveBlockName(def, layerIdx, rp)
	if err != nil {
		blockName = def.Layers.Routing.IfTrue
		if blockName == "" {
			blockName = def.Layers.Routing.IfFalse
		}
	}
	return blockName
}

