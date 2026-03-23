package arch

import (
	"fmt"
	"strconv"
	"strings"
)

// ResolvedWeights holds all weight tensor names organized by role.
type ResolvedWeights struct {
	Global map[string]string            // logical name → GGUF tensor name
	Layers []ResolvedLayerWeights
}

type ResolvedLayerWeights struct {
	Index     int
	BlockName string            // which block this layer uses (e.g. "recurrent_ssm", "full_attention")
	Common    map[string]string // logical name → GGUF tensor name
	Block     map[string]string // block-specific weights
	FFN       map[string]string // FFN weights
	FFNAlt    map[string]string // alternative FFN weights (for layers that use a different FFN builder)
}

// ResolveWeights expands all weight name templates for a given architecture.
func ResolveWeights(def *ArchDef, params *ResolvedParams) (*ResolvedWeights, error) {
	nLayers, err := resolveCountExpr(def.Layers.Count, params)
	if err != nil {
		return nil, fmt.Errorf("resolving layer count: %w", err)
	}

	rw := &ResolvedWeights{
		Global: make(map[string]string, len(def.Weights.Global)),
		Layers: make([]ResolvedLayerWeights, nLayers),
	}

	// Global weights
	for logicalName, tensorName := range def.Weights.Global {
		rw.Global[logicalName] = tensorName
	}

	// Per-layer weights
	for i := range nLayers {
		prefix := expandPrefix(def.Layers.Prefix, i)

		// Determine block assignment via routing
		blockName, err := resolveBlockName(def, i, params)
		if err != nil {
			return nil, fmt.Errorf("layer %d routing: %w", i, err)
		}

		block, ok := def.Blocks[blockName]
		if !ok {
			return nil, fmt.Errorf("layer %d: block %q not defined", i, blockName)
		}

		lw := ResolvedLayerWeights{
			Index:     i,
			BlockName: blockName,
			Common:    make(map[string]string, len(def.Layers.CommonWeights)),
			Block:     make(map[string]string, len(block.Weights)),
			FFN:       make(map[string]string, len(def.FFN.Weights)),
		}

		// Common weights
		for logicalName, suffix := range def.Layers.CommonWeights {
			lw.Common[logicalName] = prefix + suffix
		}

		// Block-specific weights
		for logicalName, suffix := range block.Weights {
			lw.Block[logicalName] = prefix + suffix
		}

		// FFN weights
		for logicalName, suffix := range def.FFN.Weights {
			lw.FFN[logicalName] = prefix + suffix
		}

		// FFN alt weights (optional, for per-layer FFN routing)
		if def.FFNAlt != nil {
			lw.FFNAlt = make(map[string]string, len(def.FFNAlt.Weights))
			for logicalName, suffix := range def.FFNAlt.Weights {
				lw.FFNAlt[logicalName] = prefix + suffix
			}
		}

		rw.Layers[i] = lw
	}

	return rw, nil
}

// ResolveWeightsFromDef builds a ResolvedWeights from a parsed ArchDef and a known
// layer count, without requiring a GGUF file. Only used for example diagram generation
// (gen-arch-diagram), never for live model processing. Routing uses fallback param
// values (e.g. full_attn_interval=4); if the rule still cannot be evaluated, all
// layers use the if_true block type.
func ResolveWeightsFromDef(def *ArchDef, nLayers int) *ResolvedWeights {
	rw := &ResolvedWeights{
		Global: make(map[string]string, len(def.Weights.Global)),
		Layers: make([]ResolvedLayerWeights, nLayers),
	}

	for logicalName, tensorName := range def.Weights.Global {
		rw.Global[logicalName] = tensorName
	}

	// this is an example value to allow mixed attention block models to show both block types
	// this parameter normally comes from the .gguf
	fallbackParams := &ResolvedParams{
		Ints:   map[string]int{"full_attn_interval": 4},
		Floats: make(map[string]float32),
	}

	for i := range nLayers {
		prefix := expandPrefix(def.Layers.Prefix, i)

		blockName, err := resolveBlockName(def, i, fallbackParams)
		if err != nil {
			blockName = def.Layers.Routing.IfTrue
			if blockName == "" {
				blockName = def.Layers.Routing.IfFalse
			}
		}

		block := def.Blocks[blockName]

		lw := ResolvedLayerWeights{
			Index:     i,
			BlockName: blockName,
			Common:    make(map[string]string, len(def.Layers.CommonWeights)),
			Block:     make(map[string]string, len(block.Weights)),
			FFN:       make(map[string]string, len(def.FFN.Weights)),
		}

		for logicalName, suffix := range def.Layers.CommonWeights {
			lw.Common[logicalName] = prefix + suffix
		}
		for logicalName, suffix := range block.Weights {
			lw.Block[logicalName] = prefix + suffix
		}
		for logicalName, suffix := range def.FFN.Weights {
			lw.FFN[logicalName] = prefix + suffix
		}
		if def.FFNAlt != nil {
			lw.FFNAlt = make(map[string]string, len(def.FFNAlt.Weights))
			for logicalName, suffix := range def.FFNAlt.Weights {
				lw.FFNAlt[logicalName] = prefix + suffix
			}
		}

		rw.Layers[i] = lw
	}

	return rw
}

func resolveCountExpr(expr string, params *ResolvedParams) (int, error) {
	// Simple case: just a param name
	if v, ok := params.Ints[expr]; ok {
		return v, nil
	}
	// Try parsing as literal
	if v, err := strconv.Atoi(expr); err == nil {
		return v, nil
	}
	return 0, fmt.Errorf("cannot resolve count %q", expr)
}

func expandPrefix(tmpl string, layerIdx int) string {
	return strings.ReplaceAll(tmpl, "@{layer_idx}", strconv.Itoa(layerIdx))
}

func resolveBlockName(def *ArchDef, layerIdx int, params *ResolvedParams) (string, error) {
	r := &def.Layers.Routing
	if r.Rule == "" {
		// No routing — all layers use if_true (or if_false if if_true is empty)
		if r.IfTrue != "" {
			return r.IfTrue, nil
		}
		return r.IfFalse, nil
	}

	result, err := EvalRoutingRule(r.Rule, layerIdx, params)
	if err != nil {
		return "", err
	}
	if result {
		return r.IfTrue, nil
	}
	return r.IfFalse, nil
}
