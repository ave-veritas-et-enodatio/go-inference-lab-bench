package arch

import (
	"fmt"
	"maps"
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
	Prefix    string            // expanded per-layer prefix (e.g. "blk.5.") — canonical, from [layers].prefix in TOML
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
	maps.Copy(rw.Global, def.Weights.Global)

	// Per-layer weights
	for i := range nLayers {
		prefix := ExpandPrefix(def.Layers.Prefix, i)

		// Determine block assignment via routing
		blockName, err := ResolveBlockName(def, i, params)
		if err != nil {
			return nil, fmt.Errorf("layer %d routing: %w", i, err)
		}

		if _, ok := def.Blocks[blockName]; !ok {
			return nil, fmt.Errorf("layer %d: block %q not defined", i, blockName)
		}

		rw.Layers[i] = fillLayerWeights(def, i, prefix, blockName)
	}

	return rw, nil
}

// fillLayerWeights builds one ResolvedLayerWeights by expanding weight suffixes
// through the per-layer prefix. Shared between strict (ResolveWeights) and
// lenient (ResolveWeightsLenient) paths.
func fillLayerWeights(def *ArchDef, i int, prefix, blockName string) ResolvedLayerWeights {
	block := def.Blocks[blockName]
	lw := ResolvedLayerWeights{
		Index:     i,
		BlockName: blockName,
		Prefix:    prefix,
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
	return lw
}

// ResolveWeightsLenient resolves weights using an explicit layer count and pre-built params.
// On routing error, falls back to the first available block name rather than returning an error.
// Used by diagram generation, which does not have a live GGUF file.
func ResolveWeightsLenient(def *ArchDef, nLayers int, params *ResolvedParams) *ResolvedWeights {
	rw := &ResolvedWeights{
		Global: make(map[string]string, len(def.Weights.Global)),
		Layers: make([]ResolvedLayerWeights, nLayers),
	}
	maps.Copy(rw.Global, def.Weights.Global)
	for i := range nLayers {
		prefix := ExpandPrefix(def.Layers.Prefix, i)
		blockName, err := ResolveBlockName(def, i, params)
		if err != nil {
			blockName = def.Layers.Routing.Uniform
			if blockName == "" {
				blockName = def.Layers.Routing.IfTrue
			}
			if blockName == "" {
				blockName = def.Layers.Routing.IfFalse
			}
		}
		rw.Layers[i] = fillLayerWeights(def, i, prefix, blockName)
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

// ExpandPrefix substitutes the layer-index sigil in a per-layer prefix template.
func ExpandPrefix(tmpl string, layerIdx int) string {
	return strings.ReplaceAll(tmpl, BuiltinLayerIdxRef, strconv.Itoa(layerIdx))
}

// ResolveBlockName picks the block type a given layer uses, per the routing config.
func ResolveBlockName(def *ArchDef, layerIdx int, params *ResolvedParams) (string, error) {
	r := &def.Layers.Routing

	// Uniform routing: all layers use the same block type
	if r.Uniform != "" {
		return r.Uniform, nil
	}

	// Pattern-based routing: index into IntArr by layer
	if r.Pattern != "" {
		arr, ok := params.IntArr[r.Pattern]
		if !ok {
			return "", fmt.Errorf("pattern param %q not found", r.Pattern)
		}
		if layerIdx >= len(arr) {
			return "", fmt.Errorf("pattern param %q has %d elements, need index %d", r.Pattern, len(arr), layerIdx)
		}
		if arr[layerIdx] != 0 {
			return r.IfTrue, nil
		}
		return r.IfFalse, nil
	}

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
