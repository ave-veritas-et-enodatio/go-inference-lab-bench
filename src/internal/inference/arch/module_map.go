package arch

import (
	"fmt"
	"strings"
)

// TensorDims holds the shape and size of a weight tensor.
// Ne0 is the fastest dimension (input width); Ne1 is the output dimension (NHeads*HeadDim or NFF).
// Nbytes is the tensor's GPU memory footprint (quantized size, not logical element count).
// A special "_head_dim" entry per block type stores HeadDim in Ne0 for head-trim fraction computation.
type TensorDims struct {
	Ne0    int64
	Ne1    int64
	Nbytes int64
}

// TensorDimsMap maps block-type name → tensor short name → TensorDims.
// Block modules use their BlockName key (e.g. "full_attention", "recurrent_ssm").
// FFN modules use "ffn"; MoE FFN modules use "ffn_moe".
type TensorDimsMap map[string]map[string]TensorDims

// BuildTensorDimsMap builds a TensorDimsMap from resolved weights.
// dimLookup returns (ne0, ne1, nbytes, ok) for a given GGUF tensor name; it must be valid at call time.
// headDim is the number of elements per attention head (from resolved params); pass 0 if unknown.
func BuildTensorDimsMap(weights *ResolvedWeights, dimLookup func(string) (int64, int64, int64, bool), headDim int) TensorDimsMap {
	dims := make(TensorDimsMap)

	// Global tensor dims
	dims[ModuleGlobal] = make(map[string]TensorDims)
	for _, ggufName := range weights.Global {
		sn := strings.TrimSuffix(ggufName, ".weight")
		if _, ok := dims[ModuleGlobal][sn]; !ok {
			if ne0, ne1, nb, ok := dimLookup(ggufName); ok {
				dims[ModuleGlobal][sn] = TensorDims{Ne0: ne0, Ne1: ne1, Nbytes: nb}
			}
		}
	}

	for _, lw := range weights.Layers {
		// lw.Prefix is the canonical expanded per-layer prefix ("blk.5.") from [layers].prefix
		// in the arch TOML. It already carries the trailing dot, so it plugs straight into
		// TrimPrefix without reconstruction.
		ctx := lw.Prefix
		short := func(ggufName string) string {
			return strings.TrimSuffix(strings.TrimPrefix(ggufName, ctx), ".weight")
		}

		// Block type dims
		bt := lw.BlockName
		if dims[bt] == nil {
			dims[bt] = make(map[string]TensorDims)
			if headDim > 0 {
				dims[bt]["_head_dim"] = TensorDims{Ne0: int64(headDim)}
			}
		}
		for _, ggufName := range lw.Block {
			sn := short(ggufName)
			if _, ok := dims[bt][sn]; !ok {
				if ne0, ne1, nb, ok := dimLookup(ggufName); ok {
					dims[bt][sn] = TensorDims{Ne0: ne0, Ne1: ne1, Nbytes: nb}
				}
			}
		}
		for _, ggufName := range lw.Common {
			sn := short(ggufName)
			if _, ok := dims[bt][sn]; !ok {
				if ne0, ne1, nb, ok := dimLookup(ggufName); ok {
					dims[bt][sn] = TensorDims{Ne0: ne0, Ne1: ne1, Nbytes: nb}
				}
			}
		}

		// FFN type dims
		ffnKey := TypeFFN
		src := lw.FFN
		if len(lw.FFNAlt) > 0 {
			ffnKey = TypeFFNMoE
			src = lw.FFNAlt
		}
		if dims[ffnKey] == nil {
			dims[ffnKey] = make(map[string]TensorDims)
		}
		for _, ggufName := range src {
			sn := short(ggufName)
			if _, ok := dims[ffnKey][sn]; !ok {
				if ne0, ne1, nb, ok := dimLookup(ggufName); ok {
					dims[ffnKey][sn] = TensorDims{Ne0: ne0, Ne1: ne1, Nbytes: nb}
				}
			}
		}
	}

	return dims
}

// BuildModuleMap constructs a ModuleMap from resolved model weights.
// Uses the arch definition's weight structure directly — no tensor name pattern matching.
//
// Module layout:
//   - Module 0 "global": all global tensors (token_embd, output_norm, output, etc.)
//   - Module "block_L" per layer: pre-attention norm (attn_norm) + attention/SSM weights (Block)
//   - Module "ffn_L" per layer: pre-FFN norm (ffn_norm) + feed-forward weights
//     including MoE expert tensors (FFN + FFNAlt)
//
// Common weights are split by purpose: attn_norm belongs to block_L (it normalizes
// the block's input); all other common weights (ffn_norm) belong
// to ffn_L because they normalize the FFN's input. This ensures culling block_L does
// not silence the FFN's pre-normalization step.
func BuildModuleMap(weights *ResolvedWeights) *ModuleMap {
	mm := &ModuleMap{}

	// Module 0: global (no weight_context — names vary and have no common prefix)
	global := Module{ID: 0, Name: ModuleGlobal}
	for _, ggufName := range weights.Global {
		addCompact(&global, "", ggufName)
	}
	mm.Modules = append(mm.Modules, global)

	// Per-layer modules
	nextID := 1
	for _, lw := range weights.Layers {
		L := lw.Index
		// Module weight_context drops the trailing "." from lw.Prefix (e.g. "blk.5." → "blk.5")
		// because addCompact re-appends it when stripping.
		ctx := strings.TrimSuffix(lw.Prefix, ".")

		// Block module: pre-attention norm + attention/SSM/recurrent weights.
		block := Module{ID: nextID, Name: fmt.Sprintf(PrefixBlock+"%d", L), BlockName: lw.BlockName, WeightContext: ctx}
		nextID++
		// FFN module: pre-FFN norm + feed-forward weights including MoE expert tensors.
		ffn := Module{ID: nextID, Name: fmt.Sprintf(PrefixFFN+"%d", L), WeightContext: ctx}
		nextID++

		// Route common weights by purpose: attn_norm is the block's pre-norm;
		// ffn_norm (or any other non-attn_norm common weight) is the FFN's pre-norm.
		for logicalName, ggufName := range lw.Common {
			if logicalName == WeightAttnNorm {
				addCompact(&block, ctx, ggufName)
			} else {
				addCompact(&ffn, ctx, ggufName)
			}
		}
		for _, ggufName := range lw.Block {
			addCompact(&block, ctx, ggufName)
		}
		for _, ggufName := range lw.FFN {
			addCompact(&ffn, ctx, ggufName)
		}
		for _, ggufName := range lw.FFNAlt {
			addCompact(&ffn, ctx, ggufName)
		}

		mm.Modules = append(mm.Modules, block, ffn)
	}

	return mm
}

// addCompact strips the weight_context prefix from a full GGUF name and appends
// the short form to the appropriate list on the module:
//   - names ending in ".weight" → strip prefix and suffix, add to Weights
//   - all others → strip prefix only, add to Params
func addCompact(m *Module, ctx, ggufName string) {
	name := ggufName
	if ctx != "" {
		name = strings.TrimPrefix(name, ctx+".")
	}
	if strings.HasSuffix(name, ".weight") {
		m.Weights = append(m.Weights, strings.TrimSuffix(name, ".weight"))
	} else {
		m.Params = append(m.Params, name)
	}
}
