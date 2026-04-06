package arch

import (
	ggml "inference-lab-bench/internal/inference/ggml"
)

// GraphInputs holds shared input tensors for the forward pass.
type GraphInputs struct {
	InpPos     ggml.Tensor
	InpMask    ggml.Tensor
	InpMaskSWA ggml.Tensor // sliding-window attention mask (nil if unused)
	NTokens    int64       // number of new tokens being processed
	NKV        int64       // total KV length (seqPos + nNew for cached, nTokens for stateless)
	SeqPos     int         // cache position (0 for stateless)
	SharedKV   *SharedKVState
	CurrentLayer int       // set by runLayers before calling the block builder
}

// SharedKVState passes in-graph K/V tensors from KV layers to non-KV layers.
// KV layers update these after computing K/V; non-KV layers read them for attention.
type SharedKVState struct {
	K map[string]ggml.Tensor // group → latest K (post-RoPE, pre-permute shape: [headDim, nKVHeads, nTokens])
	V map[string]ggml.Tensor // group → latest V (pre-permute shape)
}

// LayerCache holds per-layer cache tensors.
type LayerCache struct {
	Tensors     map[string]ggml.Tensor // cache name → tensor (e.g. "k", "v", "conv_state", "ssm_state")
	MaxSeqLen   int
	SharedGroup string // shared group name (empty = not shared)
}

// CacheWriteback describes a post-compute copy from a graph output to a persistent cache tensor.
type CacheWriteback struct {
	Src        ggml.Tensor
	Dst        ggml.Tensor
	NHeads     int
	HeadSrc    int // bytes per head in src
	HeadDst    int // bytes per head in dst
	HeadOffset int // byte offset within each head in dst
	HeadBytes  int // bytes to copy per head
}

// BuilderContract declares the expected weights, params, and config for a builder.
// Used by Validate() to catch mismatches at TOML load time.
type BuilderContract struct {
	RequiredWeights []string            // weight keys that must be present
	OptionalWeights []string            // weight keys that may be present
	RequiredParams  []string            // param names read from ResolvedParams
	ConfigSchema    map[string][]string // config key → valid values (nil slice = any value)
}

// BlockBuilder constructs ggml graph subgraphs for attention/SSM blocks.
type BlockBuilder interface {
	Contract() BuilderContract

	BuildStateless(ctx *ggml.GraphContext, input ggml.Tensor, weights map[string]ggml.Tensor,
		params *ResolvedParams, config map[string]any, inputs *GraphInputs,
		zeroFill *[]ggml.Tensor) ggml.Tensor

	BuildCached(ctx *ggml.GraphContext, input ggml.Tensor, weights map[string]ggml.Tensor,
		params *ResolvedParams, config map[string]any, inputs *GraphInputs,
		cache *LayerCache, writebacks *[]CacheWriteback) ggml.Tensor
}

// FFNBuilder constructs ggml graph subgraphs for feed-forward networks.
type FFNBuilder interface {
	Contract() BuilderContract

	BuildFFN(ctx *ggml.GraphContext, input ggml.Tensor, weights map[string]ggml.Tensor,
		params *ResolvedParams, config map[string]any) ggml.Tensor
}

// Builder registries
var blockBuilders = map[string]BlockBuilder{}
var ffnBuilders = map[string]FFNBuilder{}

func init() {
	blockBuilders["attention"] = &AttentionBuilder{}
	blockBuilders["mla_attention"] = &MLAAttentionBuilder{}
	blockBuilders["full_attention_gated"] = &FullAttentionGatedBuilder{}
	blockBuilders["gated_delta_net"] = &GatedDeltaNetBuilder{}
	ffnBuilders["swiglu"] = &SwiGLUBuilder{}
	ffnBuilders["geglu"] = &GeGLUBuilder{}
	ffnBuilders["moe"] = &MoEBuilder{}
}

// GetBlockBuilder returns a registered block builder by name.
func GetBlockBuilder(name string) (BlockBuilder, bool) {
	b, ok := blockBuilders[name]
	return b, ok
}

// GetFFNBuilder returns a registered FFN builder by name.
func GetFFNBuilder(name string) (FFNBuilder, bool) {
	b, ok := ffnBuilders[name]
	return b, ok
}

// ropeSections converts the IntArr rope_sections param to [4]int for ggml.RopeMulti.
func ropeSections(params *ResolvedParams) [4]int {
	arr := params.IntArr["rope_sections"]
	var s [4]int
	for i := 0; i < 4 && i < len(arr); i++ {
		s[i] = arr[i]
	}
	return s
}
