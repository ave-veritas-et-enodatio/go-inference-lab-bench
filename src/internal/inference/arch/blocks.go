package arch

import (
	ggml "inference-lab-bench/internal/ggml"
)

// CaptureFlags is a bitfield controlling what intermediate values ForwardStateless collects.
type CaptureFlags uint32

const (
	// CaptureAttnWeights captures the post-softmax attention weight matrix per layer.
	CaptureAttnWeights CaptureFlags = 1 << iota
)

// ForwardCaptures is passed via GraphInputs.Captures to collect inference-time tensors.
// Nil ForwardCaptures (the default) adds zero overhead to the forward pass.
// Results are populated before ForwardStateless returns, after compute is complete.
type ForwardCaptures struct {
	Flags CaptureFlags
	// AttnWeights[il] is the flat post-softmax attention matrix for layer il.
	// Shape: [nHeads * nKV * nTokens]float32 (ggml layout: nKV fastest).
	// Nil for layers that do not use scaledDotProductAttention (SSM, MLA).
	AttnWeights [][]float32
	// NHeads and NTokens are set from the forward pass, needed to interpret AttnWeights.
	NHeads  int64
	NTokens int64

	// currentLayer is set by the blkFn closure before each block builder call
	// so scaledDotProductAttention can write into the right attnTensors slot.
	currentLayer int
	// attnTensors[il] holds the captured kq tensor for layer il; nil if layer has no SDPA.
	// Pre-allocated to nLayers; filled during graph construction; read after compute.
	attnTensors []ggml.Tensor
}

// SharedKVState passes in-graph K/V tensors from KV layers to non-KV layers.
// KV layers update these after computing K/V; non-KV layers read them for attention.
type SharedKVState struct {
	K map[string]ggml.Tensor // group → latest K
	V map[string]ggml.Tensor // group → latest V
}

// GraphInputs holds shared input tensors for the forward pass.
type GraphInputs struct {
	InpPos       ggml.Tensor
	InpMask      ggml.Tensor
	InpMaskSWA   ggml.Tensor      // sliding-window attention mask (nil if unused)
	NTokens      int64            // number of new tokens being processed
	NKV          int64            // total KV length (seqPos + nNew for cached, nTokens for stateless)
	SeqPos       int              // cache position (0 for stateless)
	Captures     *ForwardCaptures // nil = no capture
	SharedKV     *SharedKVState
	CurrentLayer int  // set by runLayers before calling the block builder
	FlashAttn    bool // use ggml_flash_attn_ext instead of explicit KQ+softmax+V matmul
}

// LayerCache holds per-layer cache tensors.
type LayerCache struct {
	Tensors     map[string]ggml.Tensor // cache name → tensor (e.g. "k", "v", "conv_state", "ssm_state")
	MaxSeqLen   int
	SharedGroup string // shared group name (empty = not shared)
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

	BuildCached(ctx *ggml.GraphContext, gf *ggml.Graph, input ggml.Tensor, weights map[string]ggml.Tensor,
		params *ResolvedParams, config map[string]any, inputs *GraphInputs,
		cache *LayerCache) ggml.Tensor
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
	ffnBuilders[FFNSymMoE] = &MoEBuilder{}
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

// GetBlockBuilders returns a copy of the block builder registry.
func GetBlockBuilders() map[string]BlockBuilder {
	m := make(map[string]BlockBuilder, len(blockBuilders))
	for k, v := range blockBuilders {
		m[k] = v
	}
	return m
}

// GetFFNBuilders returns a copy of the FFN builder registry.
func GetFFNBuilders() map[string]FFNBuilder {
	m := make(map[string]FFNBuilder, len(ffnBuilders))
	for k, v := range ffnBuilders {
		m[k] = v
	}
	return m
}

// ropeSections converts the IntArr rope_sections param to [4]int for ggml.RopeMulti.
func ropeSections(params *ResolvedParams) [4]int {
	arr := params.IntArr[ParamRoPESections]
	var s [4]int
	for i := 0; i < 4 && i < len(arr); i++ {
		s[i] = arr[i]
	}
	return s
}
