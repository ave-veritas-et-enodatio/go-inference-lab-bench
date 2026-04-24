package arch

// Canonical string constants for map keys, module identifiers,
// palette prefixes, config keys/values, and generation strategies.
// Every string used as a key in ResolvedParams, weight maps, module
// names, cache tensors, config maps, or diagram palettes should be
// defined here — never scattered as bare literals.
//
// TOML key-path segment constants (for validation error paths) live
// in arch.go alongside the Validate function that uses them.

// --------------- Generation strategy values ---------------

const (
	GenerationDiffusion = "diffusion"
)

// --------------- Config key names ---------------
// Used in [blocks.*.config], [ffn.config], and [ffn_alt.config] TOML sections.

const (
	ConfigSharedKVGroup = "shared_kv_group"
	ConfigRope          = "rope"
	ConfigVNorm         = "v_norm"
	ConfigQKNorm        = "qk_norm"
	ConfigOutputGate    = "output_gate"
	ConfigQHasGate      = "q_has_gate"
	ConfigSelfNormed    = "self_normed"
	ConfigNormW         = "norm_w"
	ConfigNormWParam    = "norm_w_param"
	ConfigActivation    = "activation"
	ConfigKQScale       = "kq_scale"
	ConfigConvActivation = "conv_activation"
	ConfigGateNorm      = "gate_norm"
	ConfigGateActivation = "gate_activation"
)

// --------------- Config enum values ---------------
// Used in ConfigSchema valid-value lists and runtime comparisons.

const (
	RopeNeox     = "neox"
	RopeStandard = "standard"
	RopeMulti    = "multi"
	NormRMS      = "rms"
	NormL2       = "l2"
	ActivationSiLU = "silu"
	ActivationGELU = "gelu"
	GateSigmoid    = "sigmoid"
)

// --------------- Engine builtins ---------------

const (
	BuiltinLayerIdx    = "layer_idx"         // @{layer_idx} in DSL expressions
	BuiltinLayerIdxRef = "@{layer_idx}"      // full sigil form used in prefix templates
	BuiltinLayerPrefix = "blk.@{layer_idx}." // default GGUF per-layer prefix (stmap default)
	CacheDimMaxSeqLen  = "max_seq_len"       // special token in cache dim expressions
)

// --------------- Parameter keys (ResolvedParams) ---------------

// Integer parameter keys.
const (
	ParamNLayers          = "n_layers"
	ParamNEmbd            = "n_embd"
	ParamNVocab           = "n_vocab"
	ParamNHeads           = "n_heads"
	ParamNKVHeads         = "n_kv_heads"
	ParamHeadDim          = "head_dim"
	ParamHeadVDim         = "head_v_dim"
	ParamFullAttnInterval = "full_attn_interval"
	ParamNKVSharedLayers  = "n_kv_shared_layers"
	ParamSlidingWindow    = "sliding_window"
	ParamRoPENRot         = "rope_n_rot"
	ParamSSMDTRank        = "ssm_dt_rank"
	ParamSSMDState        = "ssm_d_state"
	ParamSSMDConv         = "ssm_d_conv"
	ParamSSMDInner        = "ssm_d_inner"
	ParamSSMNGroup        = "ssm_n_group"
	ParamConvChannels     = "conv_channels"
	ParamNExpert          = "n_expert"
	ParamNExpertUsed      = "n_expert_used"
	ParamNEmbdPerLayer    = "n_embd_per_layer"
)

// Float parameter keys.
const (
	ParamRMSEps            = "rms_eps"
	ParamRoPEFreqBase      = "rope_freq_base"
	ParamLogitSoftcap      = "logit_softcapping"
	ParamExpertWeightScale = "expert_weights_scale"
)

// MLA integer parameter keys.
const (
	ParamKVLoraRank  = "kv_lora_rank"   // KV low-rank compression dimension (DeepSeek2 MLA)
	ParamHeadKDimMLA = "head_k_dim_mla" // total K cache dim per entry = kv_lora_rank + rope_n_rot (MLA)
)

// Array parameter keys (IntArr).
const (
	ParamRoPESections = "rope_sections"
)

// --------------- Cache tensor keys ---------------

// Must match key names in [blocks.*.cache] arch TOML sections.
const (
	CacheK         = "k"
	CacheV         = "v"
	CacheConvState = "conv_state"
	CacheSSMState  = "ssm_state"
)

// --------------- Weight / tensor keys ---------------

// Global weights.
const (
	WeightTokenEmbd          = "token_embd"
	WeightOutputNorm         = "output_norm"
	WeightOutput             = "output"
	WeightTokEmbdPerLayer    = "tok_embd_per_layer"    // stacked per-layer token embeddings (Gemma4)
	WeightPerLayerModelProj  = "per_layer_model_proj"  // model embedding → per-layer space projection (Gemma4)
	WeightPerLayerProjNorm   = "per_layer_proj_norm"   // per-layer projection norm (Gemma4)
)

// Common layer weights (shared across block types).
const (
	WeightAttnNorm      = "attn_norm"
	WeightFFNNorm       = "ffn_norm"
	WeightPostAttnNorm  = "post_attention_norm"
	WeightAttnPostNorm  = "attn_post_norm"      // post-attention norm (Gemma4)
	WeightFFNPostNorm   = "ffn_post_norm"       // post-FFN norm (Gemma4)
	WeightPEInpGate     = "pe_inp_gate"         // per-layer embedding input gate (Gemma4)
	WeightPEProj        = "pe_proj"             // per-layer embedding projection (Gemma4)
	WeightPEPostNorm    = "pe_post_norm"        // per-layer embedding post-projection norm (Gemma4)
	WeightLayerOutputScale = "layer_output_scale" // per-layer residual scale (Gemma4)
)

// Block attention weights.
const (
	WeightAttnQ      = "attn_q"
	WeightAttnK      = "attn_k"
	WeightAttnV      = "attn_v"
	WeightAttnOutput = "attn_output"
	WeightAttnQNorm  = "attn_q_norm"
	WeightAttnKNorm  = "attn_k_norm"
	WeightRoPEFreqs  = "rope_freqs"
	WeightRoPE       = "rope" // synthetic — not stored; diagram placeholder
)

// MLA (Multi-head Latent Attention) weights — DeepSeek2 / GLM-4.
const (
	WeightAttnQA     = "attn_q_a"       // Q low-rank compress
	WeightAttnQANorm = "attn_q_a_norm"  // Q compressed norm
	WeightAttnQB     = "attn_q_b"       // Q low-rank expand
	WeightAttnKVAMQA = "attn_kv_a_mqa"  // KV joint compress (MQA)
	WeightAttnKVANorm = "attn_kv_a_norm" // KV compressed norm
	WeightAttnKB     = "attn_k_b"       // K low-rank expand (absorbed into Q)
	WeightAttnVB     = "attn_v_b"       // V decompression matrix
)

// Block SSM / delta-net weights.
const (
	WeightSSMNorm   = "ssm_norm"
	WeightAttnQKV   = "attn_qkv"
	WeightAttnGate  = "attn_gate"
	WeightSSMA      = "ssm_a"
	WeightSSMAlpha  = "ssm_alpha"
	WeightSSMBeta   = "ssm_beta"
	WeightSSMConv1D = "ssm_conv1d"
	WeightSSMDTBias = "ssm_dt_bias"
	WeightSSMOut    = "ssm_out"
)

// FFN dense weights.
const (
	WeightFFNGate = "ffn_gate"
	WeightFFNUp   = "ffn_up"
	WeightFFNDown = "ffn_down"
)

// FFN MoE weights.
const (
	WeightFFNGateExps  = "ffn_gate_exps"
	WeightFFNUpExps    = "ffn_up_exps"
	WeightFFNDownExps  = "ffn_down_exps"
	WeightFFNGateShexp = "ffn_gate_shexp"
	WeightFFNUpShexp   = "ffn_up_shexp"
	WeightFFNDownShexp = "ffn_down_shexp"
	WeightFFNGateUpExps = "ffn_gate_up_exps" // fused gate+up (Gemma4 MoE)
	WeightFFNDownExpsS  = "ffn_down_exps_s"  // per-expert output scale (Gemma4 MoE)
)

// MoE short-name weights (keys in [ffn.weights] / [ffn_alt.weights] TOML sections).
// These are the un-prefixed names used in Contract() and BuildFFN().
const (
	MoEGateInp      = "gate_inp"
	MoEGateInpS     = "gate_inp_s"
	MoEGateExps     = "gate_exps"
	MoEUpExps       = "up_exps"
	MoEGateUpExps   = "gate_up_exps"
	MoEDownExps     = "down_exps"
	MoEDownExpsS    = "down_exps_s"
	MoEExpProbsB    = "exp_probs_b"
	MoEGateInpShexp = "gate_inp_shexp"
	MoEGateShexp    = "gate_shexp"
	MoEUpShexp      = "up_shexp"
	MoEDownShexp    = "down_shexp"
	MoEGate         = "gate"
	MoEUp           = "up"
	MoEDown         = "down"
	MoENorm         = "norm"
	MoEPreNorm2     = "pre_norm_2"
	MoEPostNorm1    = "post_norm_1"
	MoEPostNorm2    = "post_norm_2"
)

// MoEExpertWeights is the set of MoE weight names that are routed expert tensors
// (as opposed to router, shared expert, or norm weights). Used by the diagram
// renderer to partition MoE module weights into expert vs shared groups.
var MoEExpertWeights = map[string]bool{
	MoEGateExps:   true,
	MoEUpExps:     true,
	MoEGateUpExps: true,
	MoEDownExps:   true,
	MoEDownExpsS:  true,
}


// --------------- Module name constants ---------------

// Name prefixes for ModuleMap entries.
const (
	PrefixBlock = "block_" // block module: block_0, block_1, …
	PrefixFFN   = "ffn_"   // FFN module: ffn_0, ffn_1, …
)

// Module name for the global (embedding + output) module.
const ModuleGlobal = "global"

// --------------- Block / module type identifiers ---------------
// Used as palette key prefixes, SVG gradient IDs, TensorDimsMap keys.

const (
	TypeFullAttention = "full_attention"
	TypeSWA           = "swa"
	TypeRecurrent     = "recurrent"
	TypeFFN           = "ffn"
	TypeFFNMoE        = "ffn_moe"
	TypeNorm          = "norm"
)

// FFN symbol keys for diagram rendering.
const (
	FFNSymDense = "dense"
	FFNSymMoE   = "moe"
)

// --------------- Tokenizer GGUF metadata keys ---------------

const (
	GGUFKeyTokenizerTokens       = "tokenizer.ggml.tokens"
	GGUFKeyTokenizerMerges       = "tokenizer.ggml.merges"
	GGUFKeyTokenizerTokenType    = "tokenizer.ggml.token_type"
	GGUFKeyTokenizerModel        = "tokenizer.ggml.model"
	GGUFKeyTokenizerMaskTokenID  = "tokenizer.ggml.mask_token_id"
	GGUFKeyTokenizerChatTemplate = "tokenizer.chat_template"
)

// Token type constants — values stored in the tokenizer.ggml.token_type array.
const (
	GGUFTokenTypeControl     = 3 // CONTROL tokens (special, e.g. <|im_start|>)
	GGUFTokenTypeUserDefined = 4 // USER_DEFINED tokens
)
