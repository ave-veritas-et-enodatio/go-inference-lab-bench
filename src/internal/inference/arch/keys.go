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
	ConfigSharedKVGroup  = "shared_kv_group"
	ConfigRope           = "rope"
	ConfigVNorm          = "v_norm"
	ConfigQKNorm         = "qk_norm"
	ConfigOutputGate     = "output_gate"
	ConfigQHasGate       = "q_has_gate"
	ConfigSelfNormed     = "self_normed"
	ConfigNormW          = "norm_w"
	ConfigNormWParam     = "norm_w_param"
	ConfigActivation     = "activation"
	ConfigKQScale        = "kq_scale"
	ConfigConvActivation = "conv_activation"
	ConfigGateNorm       = "gate_norm"
	ConfigGateActivation = "gate_activation"
	// ConfigNonCausal, when "true" on an attention block, suppresses the
	// causal mask for that block. Used by ViT-style bidirectional encoders
	// where every token attends to every other token. Distinct from the
	// arch-level `non_causal` flag (ArchMeta.NonCausal) which controls
	// mask population for the decoder loop. The two coexist in a
	// multimodal model whose decoder is causal but vision tower is not.
	ConfigNonCausal = "non_causal"
	// ConfigQKVFused, when "true" on a vision attention block, declares that the
	// model ships a single fused attn_qkv weight + bias (Qwen3-VL) rather than
	// separate q/k/v tensors. BuildVisionGraph's splitFusedQKV slices the fused
	// tensors into contiguous q/k/v views before the generic AttentionBuilder
	// runs — the builder itself is unchanged. Absent / false → separate weights.
	ConfigQKVFused = "qkv_fused"
	// ConfigKQPrec selects the accumulation precision of the K·Q matmul in the
	// standard SDPA path. Unset / "" preserves the decoder default (force F32
	// accumulation via MulMatSetPrecF32); KQPrecNative skips the force and uses
	// the tensor's native precision, matching llama.cpp's clip vision encoder.
	ConfigKQPrec = "kq_prec"
)

// --------------- Config enum values ---------------
// Used in ConfigSchema valid-value lists and runtime comparisons.

const (
	RopeNeox     = "neox"
	RopeStandard = "standard"
	RopeMulti    = "multi"
	// RopeImrope is interleaved multi-section M-RoPE (GGML_ROPE_TYPE_IMROPE, 40),
	// used by the Qwen3-VL decoder. Same ggml_rope_multi call as RopeMulti but
	// mode=IMROPE, requiring a [4·n_tokens] InpPos buffer (layout [t|h|w|e]). For
	// text tokens (t=h=w=pos, e=0) IMROPE reduces to the NEOX result, so it is a
	// universal replacement for RopeMulti on imrope-declaring archs. Image spans
	// get 2D positions via the get_rope_index analogue (buildImropeDecoderPositions).
	RopeImrope = "imrope"
	// RopeAxial2D is a generic axial (2D) RoPE: the first half of each
	// head's head_dim is NeoX-rotated by a per-token X position, the second
	// half by a Y position. Standard ViT positional scheme. Requires the
	// graph to supply GraphInputs.PosX / PosY (decoder paths never do).
	RopeAxial2D = "axial2d"
	// RopeMRopeVision is multi-section vision M-RoPE (Qwen3-VL tower): both Q
	// and K are rotated by ggml_rope_multi with mode GGML_ROPE_TYPE_VISION over
	// the 4-channel [y,x,y,x] position buffer that VisionMRopePositions builds.
	// Requires the graph to supply GraphInputs.InpPosVision (decoder paths never
	// do). n_dims = head_dim/2, sections = {head_dim/4 ×4}, θ from rope_freq_base.
	RopeMRopeVision = "mrope_vision"
	// RopeNone skips RoPE entirely. ViT-style encoders don't rotate Q/K —
	// positional information comes from a separate position-embedding
	// tensor added at the patch-embedding stage instead.
	RopeNone = "none"
	// KQPrecNative is the opt-in value for ConfigKQPrec: skip the forced F32
	// accumulation on the K·Q matmul and use native precision. Empty / unset
	// means F32-on (the decoder default).
	KQPrecNative   = "native"
	NormRMS        = "rms"
	NormL2         = "l2"
	ActivationSiLU      = "silu"
	ActivationGELU      = "gelu"
	ActivationGELUQuick = "gelu_quick"
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
	ParamNFF              = "n_ff"
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
	WeightTokenEmbd         = "token_embd"
	WeightOutputNorm        = "output_norm"
	WeightOutput            = "output"
	WeightTokEmbdPerLayer   = "tok_embd_per_layer"   // stacked per-layer token embeddings (Gemma4)
	WeightPerLayerModelProj = "per_layer_model_proj" // model embedding → per-layer space projection (Gemma4)
	WeightPerLayerProjNorm  = "per_layer_proj_norm"  // per-layer projection norm (Gemma4)
)

// Common layer weights (shared across block types).
const (
	WeightAttnNorm         = "attn_norm"
	WeightFFNNorm          = "ffn_norm"
	WeightPostAttnNorm     = "post_attention_norm"
	WeightAttnPostNorm     = "attn_post_norm"     // post-attention norm (Gemma4)
	WeightFFNPostNorm      = "ffn_post_norm"      // post-FFN norm (Gemma4)
	WeightPEInpGate        = "pe_inp_gate"        // per-layer embedding input gate (Gemma4)
	WeightPEProj           = "pe_proj"            // per-layer embedding projection (Gemma4)
	WeightPEPostNorm       = "pe_post_norm"       // per-layer embedding post-projection norm (Gemma4)
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
	// Optional per-projection bias weights (separate-weight form). Present only
	// on towers whose linears carry biases (Qwen3-VL vision adds a bias on every
	// q/k/v/output projection). Absent (nil) on every decoder/Gemma path, so the
	// bias-add is skipped and the graph stays byte-identical.
	WeightAttnQBias      = "attn_q_bias"
	WeightAttnKBias      = "attn_k_bias"
	WeightAttnVBias      = "attn_v_bias"
	WeightAttnOutputBias = "attn_output_bias"
)

// MLA (Multi-head Latent Attention) weights — DeepSeek2 / GLM-4.
const (
	WeightAttnQA      = "attn_q_a"       // Q low-rank compress
	WeightAttnQANorm  = "attn_q_a_norm"  // Q compressed norm
	WeightAttnQB      = "attn_q_b"       // Q low-rank expand
	WeightAttnKVAMQA  = "attn_kv_a_mqa"  // KV joint compress (MQA)
	WeightAttnKVANorm = "attn_kv_a_norm" // KV compressed norm
	WeightAttnKB      = "attn_k_b"       // K low-rank expand (absorbed into Q)
	WeightAttnVB      = "attn_v_b"       // V decompression matrix
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
	// Optional per-projection bias weights for the plain-MLP FFN (up/down).
	// Present only on towers whose FFN linears carry biases (Qwen3-VL vision).
	// Absent (nil) on every decoder/Gemma path, so the bias-add is a no-op and
	// the graph stays byte-identical.
	WeightFFNUpBias   = "ffn_up_bias"
	WeightFFNDownBias = "ffn_down_bias"
)

// FFN MoE weights.
const (
	WeightFFNGateExps   = "ffn_gate_exps"
	WeightFFNUpExps     = "ffn_up_exps"
	WeightFFNDownExps   = "ffn_down_exps"
	WeightFFNGateShexp  = "ffn_gate_shexp"
	WeightFFNUpShexp    = "ffn_up_shexp"
	WeightFFNDownShexp  = "ffn_down_shexp"
	WeightFFNGateUpExps = "ffn_gate_up_exps" // fused gate+up (Gemma4 MoE)
	WeightFFNDownExpsS  = "ffn_down_exps_s"  // per-expert output scale (Gemma4 MoE)
)

// Vision-tower weights. Per-layer attention/FFN weights reuse the standard
// Weight* constants above (attn_q, ffn_gate, etc.); these are the vision-only
// global and projector logical names (keys in [vision.weights.global] /
// [projector.weights.global] TOML and the VisionTensors.Global / .Projector maps).
const (
	WeightVisionPatchEmbd   = "patch_embd"      // Conv2D patch embedder
	WeightVisionPatchEmbd1  = "patch_embd_1"    // second Conv2D kernel (Qwen3-VL dual-conv; summed with patch_embd)
	WeightVisionPatchBias   = "patch_embd_bias" // patch-embed bias (Qwen3-VL)
	WeightVisionPosEmbd     = "position_embd"   // additive position embedding (axial table or learned grid)
	WeightVisionPostLN      = "post_ln"         // post-encoder LayerNorm weight (Qwen3-VL)
	WeightVisionPostLNBias  = "post_ln_bias"    // post-encoder LayerNorm bias (Qwen3-VL)
	WeightVisionProj        = "proj"            // projector linear (vision embd → decoder space)
	WeightVisionProj2       = "proj2"           // projector second linear (Qwen3-VL MLP projector: mm.0→GELU→mm.2)
	WeightVisionProjBias    = "proj_bias"       // projector first-linear bias (Qwen3-VL)
	WeightVisionProj2Bias   = "proj2_bias"      // projector second-linear bias (Qwen3-VL)
	WeightVisionLN1         = "ln1"             // pre-attention norm (vision-only; decoder uses attn_norm)
	WeightVisionLN1Bias     = "ln1_bias"        // pre-attention LayerNorm bias (Qwen3-VL)
	WeightVisionLN2         = "ln2"             // pre-FFN norm (vision-only; decoder uses ffn_norm)
	WeightVisionLN2Bias     = "ln2_bias"        // pre-FFN LayerNorm bias (Qwen3-VL)
	WeightVisionAttnQKV     = "attn_qkv"        // fused QKV weight (Qwen3-VL; split into q/k/v views in-graph)
	WeightVisionAttnQKVBias = "attn_qkv_bias"   // fused QKV bias (Qwen3-VL; split alongside the weight)
)

// Vision norm-type selector values for [vision].norm_type.
const (
	VisionNormRMS       = "rms"       // RmsNorm, weight-only (Gemma 4; default when unset)
	VisionNormLayerNorm = "layernorm" // ggml_norm with learned weight + bias (Qwen3-VL)
)

// Projector type selector values for [projector].type.
const (
	ProjectorLinearPostNorm = "linear_post_norm" // single linear + post RmsNorm (Gemma 4)
	ProjectorMLP            = "mlp"              // mm.0 → GELU → mm.1, with biases (Qwen3-VL)
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
	MoEUpBias       = "up_bias"   // optional plain-MLP FFN up-projection bias
	MoEDownBias     = "down_bias" // optional plain-MLP FFN down-projection bias
	MoENorm         = "norm"
	MoEPreNorm2     = "pre_norm_2"
	MoEPostNorm1    = "post_norm_1"
	MoEPostNorm2    = "post_norm_2"
)

// MoEExpertWeights is the set of MoE weight names that are routed expert tensors
// (as opposed to router, shared expert, or norm weights). Used by the diagram
// renderer to partition MoE module weights into expert vs shared groups.
//
// Keyed on the stripped tensor names stored in Module.Weights (i.e. the
// WeightFFN*Exps constants with the `ffn_` prefix), not the short logical
// names from [ffn.weights] / [ffn_alt.weights] TOML keys.
var MoEExpertWeights = map[string]bool{
	WeightFFNGateExps:   true,
	WeightFFNUpExps:     true,
	WeightFFNGateUpExps: true,
	WeightFFNDownExps:   true,
	WeightFFNDownExpsS:  true,
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

// --------------- Vision metadata keys ---------------

const (
	// KeyVisionHasEncoder is our canonical vision-tower capability flag
	// (populated for safetensors via the stmap's config_key_present derived
	// op on config.json.vision_config). visionMetadataDeclared gates vision
	// setup on it.
	KeyVisionHasEncoder = "vision.has_encoder"
	// GGUFKeyClipHasVisionEncoder is the upstream mmproj-side alias for
	// KeyVisionHasEncoder (stored under the `clip.*` namespace as BOOL).
	GGUFKeyClipHasVisionEncoder = "clip.has_vision_encoder"
)
