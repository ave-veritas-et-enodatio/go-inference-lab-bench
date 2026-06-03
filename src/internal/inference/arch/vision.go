package arch

import (
	"fmt"
	"math"

	ggml "inference-lab-bench/internal/ggml"
)

// VisionResolved captures the vision-tower's per-layer and global weight
// tensor names, plus the projector's single linear weight. It is the
// vision-side analogue of ResolvedWeights / ResolvedLayerWeights, kept
// separate because the vision tower's encoder forward graph is procedural
// (see BuildVisionGraph) rather than block-builder-dispatched, so the
// flatter, per-layer-map shape is cleaner than the
// Common/Block/FFN three-way split the decoder uses.
type VisionResolved struct {
	// NLayers is the resolved vision-tower depth.
	NLayers int
	// Global maps vision-tower global short names (e.g. patch_embd) to
	// their GGUF tensor names (e.g. v.patch_embd.weight).
	Global map[string]string
	// Layers[il] holds one map per layer index: logical short name
	// (e.g. attn_q) → expanded GGUF tensor name (e.g. v.blk.5.attn_q.weight).
	Layers []map[string]string
	// Projector maps projector logical short names (e.g. proj) to their
	// GGUF tensor names (e.g. mm.input_projection.weight).
	Projector map[string]string
}

// ResolveVisionWeights builds a VisionResolved from def.Vision and
// def.Projector by template-expanding per-layer weight suffixes through
// the vision-tower's per-layer prefix. Returns (nil, nil) when the model
// has no [vision] section (unimodal arch).
func ResolveVisionWeights(def *ArchDef, params *ResolvedParams) (*VisionResolved, error) {
	if def.Vision == nil {
		return nil, nil
	}
	nLayersExpr := def.Vision.Layers.Count
	if nLayersExpr == "" {
		return nil, fmt.Errorf("vision.layers.count is required")
	}
	nLayers, err := resolveCountExpr(nLayersExpr, params)
	if err != nil {
		return nil, fmt.Errorf("resolving vision layer count: %w", err)
	}
	prefixTemplate := def.Vision.Layers.Prefix
	if prefixTemplate == "" {
		return nil, fmt.Errorf("vision.layers.prefix is required")
	}

	vr := &VisionResolved{
		NLayers: nLayers,
		Global:  make(map[string]string, len(def.Vision.Weights.Global)),
		Layers:  make([]map[string]string, nLayers),
	}
	for k, v := range def.Vision.Weights.Global {
		vr.Global[k] = v
	}
	for il := range nLayers {
		prefix := ExpandPrefix(prefixTemplate, il)
		layerMap := make(map[string]string, len(def.Vision.Layers.CommonWeights))
		for logicalName, suffix := range def.Vision.Layers.CommonWeights {
			layerMap[logicalName] = prefix + suffix
		}
		vr.Layers[il] = layerMap
	}
	if def.Projector != nil {
		vr.Projector = make(map[string]string, len(def.Projector.Weights))
		for k, v := range def.Projector.Weights {
			vr.Projector[k] = v
		}
	}
	return vr, nil
}

// VisionTensors carries the actual loaded ggml.Tensor handles for the
// vision tower and projector, indexed by the logical short names from
// VisionResolved. Populated by the model loader after weight upload.
// All tensors live in the same WeightStore as the decoder.
//
// Clamps is the optional per-arch `Gemma4ClippableLinear`-style clamp
// info (BF16 scalars paired with each linear weight). nil when the
// model has no clamp scalars or the arch doesn't use clipped linears.
type VisionTensors struct {
	Global    map[string]ggml.Tensor
	Layers    []map[string]ggml.Tensor
	Projector map[string]ggml.Tensor

	Clamps *VisionClamps
}

// BuildVisionTensors looks up every name in vr inside index (the loaded
// tensor handle map) and assembles a VisionTensors. Missing tensors
// become NilTensor — the encoder forward graph is expected to check
// IsNil on optional ones.
func BuildVisionTensors(vr *VisionResolved, index map[string]ggml.Tensor) *VisionTensors {
	if vr == nil {
		return nil
	}
	out := &VisionTensors{
		Global: make(map[string]ggml.Tensor, len(vr.Global)),
		Layers: make([]map[string]ggml.Tensor, len(vr.Layers)),
	}
	for k, name := range vr.Global {
		if t, ok := index[name]; ok {
			out.Global[k] = t
		} else {
			out.Global[k] = ggml.NilTensor()
		}
	}
	for il, lm := range vr.Layers {
		layerHandles := make(map[string]ggml.Tensor, len(lm))
		for k, name := range lm {
			if t, ok := index[name]; ok {
				layerHandles[k] = t
			} else {
				layerHandles[k] = ggml.NilTensor()
			}
		}
		out.Layers[il] = layerHandles
	}
	if vr.Projector != nil {
		out.Projector = make(map[string]ggml.Tensor, len(vr.Projector))
		for k, name := range vr.Projector {
			if t, ok := index[name]; ok {
				out.Projector[k] = t
			} else {
				out.Projector[k] = ggml.NilTensor()
			}
		}
	}
	return out
}

// VisionParams captures the small set of scalar params the encoder graph
// needs at build time. The model-metadata-derived fields (n_layers,
// n_embd, etc.) come through the standard ResolveParams pipeline against
// `vision.*` GGUF keys; the per-arch constants (rope_theta, n_merge,
// image-token bounds, patch_size) come from `[vision]` fields in the
// arch.toml. Together they keep all vision-tower configuration data-
// driven — no Go-side branching on architecture name or hardcoded
// upstream-clip.cpp constants.
type VisionParams struct {
	// From model file metadata (`vision.*` keys via [vision.params]):
	NLayers int
	NHeads  int
	NEmbd   int     // hidden dim of the vision tower (768 for Gemma 4 E4B)
	NFF     int     // FFN intermediate dim (3072 for Gemma 4 E4B)
	HeadDim int     // per-head dim (64 for Gemma 4 E4B)
	RMSEps  float32 // epsilon for all RmsNorm calls

	// From [vision] fields in arch.toml. PreprocConfig is embedded so the
	// preprocess-relevant subset (PatchSize, NMerge, ImageMinTokens,
	// ImageMaxTokens) is shared with PreprocessImage — one field set, one
	// resolver. RopeTheta is encoder-graph-only with no preprocess analog,
	// so it stays at the top level.
	PreprocConfig
	RopeTheta float32 // RoPE theta for 2D split rope (100.0 for Gemma 4)

	// NormType selects the encoder dispatch frame's per-layer norm:
	// VisionNormRMS (default) or VisionNormLayerNorm. ProjectorType selects
	// the projector path: ProjectorLinearPostNorm (default) or ProjectorMLP.
	// Both drive data-driven tower I/O branching in BuildVisionGraph so neither
	// vision arch needs model-specific Go.
	NormType      string
	ProjectorType string
}

// ResolveVisionParams runs resolveParam for each [vision.params] entry
// against the model reader and returns a flat VisionParams struct.
// Defaults are filled in for fields the model doesn't carry (rope_theta
// and pooling_kernel are Gemma 4 architectural constants not in the GGUF
// metadata block; future stmaps can override via [gguf_metadata]).
func ResolveVisionParams(def *ArchDef, reader ModelReader) (*VisionParams, error) {
	if def.Vision == nil {
		return nil, nil
	}
	rp := &ResolvedParams{
		Ints:   make(map[string]int),
		Floats: make(map[string]float32),
	}
	for name, ggufKey := range def.Vision.Params.Keys {
		if err := resolveParam(name, ggufKey, reader, rp); err != nil {
			return nil, fmt.Errorf("vision param %q: %w", name, err)
		}
	}
	// PreprocConfigFromArchDef is the single source for the four preprocess-
	// relevant [vision] fields (PatchSize, NMerge, ImageMinTokens,
	// ImageMaxTokens) and validates them; only RopeTheta needs a separate
	// check here since it's encoder-graph-only.
	ppCfg, err := PreprocConfigFromArchDef(def)
	if err != nil {
		return nil, err
	}
	normType := def.Vision.NormType
	if normType == "" {
		normType = VisionNormRMS // Gemma 4 default — keeps existing tower byte-identical.
	}
	projectorType := ProjectorLinearPostNorm
	if def.Projector != nil && def.Projector.Type != "" {
		projectorType = def.Projector.Type
	}
	vp := &VisionParams{
		NLayers:       rp.Ints[ParamNLayers],
		NHeads:        rp.Ints[ParamNHeads],
		NEmbd:         rp.Ints[ParamNEmbd],
		NFF:           rp.Ints[ParamNFF],
		HeadDim:       rp.Ints[ParamHeadDim],
		RMSEps:        rp.Floats[ParamRMSEps],
		PreprocConfig: ppCfg,
		RopeTheta:     def.Vision.RopeTheta,
		NormType:      normType,
		ProjectorType: projectorType,
	}
	// mmproj GGUF metadata exposes n_embd + n_heads but not head_dim
	// (the upstream `clip.vision.*` namespace lacks `attention.key_length`),
	// so derive it when missing. Safe because n_embd is guaranteed
	// divisible by n_heads for any well-formed transformer config.
	if vp.HeadDim == 0 && vp.NHeads > 0 && vp.NEmbd > 0 {
		vp.HeadDim = vp.NEmbd / vp.NHeads
	}
	if vp.NEmbd == 0 || vp.NHeads == 0 || vp.HeadDim == 0 || vp.NLayers == 0 {
		return nil, fmt.Errorf("vision params incomplete (from model metadata): %+v", vp)
	}
	if vp.RopeTheta == 0 {
		return nil, fmt.Errorf("vision rope_theta missing — must be declared in .arch.toml [vision]")
	}
	if vp.NEmbd != vp.NHeads*vp.HeadDim {
		return nil, fmt.Errorf("vision n_embd (%d) != n_heads (%d) * head_dim (%d)",
			vp.NEmbd, vp.NHeads, vp.HeadDim)
	}
	if vp.RMSEps <= 0 {
		// Some configs nest the eps under vision_config.rms_norm_eps;
		// fall back to a sensible default rather than failing — vision
		// epsilons are not safety-critical the way decoder ones are.
		vp.RMSEps = 1e-6
	}
	return vp, nil
}

// VisionInputs is the per-request side of the encoder graph: the raw
// image, plus per-patch position indices into the 2D position embedding
// table. The caller (Phase 7 splice integration) is responsible for
// preprocessing the input image to the right resolution, computing the
// per-patch (x, y) coordinates, and feeding them through the graph
// scheduler.
type VisionInputs struct {
	// InpRaw is the F32 image tensor with ne = [W, H, 3, n_batches],
	// values pre-scaled to roughly [0, 1] (the encoder graph applies the
	// final * 2 - 1 rescale internally).
	InpRaw ggml.Tensor
	// PosX, PosY are I32 [n_patches] vectors with each patch's column
	// and row index into the position embedding table. Consumed by the Gemma
	// axial-2D RoPE path and the axial position-embedding lookup.
	PosX, PosY ggml.Tensor
	// InpPosVision is the I32 [4*n_patches] M-RoPE position buffer (channel-
	// major [y,x,y,x], merge-grouped) for the Qwen3-VL tower's rope=mrope_vision
	// blocks. NilTensor for the Gemma axial path. Built by VisionMRopePositions.
	InpPosVision ggml.Tensor
	// NPatchesX, NPatchesY are the spatial dims of the patch grid (e.g.
	// 14 each for a 224×224 image with 16-px patches). NPatchesX * NPatchesY
	// must equal PosX.Ne(0) and PosY.Ne(0).
	NPatchesX, NPatchesY int
}

// BuildVisionGraph builds the full vision encoder + projector forward
// graph for one image and returns the projected embeddings tensor —
// ne = [n_decoder_embd, n_image_tokens, 1] — ready to splice into the
// decoder's input embedding stream.
//
// The implementation mirrors llama.cpp's clip_graph_gemma4v::build()
// closely; numerical equivalence against `llama-mtmd` is a Phase 8
// validation step.
// mulMatClamped wraps ggml.MulMat with Gemma4ClippableLinear semantics:
// clamp input to [inMin, inMax], MulMat, clamp output to [outMin, outMax].
// When `clamp` is inactive (sentinel / absent), falls back to bare MulMat —
// byte-identical to an un-clamped matmul, which is why every decoder path
// (nil clamp map → inactive clamp) keeps its exact graph.
// Mirrors clip_graph_gemma4v::build_mm in llama.cpp's mtmd.
//
// This is the single shared clamp-around-matmul helper consumed by both the
// procedural projector path here and the generic block/FFN builders (which
// resolve their clamp from GraphInputs.LinearClamps).
func mulMatClamped(ctx *ggml.GraphContext, w, x ggml.Tensor, clamp LinearClamp) ggml.Tensor {
	if !clamp.Active() {
		return ggml.MulMat(ctx, w, x)
	}
	clamped := ggml.Clamp(ctx, x, clamp.InMin, clamp.InMax)
	out := ggml.MulMat(ctx, w, clamped)
	out = ggml.Clamp(ctx, out, clamp.OutMin, clamp.OutMax)
	return out
}

// clampFor resolves a builder weight key against a (possibly nil) clamp map.
// A nil map or absent key yields the zero LinearClamp (inactive), so callers
// route every projection through mulMatClamped unconditionally and the
// no-clamp case stays a bare matmul.
func clampFor(clamps map[string]LinearClamp, key string) LinearClamp {
	if clamps == nil {
		return LinearClamp{}
	}
	return clamps[key]
}

// projectorClamp returns the projector clamp for a logical key.
func projectorClamp(tensors *VisionTensors, key string) LinearClamp {
	if tensors == nil || tensors.Clamps == nil || tensors.Clamps.Projector == nil {
		return LinearClamp{}
	}
	return tensors.Clamps.Projector[key]
}

// VisionBuilders holds the resolved per-layer block and FFN builders for the
// vision tower, plus the builder-key → layer-map-key remaps that let the
// generic builders read from the flat per-layer tensor map (which is keyed by
// the [vision.layers.common_weights] logical names, not the builders' own
// weight keys). The Gemma-4 tower is uniform — one block type for every
// layer — so a single Block/FFN pair suffices; per-layer routing can grow into
// slices here without touching the dispatch loop's shape.
type VisionBuilders struct {
	BlockName   string
	Block       BlockBuilder
	BlockConfig map[string]any
	// BlockWeights maps a block-builder weight key (e.g. WeightAttnQ) to the
	// logical name under which that tensor lives in VisionTensors.Layers[il].
	BlockWeights map[string]string

	FFN       FFNBuilder
	FFNConfig map[string]any
	// FFNWeights maps an FFN-builder weight key (e.g. MoEGate="gate") to the
	// logical name in VisionTensors.Layers[il] (e.g. "ffn_gate").
	FFNWeights map[string]string

	// Params is a synthetic ResolvedParams projecting VisionParams into the
	// flat shape the generic builders read (head_dim, n_heads, n_kv_heads,
	// rms_eps, rope_freq_base). Built once at load time.
	Params *ResolvedParams
}

// ResolveVisionBuilders mirrors assignBuilders for the vision tower: it picks
// the uniform block builder named by [vision.layers.routing].uniform, the
// [vision.ffn] FFN builder, and captures their weight-key remaps and config.
// Returns (nil, nil) when the arch has no [vision] section.
func ResolveVisionBuilders(def *ArchDef, params *VisionParams) (*VisionBuilders, error) {
	if def.Vision == nil {
		return nil, nil
	}
	// Routing: Gemma's tower is uniform. Consume Routing.Uniform; fall back to
	// the sole declared block when routing is omitted (single-block towers).
	blockName := def.Vision.Layers.Routing.Uniform
	if blockName == "" {
		if len(def.Vision.Blocks) != 1 {
			return nil, fmt.Errorf("vision: layers.routing.uniform required when [vision.blocks] declares %d blocks", len(def.Vision.Blocks))
		}
		for name := range def.Vision.Blocks {
			blockName = name
		}
	}
	blockDef, ok := def.Vision.Blocks[blockName]
	if !ok {
		return nil, fmt.Errorf("vision: routing references unknown block %q", blockName)
	}
	bb, ok := GetBlockBuilder(blockDef.Builder)
	if !ok {
		return nil, fmt.Errorf("vision: unknown block builder %q", blockDef.Builder)
	}
	fb, ok := GetFFNBuilder(def.Vision.FFN.Builder)
	if !ok {
		return nil, fmt.Errorf("vision: unknown FFN builder %q", def.Vision.FFN.Builder)
	}

	// Synthetic params: the generic AttentionBuilder reads head_dim / n_heads /
	// n_kv_heads / rms_eps from ResolvedParams and rope_freq_base for axial2d.
	// Vision has no GQA, so n_kv_heads == n_heads. rope_freq_base carries the
	// tower's RoPE theta so the TOML need not duplicate [vision].rope_theta.
	synthParams := &ResolvedParams{
		Ints: map[string]int{
			ParamHeadDim:  params.HeadDim,
			ParamNHeads:   params.NHeads,
			ParamNKVHeads: params.NHeads,
			ParamRoPENRot: params.HeadDim,
		},
		Floats: map[string]float32{
			ParamRMSEps:       params.RMSEps,
			ParamRoPEFreqBase: params.RopeTheta,
		},
	}

	return &VisionBuilders{
		BlockName:    blockName,
		Block:        bb,
		BlockConfig:  blockDef.Config,
		BlockWeights: blockDef.Weights,
		FFN:          fb,
		FFNConfig:    def.Vision.FFN.Config,
		FFNWeights:   def.Vision.FFN.Weights,
		Params:       synthParams,
	}, nil
}

// usesMRopeVision reports whether the tower's encoder block is configured for
// rope=mrope_vision (Qwen3-VL). Drives whether the splice builds the 4-channel
// M-RoPE position buffer; the axial Gemma tower returns false and pays nothing.
func (vb *VisionBuilders) usesMRopeVision() bool {
	if vb == nil {
		return false
	}
	return configStrOr(vb.BlockConfig, ConfigRope, "") == RopeMRopeVision
}

// remapWeights builds a new weight map keyed by builder-key, pulling each
// tensor from layerMap by the logical name the remap points at. Used to feed
// the flat per-layer vision tensor map into a generic builder whose keys
// differ (e.g. FFN builder wants "gate" but the tensor lives under "ffn_gate").
func remapWeights(remap map[string]string, layerMap map[string]ggml.Tensor) map[string]ggml.Tensor {
	out := make(map[string]ggml.Tensor, len(remap))
	for builderKey, logicalName := range remap {
		if t, ok := layerMap[logicalName]; ok {
			out[builderKey] = t
		} else {
			out[builderKey] = ggml.NilTensor()
		}
	}
	return out
}

// visionLayerClamps returns the per-layer clamp map for the dispatch loop's
// GraphInputs.LinearClamps. Nil (no clamps loaded) yields a nil map → every
// clampFor lookup is inactive → bare matmul.
func visionLayerClamps(tensors *VisionTensors, il int) map[string]LinearClamp {
	if tensors == nil || tensors.Clamps == nil || il >= len(tensors.Clamps.Layer) {
		return nil
	}
	return tensors.Clamps.Layer[il]
}

func BuildVisionGraph(
	ctx *ggml.GraphContext,
	gf *ggml.Graph,
	inputs *VisionInputs,
	params *VisionParams,
	tensors *VisionTensors,
	builders *VisionBuilders,
	caps *ForwardCaptures,
) (ggml.Tensor, error) {
	if inputs == nil || params == nil || tensors == nil || builders == nil {
		return ggml.NilTensor(), fmt.Errorf("BuildVisionGraph: nil input")
	}
	if inputs.InpRaw.IsNil() {
		return ggml.NilTensor(), fmt.Errorf("BuildVisionGraph: nil input image tensor")
	}
	// Axial-2D towers need PosX/PosY; M-RoPE towers need InpPosVision. Require
	// the pair the active tower's pos-embed + rope path will consume — the
	// other pair is legitimately nil (see buildVisionSplice).
	usesMRope := builders.usesMRopeVision()
	if usesMRope {
		if inputs.InpPosVision.IsNil() {
			return ggml.NilTensor(), fmt.Errorf("BuildVisionGraph: mrope_vision tower requires InpPosVision")
		}
	} else if inputs.PosX.IsNil() || inputs.PosY.IsNil() {
		return ggml.NilTensor(), fmt.Errorf("BuildVisionGraph: axial tower requires PosX and PosY")
	}

	nEmbd := int64(params.NEmbd)
	nPatches := int64(inputs.NPatchesX * inputs.NPatchesY)

	// ---- Patch embedding ----
	// Step 1: rescale raw image from [0, 1] to [-1, 1] via x = 2x - 1.
	// (Matches clip.cpp `ggml_scale_bias(inp_raw, 2.0f, -1.0f)`; mean/std = [0.5]
	// for both Gemma 4 and Qwen3-VL.)
	inp := ggml.ScaleBias(ctx, inputs.InpRaw, 2.0, -1.0)
	caps.NamedTensor(ctx, gf, "vision.inp_raw_scaled", inp)

	// Step 2/3: produce the flat [n_embd, n_patches] token sequence. Two layouts:
	//   - dual-conv (Qwen3-VL): two Conv2D summed + bias, then a merge-grouped
	//     permute/reshape (qwen3vl.cpp build()).
	//   - single-conv (Gemma 4): one Conv2D + reshape/transpose.
	// Gated on the presence of the second patch-embed kernel so Gemma is
	// untouched.
	var err error
	inp, err = buildPatchEmbed(ctx, gf, inp, tensors, params, inputs, caps)
	if err != nil {
		return ggml.NilTensor(), err
	}

	// ---- Additive position embedding ----
	inp, err = buildPositionEmbed(ctx, gf, inp, tensors, params, inputs, caps)
	if err != nil {
		return ggml.NilTensor(), err
	}

	// ---- Layer loop (§4 generic block/FFN dispatch) ----
	//
	// This mirrors the decoder's runLayers frame (graph.go): a per-layer
	// pre-attn norm → block builder → post-attn norm → residual, then a
	// pre-FFN norm → FFN builder → post-FFN norm → residual. The vision tower
	// is "another model on the same code paths": the generic AttentionBuilder
	// (rope=axial2d/mrope_vision, qk_norm via weight presence, v_norm, kq_scale,
	// kq_prec=native, non_causal=true) reproduces both towers' attention op-for-
	// op, and the geglu/mlp builders reproduce both FFNs. Gemma's clamps ride in
	// via GraphInputs.LinearClamps.
	//
	// The frame norm is norm-type-aware (params.NormType): RmsNorm (Gemma) or
	// LayerNorm-with-bias (Qwen). Post-attn / post-FFN norms apply only when the
	// weight is present — Qwen has none, Gemma has both, so the Gemma path is
	// byte-identical.
	inpL := inp // running residual
	for il := 0; il < params.NLayers; il++ {
		lt := tensors.Layers[il]
		blockWeights := remapWeights(builders.BlockWeights, lt)
		// Fused QKV (Qwen3-VL): split the single attn_qkv weight/bias into the
		// q/k/v + bias views the generic AttentionBuilder expects. No-op for
		// Gemma (no attn_qkv tensor → splitFusedQKV leaves blockWeights as-is).
		splitFusedQKV(ctx, builders.BlockConfig, lt, blockWeights, nEmbd)
		ffnWeights := remapWeights(builders.FFNWeights, lt)

		layerInputs := &GraphInputs{
			NTokens:      nPatches,
			NKV:          nPatches,
			PosX:         inputs.PosX,
			PosY:         inputs.PosY,
			InpPosVision: inputs.InpPosVision,
			InpMask:      ggml.NilTensor(), // non_causal block config selects nil regardless
			FlashAttn:    false,
			Captures:     caps,
			CurrentLayer: il,
			LinearClamps: visionLayerClamps(tensors, il),
		}

		// --- Attention block: ln1 → block builder → attn_post_norm → residual ---
		cur := visionNormApply(ctx, inpL, lt[WeightVisionLN1], lt[WeightVisionLN1Bias], params)
		caps.NamedTensor(ctx, gf, fmt.Sprintf("vision.ln1_%d", il), cur)
		cur = builders.Block.BuildStateless(ctx, cur, blockWeights, builders.Params,
			builders.BlockConfig, layerInputs, nil)
		caps.NamedTensor(ctx, gf, fmt.Sprintf("vision.attn_out_%d", il), cur)
		cur = visionPostNorm(ctx, cur, lt[WeightAttnPostNorm], params)
		inpL = ggml.Add(ctx, inpL, cur)
		caps.NamedTensor(ctx, gf, fmt.Sprintf("vision.ffn_inp_%d", il), inpL)

		// --- FFN block: ln2 → FFN builder → ffn_post_norm → residual ---
		ffn := visionNormApply(ctx, inpL, lt[WeightVisionLN2], lt[WeightVisionLN2Bias], params)
		caps.NamedTensor(ctx, gf, fmt.Sprintf("vision.ffn_inp_normed_%d", il), ffn)
		ffn = builders.FFN.BuildFFN(ctx, ffn, ffnWeights, builders.Params,
			builders.FFNConfig, layerInputs)
		caps.NamedTensor(ctx, gf, fmt.Sprintf("vision.ffn_out_%d", il), ffn)
		ffn = visionPostNorm(ctx, ffn, lt[WeightFFNPostNorm], params)
		inpL = ggml.Add(ctx, inpL, ffn)

		caps.NamedTensor(ctx, gf, fmt.Sprintf("vision.layer_%d", il), inpL)
	}

	// ---- Post-encoder norm (Qwen3-VL: post_ln LayerNorm; Gemma: absent) ----
	if pln := tensors.Global[WeightVisionPostLN]; !pln.IsNil() {
		inpL = visionNormApply(ctx, inpL, pln, tensors.Global[WeightVisionPostLNBias], params)
		caps.NamedTensor(ctx, gf, "vision.post_ln", inpL)
	}

	// ---- Merger + projector ----
	projected, err := buildProjector(ctx, gf, inpL, tensors, params, inputs, caps)
	if err != nil {
		return ggml.NilTensor(), err
	}

	gf.BuildForwardExpand(projected)
	return projected, nil
}

// visionRopeNoYaRN applies NeoX RoPE without YaRN scaling — beta_fast and
// beta_slow are both zero, which disables the YaRN ramp the decoder uses
// (the decoder's defaultRopeExt passes 32.0 / 1.0). Vision encoders rotate
// over patch-grid positions, not sequence positions, so the long-context
// YaRN correction doesn't apply.
func visionRopeNoYaRN(ctx *ggml.GraphContext, a, pos ggml.Tensor, nRot int, ropeTheta float32) ggml.Tensor {
	return ggml.RopeExt(ctx, a, pos, ggml.NilTensor(), nRot, ggml.RopeTypeNeoX, 0,
		ropeTheta, 1.0, 0.0, 1.0, 0.0, 0.0)
}

// applyRope2D applies NeoX RoPE separately to the two halves of each
// head's dim: first half rotated by pos_x, second by pos_y. Input cur
// has ne = [head_dim, n_heads, n_patches]; output has the same shape
// with halves rejoined via Concat along axis 0.
func applyRope2D(ctx *ggml.GraphContext, cur, posX, posY ggml.Tensor, headDim int64, ropeTheta float32) ggml.Tensor {
	half := headDim / 2
	// Strides: nb1 is between rows of head_dim; nb2 between rows of
	// (head_dim*n_heads). View3D reads (ne0, ne1, ne2, nb1, nb2, offset).
	nb1 := int(cur.Nb(1))
	nb2 := int(cur.Nb(2))
	elemSize := int(cur.Nb(0))

	first := ggml.View3D(ctx, cur, half, cur.Ne(1), cur.Ne(2), nb1, nb2, 0)
	first = visionRopeNoYaRN(ctx, first, posX, int(half), ropeTheta)

	second := ggml.View3D(ctx, cur, half, cur.Ne(1), cur.Ne(2), nb1, nb2, int(half)*elemSize)
	second = visionRopeNoYaRN(ctx, second, posY, int(half), ropeTheta)

	return ggml.Concat(ctx, first, second, 0)
}

// visionNormApply applies the encoder dispatch frame's pre-block norm in a
// norm-type-aware way: VisionNormLayerNorm → ggml_norm + (weight, optBias)
// affine (Qwen3-VL); anything else → weight-only RmsNorm (Gemma 4). The bias
// arg is ignored for RMS (Gemma carries none). One code path, data-driven on
// params.NormType, keeps both towers in the same frame.
func visionNormApply(ctx *ggml.GraphContext, x, weight, optBias ggml.Tensor, params *VisionParams) ggml.Tensor {
	if params.NormType == VisionNormLayerNorm {
		return layerNormApply(ctx, x, weight, optBias, params.RMSEps)
	}
	return rmsNormApply(ctx, x, weight, params.RMSEps)
}

// visionPostNorm applies an optional post-attn / post-FFN norm ONLY when the
// weight is present. Gemma 4 supplies both (RmsNorm); Qwen3-VL supplies neither
// (the IsNil short-circuit makes it a no-op, leaving the graph byte-identical).
// Honors the configured norm type for the present-weight case.
func visionPostNorm(ctx *ggml.GraphContext, x, weight ggml.Tensor, params *VisionParams) ggml.Tensor {
	if weight.IsNil() {
		return x
	}
	if params.NormType == VisionNormLayerNorm {
		return layerNormApply(ctx, x, weight, ggml.NilTensor(), params.RMSEps)
	}
	return rmsNormApply(ctx, x, weight, params.RMSEps)
}

// splitFusedQKV implements config flag `qkv_fused = true` (Qwen3-VL): the model
// ships a single attn_qkv weight [n_embd, 3*n_embd] and bias [3*n_embd]. The
// generic AttentionBuilder wants three separate q/k/v weights + biases. Rather
// than change the builder, we slice the fused tensors into contiguous View2D
// (weight) / View1D (bias) thirds — q=[0:n_embd], k=[n:2n], v=[2n:3n] along the
// output dim. Three matmuls on these contiguous row-slices are numerically
// identical to llama.cpp's fused matmul-then-view (qwen3vl.cpp self-attention).
//
// No-op when the config flag is unset or the attn_qkv tensor is absent (every
// Gemma / decoder path), leaving blockWeights as remapWeights produced it.
//
// blockWeights is mutated in place: the q/k/v + bias builder keys are filled
// with the views.
func splitFusedQKV(ctx *ggml.GraphContext, config map[string]any, layerTensors, blockWeights map[string]ggml.Tensor, nEmbd int64) {
	if !configBoolOr(config, ConfigQKVFused, false) {
		return
	}
	qkvW := layerTensors[WeightVisionAttnQKV]
	if qkvW.IsNil() {
		return
	}
	// Weight ne = [n_embd_in, 3*n_embd_out]; row stride nb1 spans one output row.
	rowBytes := int(qkvW.Nb(1))
	blockWeights[WeightAttnQ] = ggml.View2D(ctx, qkvW, nEmbd, nEmbd, rowBytes, 0)
	blockWeights[WeightAttnK] = ggml.View2D(ctx, qkvW, nEmbd, nEmbd, rowBytes, int(nEmbd)*rowBytes)
	blockWeights[WeightAttnV] = ggml.View2D(ctx, qkvW, nEmbd, nEmbd, rowBytes, int(2*nEmbd)*rowBytes)

	qkvB := layerTensors[WeightVisionAttnQKVBias]
	if !qkvB.IsNil() {
		// Bias ne = [3*n_embd]; slice the same thirds. A 1D slice is a View2D
		// with ne1 = 1 (one logical column), elem-stride along ne0.
		elem := int(qkvB.Nb(0))
		blockWeights[WeightAttnQBias] = ggml.View2D(ctx, qkvB, nEmbd, 1, int(nEmbd)*elem, 0)
		blockWeights[WeightAttnKBias] = ggml.View2D(ctx, qkvB, nEmbd, 1, int(nEmbd)*elem, int(nEmbd)*elem)
		blockWeights[WeightAttnVBias] = ggml.View2D(ctx, qkvB, nEmbd, 1, int(nEmbd)*elem, int(2*nEmbd)*elem)
	}
}

// buildPatchEmbed produces the flat [n_embd, n_patches] token sequence from the
// scaled image `inp`. Dual-conv (Qwen3-VL) when a second patch-embed kernel is
// present; single-conv (Gemma 4) otherwise. Returns the embedded sequence.
func buildPatchEmbed(ctx *ggml.GraphContext, gf *ggml.Graph, inp ggml.Tensor,
	tensors *VisionTensors, params *VisionParams, inputs *VisionInputs, caps *ForwardCaptures) (ggml.Tensor, error) {

	nEmbd := int64(params.NEmbd)
	patchSize := params.PatchSize
	nPatchesX := int64(inputs.NPatchesX)
	nPatchesY := int64(inputs.NPatchesY)
	nPatches := nPatchesX * nPatchesY

	patchEmbd := tensors.Global[WeightVisionPatchEmbd]
	if patchEmbd.IsNil() {
		return ggml.NilTensor(), fmt.Errorf("BuildVisionGraph: vision.patch_embd weight missing")
	}
	patchEmbd1 := tensors.Global[WeightVisionPatchEmbd1]

	if !patchEmbd1.IsNil() {
		// --- Dual-conv merge-grouped patch embed (Qwen3-VL) ---
		// Mirrors qwen3vl.cpp build() lines 19-39 op-for-op.
		conv := ggml.Conv2D(ctx, patchEmbd, inp, patchSize, patchSize, 0, 0, 1, 1)
		conv1 := ggml.Conv2D(ctx, patchEmbd1, inp, patchSize, patchSize, 0, 0, 1, 1)
		e := ggml.Add(ctx, conv, conv1) // [w, h, c, 1]

		// permute [w,h,c,b] -> [c,w,h,b], then the two reshape/permute steps that
		// bake the NMerge×NMerge spatial-merge grouping into token order.
		merge := int64(params.NMerge)
		e = ggml.Permute(ctx, e, 1, 2, 0, 3) // -> [c, w, h, b]
		e = ggml.Cont4D(ctx, e, nEmbd*merge, nPatchesX/merge, nPatchesY, 1)
		e = ggml.Reshape4D(ctx, e, nEmbd*merge, nPatchesX/merge, merge, nPatchesY/merge)
		e = ggml.Permute(ctx, e, 0, 2, 1, 3)
		e = ggml.Cont3D(ctx, e, nEmbd, nPatches, 1)

		// add patch bias (broadcast over tokens)
		if bias := tensors.Global[WeightVisionPatchBias]; !bias.IsNil() {
			e = ggml.Add(ctx, e, bias)
		}
		caps.NamedTensor(ctx, gf, "vision.patch_embd", e)
		return e, nil
	}

	// --- Single-conv patch embed (Gemma 4 — byte-identical to prior code) ---
	var kernel ggml.Tensor
	if patchEmbd.Ne(0) == int64(patchSize) && patchEmbd.Ne(2) == 3 {
		// mmproj GGUF: already ne=[kw, kh, ic, oc].
		kernel = patchEmbd
	} else {
		// Safetensors-flat: reshape to (c, kw, kh, oc) and permute.
		kernel = ggml.Reshape4D(ctx, patchEmbd, 3, int64(patchSize), int64(patchSize), nEmbd)
		kernel = ggml.Cont(ctx, ggml.Permute(ctx, kernel, 2, 0, 1, 3))
	}
	e := ggml.Conv2D(ctx, kernel, inp, patchSize, patchSize, 0, 0, 1, 1)
	caps.NamedTensor(ctx, gf, "vision.patch_embd", e)
	e = ggml.Reshape2D(ctx, e, nPatches, nEmbd)
	e = ggml.Cont(ctx, ggml.Transpose(ctx, e))
	return e, nil
}

// buildPositionEmbed adds the learned position embedding to the patch sequence.
// Two layouts, distinguished by the table rank (data-driven, no arch branch):
//   - rank-3 axial table [n_embd, max_per_axis, 2] (Gemma 4): independent X/Y
//     GetRows lookups, summed. Byte-identical to prior code.
//   - rank-2 learned grid [n_embd, n_per_side^2] (Qwen3-VL): bilinearly resize
//     to the patch grid via ggml_interpolate, then merge-group reshape + add.
//     Mirrors clip.cpp resize_position_embeddings + qwen3vl.cpp build() 43-58.
func buildPositionEmbed(ctx *ggml.GraphContext, gf *ggml.Graph, inp ggml.Tensor,
	tensors *VisionTensors, params *VisionParams, inputs *VisionInputs, caps *ForwardCaptures) (ggml.Tensor, error) {

	nEmbd := int64(params.NEmbd)
	posEmbd := tensors.Global[WeightVisionPosEmbd]
	if posEmbd.IsNil() {
		return ggml.NilTensor(), fmt.Errorf("BuildVisionGraph: vision.position_embd missing")
	}
	nPatchesX := int64(inputs.NPatchesX)
	nPatchesY := int64(inputs.NPatchesY)
	nPatches := nPatchesX * nPatchesY

	if posEmbd.Ne(2) <= 1 {
		// --- Qwen3-VL learned-grid, interpolated ---
		// pos_embd ne = [n_embd, n_per_side*n_per_side]. resize_position_embeddings:
		//   reshape -> [n_embd, n_per_side, n_per_side]
		//   permute(2,0,1) -> [n_per_side, n_per_side, n_embd]
		//   interpolate -> [width, height, n_embd]   (bilinear|antialias)
		//   permute(1,2,0) -> [n_embd, width, height]
		//   cont_2d -> [n_embd, width*height]
		nPerSide := int64(math.Round(math.Sqrt(float64(posEmbd.Ne(1)))))
		// ggml_interpolate requires an F32 input. The mmproj GGUF stores
		// v.position_embd.weight as F32, but the safetensors path loads it at
		// the source dtype (F16 from a BF16 .st tensor) — cast to keep both
		// formats on the same op contract. No-op when already F32 (GGUF).
		if ggml.TensorType(posEmbd) != ggml.TypeF32 {
			posEmbd = ggml.Cast(ctx, posEmbd, ggml.TypeF32)
		}
		pe := ggml.Reshape3D(ctx, posEmbd, nEmbd, nPerSide, nPerSide)
		pe = ggml.Permute(ctx, pe, 2, 0, 1, 3)
		mode := ggml.ScaleModeBilinear | ggml.ScaleFlagAntialias // DEFAULT_INTERPOLATION_MODE (clip-graph.h)
		if nPerSide != nPatchesX || nPerSide != nPatchesY {
			pe = ggml.Interpolate(ctx, pe, nPatchesX, nPatchesY, nEmbd, 1, mode)
		}
		pe = ggml.Permute(ctx, pe, 1, 2, 0, 3)
		pe = ggml.Cont2D(ctx, pe, nEmbd, nPatches)

		// Merge-group the resized table into the same token order the patch
		// embed uses (qwen3vl.cpp build() lines 44-57).
		merge := int64(params.NMerge)
		pe = ggml.Cont4D(ctx, pe, nEmbd*merge, nPatchesX/merge, nPatchesY, 1)
		pe = ggml.Reshape4D(ctx, pe, nEmbd*merge, nPatchesX/merge, merge, nPatchesY/merge)
		pe = ggml.Permute(ctx, pe, 0, 2, 1, 3)
		pe = ggml.Cont3D(ctx, pe, nEmbd, nPatches, 1)

		inp = ggml.Add(ctx, inp, pe)
		caps.NamedTensor(ctx, gf, "vision.inp_pos_embd", inp)
		return inp, nil
	}

	// --- Gemma 4 axial table (byte-identical to prior code) ---
	maxPerAxis := posEmbd.Ne(1)
	rowBytes := int(posEmbd.Nb(1))
	tblX := ggml.View2D(ctx, posEmbd, nEmbd, maxPerAxis, rowBytes, 0)
	tblY := ggml.View2D(ctx, posEmbd, nEmbd, maxPerAxis, rowBytes, int(maxPerAxis)*rowBytes)
	embX := ggml.GetRows(ctx, tblX, inputs.PosX)
	embY := ggml.GetRows(ctx, tblY, inputs.PosY)
	inp = ggml.Add(ctx, inp, embX)
	inp = ggml.Add(ctx, inp, embY)
	caps.NamedTensor(ctx, gf, "vision.inp_pos_embd", inp)
	return inp, nil
}

// buildProjector runs the merger + projector. Two types, gated on
// params.ProjectorType:
//   - ProjectorLinearPostNorm (Gemma 4): 2D avg-pool merger + sqrt(n_embd)
//     scale, unit-weight pre-projection RmsNorm, then single linear
//     (gemma4v.cpp Gemma4MultimodalEmbedder: rms_norm BEFORE build_mm).
//   - ProjectorMLP (Qwen3-VL): reshape-only [n_embd*4, n/4] merger (the 2x2
//     grouping was baked in at patch embed), then mm.0 → GELU → mm.2 with
//     biases (qwen3vl.cpp build() multimodal projection).
func buildProjector(ctx *ggml.GraphContext, gf *ggml.Graph, inpL ggml.Tensor,
	tensors *VisionTensors, params *VisionParams, inputs *VisionInputs, caps *ForwardCaptures) (ggml.Tensor, error) {

	nEmbd := int64(params.NEmbd)

	if params.ProjectorType == ProjectorMLP {
		// Merger: reshape-only NMerge×NMerge (grouping already in token order).
		// inpL ne = [n_embd, n_patches] -> [n_embd*NMerge², n_patches/NMerge²].
		mergeSq := int64(params.NMerge * params.NMerge)
		merged := ggml.Reshape3D(ctx, inpL, nEmbd*mergeSq, inpL.Ne(1)/mergeSq, 1)
		caps.NamedTensor(ctx, gf, "vision.merger", merged)

		w0 := tensors.Projector[WeightVisionProj]
		w1 := tensors.Projector[WeightVisionProj2]
		if w0.IsNil() || w1.IsNil() {
			return ggml.NilTensor(), fmt.Errorf("BuildVisionGraph: mlp projector requires proj + proj2 weights")
		}
		x := ggml.MulMat(ctx, w0, merged)
		x = addFFNBias(ctx, x, tensors.Projector[WeightVisionProjBias])
		x = ggml.Gelu(ctx, x) // FFN_GELU (tanh-approx) per clip.use_gelu / qwen3vl.cpp
		x = ggml.MulMat(ctx, w1, x)
		x = addFFNBias(ctx, x, tensors.Projector[WeightVisionProj2Bias])
		x = ggml.Reshape2D(ctx, x, x.Ne(0), x.Ne(1))
		caps.NamedTensor(ctx, gf, "vision.projected", x)
		return x, nil
	}

	// --- Gemma 4 pooler + linear + post-norm (byte-identical to prior code) ---
	pooled := ggml.Cont4D(ctx, ggml.Transpose(ctx, inpL),
		int64(inputs.NPatchesX), int64(inputs.NPatchesY), nEmbd, 1)
	k := params.NMerge
	pooled = ggml.Pool2D(ctx, pooled, ggml.PoolAvg, k, k, k, k, 0.0, 0.0)
	outX := int64(inputs.NPatchesX / k)
	outY := int64(inputs.NPatchesY / k)
	pooled = ggml.Reshape3D(ctx, pooled, outX*outY, nEmbd, 1)
	pooled = ggml.Cont(ctx, ggml.Transpose(ctx, pooled)) // [n_embd, n_pooled]
	pooled = ggml.Scale(ctx, pooled, float32(math.Sqrt(float64(nEmbd))))
	caps.NamedTensor(ctx, gf, "vision.pooled", pooled)

	proj := tensors.Projector[WeightVisionProj]
	if proj.IsNil() {
		return ggml.NilTensor(), fmt.Errorf("BuildVisionGraph: projector weight missing")
	}
	// Gemma4MultimodalEmbedder (qwen3vl/gemma4v.cpp build()): unit-weight
	// embedding_pre_projection_norm on the pooled n_embd-dim input, THEN the
	// linear projection. (No learned norm weight; std_bias/std_scale absent for
	// this mmproj.) Earlier this applied the norm AFTER projection on the
	// out-dim tensor, which forced a constant per-token L2 and dropped per-token
	// magnitude into the decoder residual stream.
	normed := ggml.RmsNorm(ctx, pooled, params.RMSEps)
	caps.NamedTensor(ctx, gf, "vision.projected_normed", normed)
	projected := mulMatClamped(ctx, proj, normed, projectorClamp(tensors, WeightVisionProj))
	caps.NamedTensor(ctx, gf, "vision.projected", projected)
	return projected, nil
}
