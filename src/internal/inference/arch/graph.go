package arch

import (
	"fmt"
	"math"
	"strings"
	"unsafe"

	ggml "inference-lab-bench/internal/ggml"
)

// nodesPerLayer estimates ggml graph nodes per transformer layer. The +1 in
// graphCtxSize accounts for global ops (embedding lookup, final norm, LM head).
const nodesPerLayer = 160

func graphCtxSize(nLayers int) int {
	return 4*1024*1024 + ggml.TensorOverhead()*(nLayers+1)*nodesPerLayer
}

// buildCausalMaskData generates the float32 mask data for causal attention.
// Positions where key > query get -Inf. Returns all zeros if nonCausal.
// If buf is non-nil and large enough (len >= nQuery*nKV), it is zeroed and reused;
// otherwise a new slice is allocated. The returned slice may alias buf.
func buildCausalMaskData(buf []float32, nQuery, nKV int64, startPos int, nonCausal bool) []float32 {
	need := int(nQuery * nKV)
	var data []float32
	if len(buf) >= need {
		data = buf[:need]
		clear(data)
	} else {
		data = make([]float32, need)
	}
	if !nonCausal {
		for qi := int64(0); qi < nQuery; qi++ {
			absQI := int64(startPos) + qi
			for kj := int64(0); kj < nKV; kj++ {
				if kj > absQI {
					data[qi*nKV+kj] = float32(math.Inf(-1))
				}
			}
		}
	}
	return data
}

// buildSWAMaskData generates the float32 mask data for sliding window attention.
// Positions where key > query or key < query - window get -Inf.
// If buf is non-nil and large enough (len >= nQuery*nKV), it is zeroed and reused;
// otherwise a new slice is allocated. The returned slice may alias buf.
func buildSWAMaskData(buf []float32, nQuery, nKV int64, startPos, window int) []float32 {
	need := int(nQuery * nKV)
	var data []float32
	if len(buf) >= need {
		data = buf[:need]
		clear(data)
	} else {
		data = make([]float32, need)
	}
	for qi := int64(0); qi < nQuery; qi++ {
		absQI := int64(startPos) + qi
		for kj := int64(0); kj < nKV; kj++ {
			if kj > absQI || kj < absQI-int64(window) {
				data[qi*nKV+kj] = float32(math.Inf(-1))
			}
		}
	}
	return data
}

// blockFunc is the per-layer block builder call, injected by the caller to
// select BuildStateless or BuildCached. The surrounding layer loop (runLayers)
// enforces the residual stream invariant: norm → block → residual → FFN → residual.
type blockFunc func(gctx *ggml.GraphContext, cur ggml.Tensor,
	lt map[string]ggml.Tensor, il int, config map[string]any,
	inputs *GraphInputs) ggml.Tensor

// runLayers executes the per-layer forward pass: for each layer, apply attention/SSM
// block (via blkFn) with residual, then FFN with residual. Layers with culled attn_norm
// skip the block entirely; layers with all FFN tensors culled skip FFN.
func (m *GenericModel) runLayers(gctx *ggml.GraphContext, x ggml.Tensor,
	inputs *GraphInputs, rmsEps float32, mask *CullingMask,
	blkFn blockFunc, perLayerEmbd ggml.Tensor,
	engagement *EngagementData) ggml.Tensor {
	for il := range m.Params.Ints["n_layers"] {
		lt := m.MaskedLayer(il, mask)
		blockName := m.LayerBlockNames[il]
		blockDef := m.Def.Blocks[blockName]
		inputs.CurrentLayer = il

		xPreBlock := x
		if !lt["attn_norm"].IsNil() {
			cur := rmsNormApply(gctx, x, lt["attn_norm"], rmsEps)
			cur = blkFn(gctx, cur, lt, il, blockDef.Config, inputs)
			// Optional post-attention norm
			if !lt["attn_post_norm"].IsNil() {
				cur = rmsNormApply(gctx, cur, lt["attn_post_norm"], rmsEps)
			}
			x = ggml.Add(gctx, x, cur)
			if engagement != nil {
				engagement.blockTensors[il] = buildCosineSim(gctx, xPreBlock, x)
			}
		}

		xPreFFN := x
		x = m.buildFFNBlock(gctx, x, lt, il, rmsEps)
		if engagement != nil && x != xPreFFN {
			engagement.ffnTensors[il] = buildCosineSim(gctx, xPreFFN, x)
		}

		// Per-layer embedding injection (Gemma 4)
		if !perLayerEmbd.IsNil() && !lt["pe_inp_gate"].IsNil() {
			x = m.perLayerEmbedInject(gctx, x, lt, il, perLayerEmbd, rmsEps)
		}

		// Layer output scaling
		if !lt["layer_output_scale"].IsNil() {
			x = ggml.Mul(gctx, x, lt["layer_output_scale"])
		}
	}
	return x
}

// buildFFNBlock extracts FFN weights, applies norm, builds FFN, and adds residual.
// Returns x unchanged if all FFN tensors are culled.
func (m *GenericModel) buildFFNBlock(ctx *ggml.GraphContext, x ggml.Tensor,
	lt map[string]ggml.Tensor, il int, rmsEps float32) ggml.Tensor {
	ffnInp := x
	ffnWeights := m.ffnScratch
	clear(ffnWeights)
	for k, v := range lt {
		if after, ok := strings.CutPrefix(k, "ffn_"); ok {
			ffnWeights[after] = v
		}
	}
	if anyNonNil(ffnWeights) {
		ffnConfig := m.FFNConfigs[il]
		// Self-normed builders manage their own pre/post norms internally
		// (e.g. MoE with parallel shared+expert paths that need separate norms).
		if configStr(ffnConfig, "self_normed") == "true" {
			ffnOut := m.FFNBuilders[il].BuildFFN(ctx, x, ffnWeights, m.Params, ffnConfig)
			// Self-normed builders handle internal pre/post norms for each path,
			// but the common post-FFN norm still wraps the combined output.
			if !lt["ffn_post_norm"].IsNil() {
				ffnOut = rmsNormApply(ctx, ffnOut, lt["ffn_post_norm"], rmsEps)
			}
			return ggml.Add(ctx, ffnInp, ffnOut)
		}
		xn2 := rmsNormApply(ctx, x, ffnNorm(lt), rmsEps)
		ffnOut := m.FFNBuilders[il].BuildFFN(ctx, xn2, ffnWeights, m.Params, ffnConfig)
		// Optional post-FFN norm
		if !lt["ffn_post_norm"].IsNil() {
			ffnOut = rmsNormApply(ctx, ffnOut, lt["ffn_post_norm"], rmsEps)
		}
		return ggml.Add(ctx, ffnInp, ffnOut)
	}
	return x
}

// anyNonNil returns true if any tensor in the map is non-nil.
// Used to detect whether an FFN module was culled (all tensors absent).
func anyNonNil(weights map[string]ggml.Tensor) bool {
	for _, v := range weights {
		if !v.IsNil() {
			return true
		}
	}
	return false
}

// ffnNorm returns the pre-FFN norm weight from the layer tensor map.
// All architectures use "ffn_norm" as the canonical logical key in layers.common_weights.
func ffnNorm(lt map[string]ggml.Tensor) ggml.Tensor {
	return lt["ffn_norm"]
}

// buildLogits extracts the last token's embedding, applies final norm, and projects
// through the LM head. Returns the logits tensor (already marked as output).
func (m *GenericModel) buildLogits(gctx *ggml.GraphContext, x ggml.Tensor,
	nEmbd int, nSeq int64, rmsEps float32) ggml.Tensor {
	last := ggml.View2D(gctx, x, int64(nEmbd), 1, x.Nb(1), int(nSeq-1)*x.Nb(1))
	xn := rmsNormApply(gctx, last, m.MaskedGlobal(gctx, "output_norm"), rmsEps)
	logits := ggml.MulMat(gctx, m.MaskedGlobal(gctx, "output"), xn)

	// Logit softcapping: cap * tanh(logits / cap)
	if cap, ok := m.Params.Floats["logit_softcapping"]; ok && cap > 0 {
		logits = ggml.Scale(gctx, logits, 1.0/cap)
		logits = ggml.Tanh(gctx, logits)
		logits = ggml.Scale(gctx, logits, cap)
	}

	ggml.SetOutput(logits)
	return logits
}

// buildAllLogits applies final norm and LM head projection to all token positions.
// Returns logits shaped [nVocab, nTokens] — all positions, all marked as output.
// Used by diffusion generation which needs per-position logits over the full sequence.
func (m *GenericModel) buildAllLogits(gctx *ggml.GraphContext, x ggml.Tensor,
	nEmbd int, rmsEps float32) ggml.Tensor {
	xn := rmsNormApply(gctx, x, m.MaskedGlobal(gctx, "output_norm"), rmsEps)
	logits := ggml.MulMat(gctx, m.MaskedGlobal(gctx, "output"), xn)

	// Logit softcapping: cap * tanh(logits / cap)
	if cap, ok := m.Params.Floats["logit_softcapping"]; ok && cap > 0 {
		logits = ggml.Scale(gctx, logits, 1.0/cap)
		logits = ggml.Tanh(gctx, logits)
		logits = ggml.Scale(gctx, logits, cap)
	}

	ggml.SetOutput(logits)
	return logits
}

// statelessCtx holds the live graph context and scheduler for a stateless forward pass.
// Tensors returned by forwardStatelessCore live in gctx's arena — callers must not
// call Free() until after all readback is complete. Call Free() via defer.
// On error from forwardStatelessCore, do NOT call Free() — the helper frees its own resources.
type statelessCtx struct {
	gctx       *ggml.GraphContext
	sched      *ggml.Sched
	gf         *ggml.Graph
	x          ggml.Tensor
	inputs     *GraphInputs
	inpTokens  ggml.Tensor  // input token ID tensor — needed for TensorSet after AllocGraph
	nEmbd      int
	rmsEps     float32
	nVocab     int
	nTokens    int64
	hasSWA     bool
	swaWindow  int
	zeroFill   []ggml.Tensor
	engagement *EngagementData // non-nil only when withEngagement=true
}

func (sc *statelessCtx) Free() {
	sc.sched.Free()
	sc.gctx.Free()
}

// forwardStatelessCore builds the graph context, scheduler, inputs, and runs all layers
// for a stateless forward pass. Returns a populated *statelessCtx on success.
//
// Callers MUST defer sc.Free() after their readback calls complete —
// tensors live in sc.gctx's arena.
//
// On error, the helper frees its own resources before returning nil — callers
// must NOT call Free() on error.
//
//   withEngagement — allocates and wires EngagementData (ForwardStateless path only)
//   caps           — ForwardCaptures or nil (ForwardStateless path only)
func (m *GenericModel) forwardStatelessCore(
	tokenIDs []int32,
	mask *CullingMask,
	flashAttn bool,
	withEngagement bool,
	caps *ForwardCaptures,
) (*statelessCtx, error) {
	nLayers := m.Params.Ints["n_layers"]
	nEmbd := m.Params.Ints["n_embd"]
	nVocab := m.Params.Ints["n_vocab"]
	rmsEps := m.Params.Floats["rms_eps"]
	nTokens := int64(len(tokenIDs))

	gctx := ggml.NewGraphContext(graphCtxSize(nLayers))
	if gctx == nil {
		return nil, fmt.Errorf("failed to create graph context")
	}

	sched := ggml.NewSched(m.Store.GPU, m.Store.CPU, 16384)
	if sched == nil {
		gctx.Free()
		return nil, fmt.Errorf("failed to create scheduler")
	}

	// Inputs
	inpTokens := ggml.NewTensor1D(gctx, ggml.TypeI32, nTokens)
	ggml.SetInput(inpTokens)
	inpPos := ggml.NewTensor1D(gctx, ggml.TypeI32, nTokens)
	ggml.SetInput(inpPos)
	inpMask := ggml.NewTensor2D(gctx, ggml.TypeF32, nTokens, nTokens)
	ggml.SetInput(inpMask)

	x := ggml.GetRows(gctx, m.MaskedGlobal(gctx, "token_embd"), inpTokens)

	if m.Def.Architecture.EmbedScale {
		x = ggml.Scale(gctx, x, float32(math.Sqrt(float64(nEmbd))))
	}

	swaWindow, hasSWA := m.Params.Ints["sliding_window"]
	var inpMaskSWA ggml.Tensor
	if hasSWA && swaWindow > 0 {
		inpMaskSWA = ggml.NewTensor2D(gctx, ggml.TypeF32, nTokens, nTokens)
		ggml.SetInput(inpMaskSWA)
	}

	if caps != nil && caps.Flags&CaptureAttnWeights != 0 {
		caps.attnTensors = make([]ggml.Tensor, nLayers)
		caps.NHeads = int64(m.Params.Ints["n_heads"])
		caps.NTokens = nTokens
	}

	effectiveFlashAttn := flashAttn && !m.Def.Architecture.NoFlashAttn
	inputs := &GraphInputs{
		InpPos:     inpPos,
		InpMask:    inpMask,
		InpMaskSWA: inpMaskSWA,
		NTokens:    nTokens,
		NKV:        nTokens,
		SeqPos:     0,
		Captures:   caps,
		SharedKV:   &SharedKVState{K: make(map[string]ggml.Tensor), V: make(map[string]ggml.Tensor)},
		FlashAttn:  effectiveFlashAttn,
	}

	perLayerEmbd := m.buildPerLayerEmbedSetup(gctx, inpTokens, x, nTokens, rmsEps)

	var engagement *EngagementData
	if withEngagement {
		engagement = newEngagementData(nLayers)
	}

	var zeroFill []ggml.Tensor
	blkFn := func(gctx *ggml.GraphContext, cur ggml.Tensor,
		lt map[string]ggml.Tensor, il int, config map[string]any,
		inputs *GraphInputs) ggml.Tensor {
		if inputs.Captures != nil {
			inputs.Captures.currentLayer = il
		}
		return m.BlockBuilders[il].BuildStateless(gctx, cur, lt, m.Params, config, inputs, &zeroFill)
	}
	x = m.runLayers(gctx, x, inputs, rmsEps, mask, blkFn, perLayerEmbd, engagement)

	gf := ggml.NewGraph(gctx, 16384)

	return &statelessCtx{
		gctx:       gctx,
		sched:      sched,
		gf:         gf,
		x:          x,
		inputs:     inputs,
		inpTokens:  inpTokens,
		nEmbd:      nEmbd,
		rmsEps:     rmsEps,
		nVocab:     nVocab,
		nTokens:    nTokens,
		hasSWA:     hasSWA,
		swaWindow:  swaWindow,
		zeroFill:   zeroFill,
		engagement: engagement,
	}, nil
}

// ForwardStatelessAllLogits runs a full-sequence forward pass and returns logits for
// all token positions. Returns a slice of length nVocab*nTokens, row-major by position:
// position p occupies allLogits[p*nVocab : (p+1)*nVocab].
// No engagement data or captures are collected — this is intended for diffusion generation.
func (m *GenericModel) ForwardStatelessAllLogits(tokenIDs []int32, mask *CullingMask, flashAttn bool) ([]float32, error) {
	sc, err := m.forwardStatelessCore(tokenIDs, mask, flashAttn, false, nil)
	if err != nil {
		return nil, err
	}
	defer sc.Free()

	logits := m.buildAllLogits(sc.gctx, sc.x, sc.nEmbd, sc.rmsEps)
	sc.gf.BuildForwardExpand(logits)

	if !sc.sched.AllocGraph(sc.gf) {
		return nil, fmt.Errorf("failed to allocate graph")
	}

	for _, t := range sc.zeroFill {
		zeros := make([]byte, t.Nbytes())
		ggml.TensorSetBytes(t, zeros, 0)
	}

	ggml.TensorSet(sc.inpTokens, unsafe.Pointer(&tokenIDs[0]), 0, sc.inpTokens.Nbytes())

	positions := make([]int32, sc.nTokens)
	for i := range positions {
		positions[i] = int32(i)
	}
	ggml.TensorSet(sc.inputs.InpPos, unsafe.Pointer(&positions[0]), 0, sc.inputs.InpPos.Nbytes())

	maskData := buildCausalMaskData(nil, sc.nTokens, sc.nTokens, 0, m.Def.Architecture.NonCausal)
	ggml.TensorSet(sc.inputs.InpMask, unsafe.Pointer(&maskData[0]), 0, sc.inputs.InpMask.Nbytes())

	if sc.hasSWA && sc.swaWindow > 0 {
		swaMaskData := buildSWAMaskData(nil, sc.nTokens, sc.nTokens, 0, sc.swaWindow)
		ggml.TensorSet(sc.inputs.InpMaskSWA, unsafe.Pointer(&swaMaskData[0]), 0, sc.inputs.InpMaskSWA.Nbytes())
	}

	status := sc.sched.Compute(sc.gf)
	if status != ggml.StatusSuccess {
		return nil, fmt.Errorf("%w (status=%d)", ErrComputeFailed, status)
	}

	result := make([]float32, sc.nVocab*int(sc.nTokens))
	ggml.TensorGet(logits, unsafe.Pointer(&result[0]), 0, logits.Nbytes())
	return result, nil
}

// readLogits copies the logits tensor from GPU to a newly allocated float32 slice.
func readLogits(logits ggml.Tensor, nVocab int) []float32 {
	result := make([]float32, nVocab)
	ggml.TensorGet(logits, unsafe.Pointer(&result[0]), 0, logits.Nbytes())
	return result
}

// readLogitsInto copies the logits tensor from GPU into buf and returns buf.
// buf must be pre-allocated with length >= nVocab. The returned slice aliases buf;
// callers must consume the logits before the next decode token overwrites it.
func readLogitsInto(buf []float32, logits ggml.Tensor) []float32 {
	ggml.TensorGet(logits, unsafe.Pointer(&buf[0]), 0, logits.Nbytes())
	return buf
}

// ForwardStateless runs a full-sequence forward pass without caching.
// caps is optional: pass nil for normal inference, or a *ForwardCaptures to extract
// intermediate tensors (e.g. attention weights) alongside the logits.
// Always returns EngagementData (per-layer cosine similarity of the residual stream).
func (m *GenericModel) ForwardStateless(tokenIDs []int32, mask *CullingMask, caps *ForwardCaptures, flashAttn bool) ([]float32, *EngagementData, error) {
	sc, err := m.forwardStatelessCore(tokenIDs, mask, flashAttn, true, caps)
	if err != nil {
		return nil, nil, err
	}
	defer sc.Free()

	logits := m.buildLogits(sc.gctx, sc.x, sc.nEmbd, sc.nTokens, sc.rmsEps)

	// Build graph: logits first, then engagement outputs, then capture outputs
	sc.gf.BuildForwardExpand(logits)
	for _, tn := range sc.engagement.blockTensors {
		if !tn.IsNil() {
			sc.gf.BuildForwardExpand(tn)
		}
	}
	for _, tn := range sc.engagement.ffnTensors {
		if !tn.IsNil() {
			sc.gf.BuildForwardExpand(tn)
		}
	}
	if caps != nil {
		for _, t := range caps.attnTensors {
			if !t.IsNil() {
				sc.gf.BuildForwardExpand(t)
			}
		}
	}

	if !sc.sched.AllocGraph(sc.gf) {
		return nil, nil, fmt.Errorf("failed to allocate graph")
	}

	// Zero-fill SSM state tensors (appended to zeroFill by the SSM block builder).
	for _, t := range sc.zeroFill {
		zeros := make([]byte, t.Nbytes())
		ggml.TensorSetBytes(t, zeros, 0)
	}

	// Set inputs
	ggml.TensorSet(sc.inpTokens, unsafe.Pointer(&tokenIDs[0]), 0, sc.inpTokens.Nbytes())

	positions := make([]int32, sc.nTokens)
	for i := range positions {
		positions[i] = int32(i)
	}
	ggml.TensorSet(sc.inputs.InpPos, unsafe.Pointer(&positions[0]), 0, sc.inputs.InpPos.Nbytes())

	maskData := buildCausalMaskData(nil, sc.nTokens, sc.nTokens, 0, m.Def.Architecture.NonCausal)
	ggml.TensorSet(sc.inputs.InpMask, unsafe.Pointer(&maskData[0]), 0, sc.inputs.InpMask.Nbytes())

	if sc.hasSWA && sc.swaWindow > 0 {
		swaMaskData := buildSWAMaskData(nil, sc.nTokens, sc.nTokens, 0, sc.swaWindow)
		ggml.TensorSet(sc.inputs.InpMaskSWA, unsafe.Pointer(&swaMaskData[0]), 0, sc.inputs.InpMaskSWA.Nbytes())
	}

	// Execute
	status := sc.sched.Compute(sc.gf)
	if status != ggml.StatusSuccess {
		return nil, nil, fmt.Errorf("%w (status=%d)", ErrComputeFailed, status)
	}

	// Read engagement scalars
	sc.engagement.readResults()

	// Read captured attention weights before defers free the tensors.
	nLayers := m.Params.Ints["n_layers"]
	if caps != nil && caps.Flags&CaptureAttnWeights != 0 {
		caps.AttnWeights = make([][]float32, nLayers)
		for il, t := range caps.attnTensors {
			if t.IsNil() {
				continue
			}
			n := t.Nbytes() / 4
			data := make([]float32, n)
			ggml.TensorGet(t, unsafe.Pointer(&data[0]), 0, t.Nbytes())
			caps.AttnWeights[il] = data
		}
	}

	return readLogits(logits, sc.nVocab), sc.engagement, nil
}

// ForwardCached runs a cached forward pass, processing only new tokens.
//
// Metrics capture (ForwardCaptures) is intentionally absent here. Cached mode
// distributes computation across many separate calls (prefill + per-token decodes),
// so there is no single clean attention matrix to capture. Any research question
// answerable through a stateless pass should use ForwardStateless instead.
// Do not add capture support here without explicit research into cached-mode mechanics.
func (m *GenericModel) ForwardCached(gc *GenericCache, tokenIDs []int32, mask *CullingMask, flashAttn bool) ([]float32, error) {
	nEmbd := m.Params.Ints["n_embd"]
	rmsEps := m.Params.Floats["rms_eps"]
	nNew := int64(len(tokenIDs))
	seqPos := gc.SeqPos
	nKV := int64(seqPos) + nNew

	if int(nKV) > gc.MaxSeqLen {
		return nil, fmt.Errorf("cache overflow: %d > %d", nKV, gc.MaxSeqLen)
	}

	m.cachedCtx.Reset()
	gctx := m.cachedCtx

	m.cachedSched.Reset()
	sched := m.cachedSched

	// Inputs (only new tokens)
	inpTokens := ggml.NewTensor1D(gctx, ggml.TypeI32, nNew)
	ggml.SetInput(inpTokens)
	inpPos := ggml.NewTensor1D(gctx, ggml.TypeI32, nNew)
	ggml.SetInput(inpPos)
	inpMask := ggml.NewTensor2D(gctx, ggml.TypeF32, nKV, nNew)
	ggml.SetInput(inpMask)

	x := ggml.GetRows(gctx, m.MaskedGlobal(gctx, "token_embd"), inpTokens)

	// Embedding scaling: multiply by sqrt(n_embd)
	if m.Def.Architecture.EmbedScale {
		x = ggml.Scale(gctx, x, float32(math.Sqrt(float64(nEmbd))))
	}

	// SWA mask
	swaWindow, hasSWA := m.Params.Ints["sliding_window"]
	var inpMaskSWA ggml.Tensor
	if hasSWA && swaWindow > 0 {
		inpMaskSWA = ggml.NewTensor2D(gctx, ggml.TypeF32, nKV, nNew)
		ggml.SetInput(inpMaskSWA)
	}

	effectiveFlashAttn := flashAttn && !m.Def.Architecture.NoFlashAttn
	inputs := &GraphInputs{
		InpPos:     inpPos,
		InpMask:    inpMask,
		InpMaskSWA: inpMaskSWA,
		NTokens:    nNew,
		NKV:        nKV,
		SeqPos:     seqPos,
		SharedKV:   &SharedKVState{K: make(map[string]ggml.Tensor), V: make(map[string]ggml.Tensor)},
		FlashAttn:  effectiveFlashAttn,
	}

	// Per-layer embedding preparation
	perLayerEmbd := m.buildPerLayerEmbedSetup(gctx, inpTokens, x, nNew, rmsEps)

	// Build the graph before running layers so that BuildCached can emit in-graph
	// cpy ops for KV and SSM state writebacks directly via gf.BuildForwardExpand.
	gf := ggml.NewGraph(gctx, 16384)

	blkFn := func(gctx *ggml.GraphContext, cur ggml.Tensor,
		lt map[string]ggml.Tensor, il int, config map[string]any,
		inputs *GraphInputs) ggml.Tensor {
		return m.BlockBuilders[il].BuildCached(gctx, gf, cur, lt, m.Params, config, inputs,
			&gc.Layers[il])
	}
	x = m.runLayers(gctx, x, inputs, rmsEps, mask, blkFn, perLayerEmbd, nil)

	logits := m.buildLogits(gctx, x, nEmbd, nNew, rmsEps)

	gf.BuildForwardExpand(logits)

	if !sched.AllocGraph(gf) {
		return nil, fmt.Errorf("failed to allocate graph")
	}

	// Set inputs
	ggml.TensorSet(inpTokens, unsafe.Pointer(&tokenIDs[0]), 0, inpTokens.Nbytes())

	positions := make([]int32, nNew)
	for i := range positions {
		positions[i] = int32(seqPos + int(i))
	}
	ggml.TensorSet(inpPos, unsafe.Pointer(&positions[0]), 0, inpPos.Nbytes())

	maskData := buildCausalMaskData(gc.maskBuf, nNew, nKV, seqPos, m.Def.Architecture.NonCausal)
	ggml.TensorSet(inpMask, unsafe.Pointer(&maskData[0]), 0, inpMask.Nbytes())

	if hasSWA && swaWindow > 0 {
		swaMaskData := buildSWAMaskData(gc.swaMaskBuf, nNew, nKV, seqPos, swaWindow)
		ggml.TensorSet(inpMaskSWA, unsafe.Pointer(&swaMaskData[0]), 0, inpMaskSWA.Nbytes())
	}

	status := sched.Compute(gf)
	if status != ggml.StatusSuccess {
		return nil, fmt.Errorf("%w (status=%d)", ErrComputeFailed, status)
	}

	gc.SeqPos += int(nNew)

	return readLogitsInto(m.logitBuf, logits), nil
}

// buildPerLayerEmbedSetup prepares the combined per-layer embedding tensor
// used by perLayerEmbedInject inside the layer loop. Returns NilTensor if unused.
func (m *GenericModel) buildPerLayerEmbedSetup(gctx *ggml.GraphContext,
	inpTokens, x ggml.Tensor, nTokens int64, rmsEps float32) ggml.Tensor {
	tokEmbdPL := m.MaskedGlobal(gctx, "tok_embd_per_layer")
	if tokEmbdPL.IsNil() {
		return ggml.NilTensor()
	}

	nLayers := int64(m.Params.Ints["n_layers"])
	nEmbdPL := int64(m.Params.Ints["n_embd_per_layer"])
	nEmbd := int64(m.Params.Ints["n_embd"])

	// 1. Per-layer token embeddings: [n_embd_per_layer * n_layers, n_tokens]
	inp := ggml.GetRows(gctx, tokEmbdPL, inpTokens)
	inp = ggml.Reshape3D(gctx, inp, nEmbdPL, nLayers, nTokens)
	inp = ggml.Scale(gctx, inp, float32(math.Sqrt(float64(nEmbdPL))))

	// 2. Project model embeddings → per-layer space
	proj := ggml.MulMat(gctx, m.MaskedGlobal(gctx, "per_layer_model_proj"), x)
	proj = ggml.Scale(gctx, proj, float32(1.0/math.Sqrt(float64(nEmbd))))
	proj = ggml.Reshape3D(gctx, proj, nEmbdPL, nLayers, nTokens)
	proj = rmsNormApply(gctx, proj, m.MaskedGlobal(gctx, "per_layer_proj_norm"), rmsEps)

	// 3. Combine and scale by 1/sqrt(2)
	inp = ggml.Add(gctx, proj, inp)
	inp = ggml.Scale(gctx, inp, float32(1.0/math.Sqrt(2.0)))

	// 4. Permute to [n_embd_per_layer, n_tokens, n_layers] for per-layer slicing
	inp = ggml.Cont(gctx, ggml.Permute(gctx, inp, 0, 2, 1, 3))
	return inp
}

// perLayerEmbedInject applies gated per-layer embedding injection for one layer.
// perLayerEmbd shape: [n_embd_per_layer, n_tokens, n_layers].
func (m *GenericModel) perLayerEmbedInject(gctx *ggml.GraphContext, x ggml.Tensor,
	lt map[string]ggml.Tensor, il int, perLayerEmbd ggml.Tensor, rmsEps float32) ggml.Tensor {

	if lt["pe_proj"].IsNil() {
		return x
	}

	nEmbdPL := perLayerEmbd.Ne(0)
	nTokens := perLayerEmbd.Ne(1)

	// Slice this layer's embedding: [n_embd_per_layer, n_tokens]
	nb1 := perLayerEmbd.Nb(1)
	offset := il * int(nTokens) * nb1
	peSlice := ggml.View2D(gctx, perLayerEmbd, nEmbdPL, nTokens, nb1, offset)

	// gate = gelu(pe_inp_gate @ x) → [n_embd_per_layer, n_tokens]
	gate := ggml.MulMat(gctx, lt["pe_inp_gate"], x)
	gate = ggml.Gelu(gctx, gate)

	// Gated element-wise multiplication
	gate = ggml.Mul(gctx, gate, peSlice)

	out := ggml.MulMat(gctx, lt["pe_proj"], gate)

	// Post-projection norm is optional.
	if !lt["pe_post_norm"].IsNil() {
		out = rmsNormApply(gctx, out, lt["pe_post_norm"], rmsEps)
	}

	return ggml.Add(gctx, x, out)
}
