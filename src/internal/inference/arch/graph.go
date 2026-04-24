package arch

import (
	"fmt"
	"math"
	"unsafe"

	ggml "inference-lab-bench/internal/ggml"
	"inference-lab-bench/internal/log"
)

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
// select BuildStateless or BuildCached. Receives the block-specific weight map
// (not common or FFN weights). The surrounding layer loop (runLayers)
// enforces the residual stream invariant: norm → block → residual → FFN → residual.
type blockFunc func(gctx *ggml.GraphContext, cur ggml.Tensor,
	weights map[string]ggml.Tensor, il int, config map[string]any,
	inputs *GraphInputs) ggml.Tensor

// runLayers executes the per-layer forward pass: for each layer, apply attention/SSM
// block (via blkFn) with residual, then FFN with residual. Layers without attn_norm
// skip the block entirely (pure-FFN layer); layers without FFN tensors skip FFN
// (recurrent-only layer). Both branches are driven by arch-level weight presence.
//
// Extension pattern: each optional per-layer behavior below is gated by a
// weight- or param-presence check — NEVER by architecture name. This is the
// data-driven invariant from AGENTS.md. The current set is:
//
//   - attn_norm presence      → gates the attention/recurrent block + residual
//   - attn_post_norm presence → post-attention norm (inside the block branch)
//   - pe_inp_gate presence    → Gemma4 per-layer embedding injection
//   - layer_output_scale      → Gemma4 per-layer residual scale
//
// Before adding a 5th hook, consider replacing this implicit nil-check
// extension mechanism with an explicit per-layer hook registry — at that
// scale the readability cost of another nil-check outweighs the benefit of
// inlining another feature. See AGENTS.md §"Phase 4: Extend graph.go".
func (m *GenericModel) runLayers(gctx *ggml.GraphContext, x ggml.Tensor,
	inputs *GraphInputs, rmsEps float32,
	blkFn blockFunc, perLayerEmbd ggml.Tensor) ggml.Tensor {
	for il := range m.Params.Ints[ParamNLayers] {
		lt := m.Store.Layer(il)
		if lt == nil {
			log.Error("runLayers: layer %d out of bounds — skipping", il)
			continue
		}
		blockName := m.LayerBlockNames[il]
		blockDef := m.Def.Blocks[blockName]
		inputs.CurrentLayer = il

		if !lt.Common[WeightAttnNorm].IsNil() {
			cur := rmsNormApply(gctx, x, lt.Common[WeightAttnNorm], rmsEps)
			cur = blkFn(gctx, cur, lt.Block, il, blockDef.Config, inputs)
			if !lt.Common[WeightAttnPostNorm].IsNil() {
				cur = rmsNormApply(gctx, cur, lt.Common[WeightAttnPostNorm], rmsEps)
			}
			x = ggml.Add(gctx, x, cur)
		}

		x = m.buildFFNBlock(gctx, x, lt, il, rmsEps)

		// Per-layer embedding injection (Gemma 4)
		if !perLayerEmbd.IsNil() && !lt.Common[WeightPEInpGate].IsNil() {
			x = m.perLayerEmbedInject(gctx, x, lt.Common, il, perLayerEmbd, rmsEps)
		}

		// Layer output scaling
		if !lt.Common[WeightLayerOutputScale].IsNil() {
			x = ggml.Mul(gctx, x, lt.Common[WeightLayerOutputScale])
		}
	}
	return x
}

// buildFFNBlock applies norm, builds FFN, and adds residual.
// Returns x unchanged when the layer has no FFN tensors (recurrent-only layer).
func (m *GenericModel) buildFFNBlock(ctx *ggml.GraphContext, x ggml.Tensor,
	lt *LayerTensors, il int, rmsEps float32) ggml.Tensor {
	ffnInp := x
	if anyNonNil(lt.FFN) {
		ffnConfig := m.FFNConfigs[il]
		// Self-normed builders manage their own pre/post norms internally
		// (e.g. MoE with parallel shared+expert paths that need separate norms).
		if configBoolOr(ffnConfig, ConfigSelfNormed, false) {
			ffnOut := m.FFNBuilders[il].BuildFFN(ctx, x, lt.FFN, m.Params, ffnConfig)
			// Self-normed builders handle internal pre/post norms for each path,
			// but the common post-FFN norm still wraps the combined output.
			if !lt.Common[WeightFFNPostNorm].IsNil() {
				ffnOut = rmsNormApply(ctx, ffnOut, lt.Common[WeightFFNPostNorm], rmsEps)
			}
			return ggml.Add(ctx, ffnInp, ffnOut)
		}
		xn2 := rmsNormApply(ctx, x, lt.Common[WeightFFNNorm], rmsEps)
		ffnOut := m.FFNBuilders[il].BuildFFN(ctx, xn2, lt.FFN, m.Params, ffnConfig)
		// Optional post-FFN norm
		if !lt.Common[WeightFFNPostNorm].IsNil() {
			ffnOut = rmsNormApply(ctx, ffnOut, lt.Common[WeightFFNPostNorm], rmsEps)
		}
		return ggml.Add(ctx, ffnInp, ffnOut)
	}
	return x
}

// anyNonNil returns true if any tensor in the map is non-nil.
// Used to detect whether a layer has any FFN weights (pure-recurrent
// layers define no FFN tensors).
func anyNonNil(weights map[string]ggml.Tensor) bool {
	for _, v := range weights {
		if !v.IsNil() {
			return true
		}
	}
	return false
}


// buildLogits extracts the last token's embedding, applies final norm, and projects
// through the LM head. Returns the logits tensor (already marked as output).
func (m *GenericModel) buildLogits(gctx *ggml.GraphContext, x ggml.Tensor,
	nEmbd int, nSeq int64, rmsEps float32) ggml.Tensor {
	last := ggml.View2D(gctx, x, int64(nEmbd), 1, x.Nb(1), int(nSeq-1)*x.Nb(1))
	xn := rmsNormApply(gctx, last, m.Store.Global(WeightOutputNorm), rmsEps)
	logits := ggml.MulMat(gctx, m.Store.Global(WeightOutput), xn)

	// Logit softcapping: cap * tanh(logits / cap)
	if cap, ok := m.Params.Floats[ParamLogitSoftcap]; ok && cap > 0 {
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
	xn := rmsNormApply(gctx, x, m.Store.Global(WeightOutputNorm), rmsEps)
	logits := ggml.MulMat(gctx, m.Store.Global(WeightOutput), xn)

	// Logit softcapping: cap * tanh(logits / cap)
	if cap, ok := m.Params.Floats[ParamLogitSoftcap]; ok && cap > 0 {
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
	gctx      *ggml.GraphContext
	sched     *ggml.Sched
	gf        *ggml.Graph
	x         ggml.Tensor
	inputs    *GraphInputs
	inpTokens ggml.Tensor // input token ID tensor — needed for TensorSet after AllocGraph
	nEmbd     int
	rmsEps    float32
	nVocab    int
	nTokens   int64
	hasSWA    bool
	swaWindow int
	zeroFill  []ggml.Tensor
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
//	caps — ForwardCaptures or nil (ForwardStateless path only)
func (m *GenericModel) forwardStatelessCore(
	tokenIDs []int32,
	flashAttn bool,
	caps *ForwardCaptures,
) (*statelessCtx, error) {
	nLayers := m.Params.Ints[ParamNLayers]
	nEmbd := m.Params.Ints[ParamNEmbd]
	nVocab := m.Params.Ints[ParamNVocab]
	rmsEps := m.Params.Floats[ParamRMSEps]
	nTokens := int64(len(tokenIDs))

	ctxSize := graphCtxSize()
	gctx := ggml.NewGraphContext(ctxSize, ggml.AllocPermDisallow)
	if gctx == nil {
		return nil, fmt.Errorf("failed to create graph context")
	}

	sched := ggml.NewSched(m.Store.GPU, m.Store.CPU, maxGraphNodes)
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

	x := ggml.GetRows(gctx, m.Store.Global(WeightTokenEmbd), inpTokens)

	if m.Def.Architecture.EmbedScale {
		x = ggml.Scale(gctx, x, float32(math.Sqrt(float64(nEmbd))))
	}

	swaWindow, hasSWA := m.Params.Ints[ParamSlidingWindow]
	var inpMaskSWA ggml.Tensor
	if hasSWA && swaWindow > 0 {
		inpMaskSWA = ggml.NewTensor2D(gctx, ggml.TypeF32, nTokens, nTokens)
		ggml.SetInput(inpMaskSWA)
	}

	if caps != nil && caps.Flags&CaptureAttnWeights != 0 {
		caps.attnTensors = make([]ggml.Tensor, nLayers)
		caps.NHeads = int64(m.Params.Ints[ParamNHeads])
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

	var zeroFill []ggml.Tensor
	blkFn := func(gctx *ggml.GraphContext, cur ggml.Tensor,
		weights map[string]ggml.Tensor, il int, config map[string]any,
		inputs *GraphInputs) ggml.Tensor {
		if inputs.Captures != nil {
			inputs.Captures.currentLayer = il
		}
		return m.BlockBuilders[il].BuildStateless(gctx, cur, weights, m.Params, config, inputs, &zeroFill)
	}
	x = m.runLayers(gctx, x, inputs, rmsEps, blkFn, perLayerEmbd)

	gf := ggml.NewGraph(gctx, maxGraphNodes)

	log.Debug("stateless graph ctx: %d / %d bytes used", gctx.UsedMem(), ctxSize)

	return &statelessCtx{
		gctx:      gctx,
		sched:     sched,
		gf:        gf,
		x:         x,
		inputs:    inputs,
		inpTokens: inpTokens,
		nEmbd:     nEmbd,
		rmsEps:    rmsEps,
		nVocab:    nVocab,
		nTokens:   nTokens,
		hasSWA:    hasSWA,
		swaWindow: swaWindow,
		zeroFill:  zeroFill,
	}, nil
}

// ForwardStatelessAllLogits runs a full-sequence forward pass and returns logits for
// all token positions. Returns a slice of length nVocab*nTokens, row-major by position:
// position p occupies allLogits[p*nVocab : (p+1)*nVocab].
// No captures are collected — this is intended for diffusion generation.
func (m *GenericModel) ForwardStatelessAllLogits(tokenIDs []int32, flashAttn bool) ([]float32, error) {
	sc, err := m.forwardStatelessCore(tokenIDs, flashAttn, nil)
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
		if t.IsNil() {
			log.Error("zeroFill: nil tensor in zero-fill list — uninitialized GPU memory")
			continue
		}
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
func (m *GenericModel) ForwardStateless(tokenIDs []int32, caps *ForwardCaptures, flashAttn bool) ([]float32, error) {
	sc, err := m.forwardStatelessCore(tokenIDs, flashAttn, caps)
	if err != nil {
		return nil, err
	}
	defer sc.Free()

	logits := m.buildLogits(sc.gctx, sc.x, sc.nEmbd, sc.nTokens, sc.rmsEps)

	sc.gf.BuildForwardExpand(logits)
	if caps != nil {
		for _, t := range caps.attnTensors {
			if !t.IsNil() {
				sc.gf.BuildForwardExpand(t)
			}
		}
	}

	if !sc.sched.AllocGraph(sc.gf) {
		return nil, fmt.Errorf("failed to allocate graph")
	}

	// Zero-fill SSM state tensors (appended to zeroFill by the SSM block builder).
	for _, t := range sc.zeroFill {
		if t.IsNil() {
			log.Error("zeroFill: nil tensor in zero-fill list — uninitialized GPU memory")
			continue
		}
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
		return nil, fmt.Errorf("%w (status=%d)", ErrComputeFailed, status)
	}

	// Read captured attention weights before defers free the tensors.
	nLayers := m.Params.Ints[ParamNLayers]
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

	return readLogits(logits, sc.nVocab), nil
}

// ForwardCached runs a cached forward pass, processing only new tokens.
//
// Metrics capture (ForwardCaptures) is intentionally absent here. Cached mode
// distributes computation across many separate calls (prefill + per-token decodes),
// so there is no single clean attention matrix to capture. Any research question
// answerable through a stateless pass should use ForwardStateless instead.
// Do not add capture support here without explicit research into cached-mode mechanics.
func (m *GenericModel) ForwardCached(gc *GenericCache, tokenIDs []int32, flashAttn bool) ([]float32, error) {
	nEmbd := m.Params.Ints[ParamNEmbd]
	rmsEps := m.Params.Floats[ParamRMSEps]
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

	x := ggml.GetRows(gctx, m.Store.Global(WeightTokenEmbd), inpTokens)

	// Embedding scaling: multiply by sqrt(n_embd)
	if m.Def.Architecture.EmbedScale {
		x = ggml.Scale(gctx, x, float32(math.Sqrt(float64(nEmbd))))
	}

	// SWA mask
	swaWindow, hasSWA := m.Params.Ints[ParamSlidingWindow]
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
	gf := ggml.NewGraph(gctx, maxGraphNodes)

	blkFn := func(gctx *ggml.GraphContext, cur ggml.Tensor,
		weights map[string]ggml.Tensor, il int, config map[string]any,
		inputs *GraphInputs) ggml.Tensor {
		return m.BlockBuilders[il].BuildCached(gctx, gf, cur, weights, m.Params, config, inputs,
			&gc.Layers[il])
	}
	x = m.runLayers(gctx, x, inputs, rmsEps, blkFn, perLayerEmbd)

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
	tokEmbdPL := m.Store.Global(WeightTokEmbdPerLayer)
	if tokEmbdPL.IsNil() {
		return ggml.NilTensor()
	}

	nLayers := int64(m.Params.Ints[ParamNLayers])
	nEmbdPL := int64(m.Params.Ints[ParamNEmbdPerLayer])
	nEmbd := int64(m.Params.Ints[ParamNEmbd])

	// 1. Per-layer token embeddings: [n_embd_per_layer * n_layers, n_tokens]
	inp := ggml.GetRows(gctx, tokEmbdPL, inpTokens)
	inp = ggml.Reshape3D(gctx, inp, nEmbdPL, nLayers, nTokens)
	inp = ggml.Scale(gctx, inp, float32(math.Sqrt(float64(nEmbdPL))))

	// 2. Project model embeddings → per-layer space
	proj := ggml.MulMat(gctx, m.Store.Global(WeightPerLayerModelProj), x)
	proj = ggml.Scale(gctx, proj, float32(1.0/math.Sqrt(float64(nEmbd))))
	proj = ggml.Reshape3D(gctx, proj, nEmbdPL, nLayers, nTokens)
	proj = rmsNormApply(gctx, proj, m.Store.Global(WeightPerLayerProjNorm), rmsEps)

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

	if lt[WeightPEProj].IsNil() {
		return x
	}

	nEmbdPL := perLayerEmbd.Ne(0)
	nTokens := perLayerEmbd.Ne(1)

	// Slice this layer's embedding: [n_embd_per_layer, n_tokens]
	nb1 := perLayerEmbd.Nb(1)
	offset := il * int(nTokens) * nb1
	peSlice := ggml.View2D(gctx, perLayerEmbd, nEmbdPL, nTokens, nb1, offset)

	// gate = gelu(pe_inp_gate @ x) → [n_embd_per_layer, n_tokens]
	gate := ggml.MulMat(gctx, lt[WeightPEInpGate], x)
	gate = ggml.Gelu(gctx, gate)

	// Gated element-wise multiplication
	gate = ggml.Mul(gctx, gate, peSlice)

	out := ggml.MulMat(gctx, lt[WeightPEProj], gate)

	// Post-projection norm is optional.
	if !lt[WeightPEPostNorm].IsNil() {
		out = rmsNormApply(gctx, out, lt[WeightPEPostNorm], rmsEps)
	}

	return ggml.Add(gctx, x, out)
}
