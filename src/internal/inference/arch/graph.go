package arch

import (
	"fmt"
	"math"
	"os"
	"strings"
	"unsafe"

	ggml "inference-lab-bench/internal/inference/ggml"
)

// buildCausalMaskData generates the float32 mask data for causal attention.
// Positions where key > query get -Inf. Returns all zeros if nonCausal.
func buildCausalMaskData(nQuery, nKV int64, startPos int, nonCausal bool) []float32 {
	data := make([]float32, nQuery*nKV)
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
func buildSWAMaskData(nQuery, nKV int64, startPos, window int) []float32 {
	data := make([]float32, nQuery*nKV)
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

// nodesPerLayer estimates ggml graph nodes per transformer layer. The +1 in
// graphCtxSize accounts for global ops (embedding lookup, final norm, LM head).
const nodesPerLayer = 128

func graphCtxSize(nLayers int) int {
	return 4*1024*1024 + ggml.TensorOverhead()*(nLayers+1)*nodesPerLayer
}

// blockFunc is the per-layer block builder call, injected by the caller to
// select BuildStateless or BuildCached. The surrounding layer loop (runLayers)
// enforces the residual stream invariant: norm → block → residual → FFN → residual.
type blockFunc func(gctx *ggml.GraphContext, cur ggml.Tensor,
	lt map[string]ggml.Tensor, il int, config map[string]any,
	inputs *GraphInputs) ggml.Tensor

// runLayers executes the per-layer forward pass: for each layer, apply attention/SSM
// block (via blkFn) with residual, then FFN with residual. Layers with nil attn_norm
// skip the block; layers with all FFN tensors nil skip FFN.
func (m *GenericModel) runLayers(gctx *ggml.GraphContext, x ggml.Tensor,
	inputs *GraphInputs, rmsEps float32, blkFn blockFunc,
	perLayerEmbd ggml.Tensor) ggml.Tensor {
	for il := range m.Params.Ints["n_layers"] {
		lt := m.Store.Layer(il)
		blockName := m.LayerBlockNames[il]
		blockDef := m.Def.Blocks[blockName]
		inputs.CurrentLayer = il
		dbg0 := il == 0 && inputs.DebugTensors != nil

		if !lt["attn_norm"].IsNil() {
			cur := rmsNormApply(gctx, x, lt["attn_norm"], rmsEps)
			if dbg0 {
				debugCapture(inputs, "attn_norm_0", cur)
			}
			cur = blkFn(gctx, cur, lt, il, blockDef.Config, inputs)
			if dbg0 {
				debugCapture(inputs, "attn_out_0", cur)
			}
			// Optional post-attention norm
			if !lt["attn_post_norm"].IsNil() {
				cur = rmsNormApply(gctx, cur, lt["attn_post_norm"], rmsEps)
				if dbg0 {
					debugCapture(inputs, "post_attn_norm_0", cur)
				}
			}
			x = ggml.Add(gctx, x, cur)
			if dbg0 {
				debugCapture(inputs, "attn_residual_0", x)
			}
		}

		x = m.buildFFNBlock(gctx, x, lt, il, rmsEps)
		if dbg0 {
			debugCapture(inputs, "ffn_residual_0", x)
		}

		// Per-layer embedding injection (Gemma 4)
		if !perLayerEmbd.IsNil() && !lt["pe_inp_gate"].IsNil() {
			x = m.perLayerEmbedInject(gctx, x, lt, il, perLayerEmbd, rmsEps)
			if dbg0 {
				debugCapture(inputs, "pe_inject_0", x)
			}
		}

		// Layer output scaling
		if !lt["layer_output_scale"].IsNil() {
			x = ggml.Mul(gctx, x, lt["layer_output_scale"])
			if dbg0 {
				debugCapture(inputs, "layer_scaled_0", x)
			}
		}
	}
	return x
}

// buildFFNBlock extracts FFN weights, applies norm, builds FFN, and adds residual.
// Returns x unchanged if all FFN tensors are nil.
func (m *GenericModel) buildFFNBlock(ctx *ggml.GraphContext, x ggml.Tensor,
	lt map[string]ggml.Tensor, il int, rmsEps float32) ggml.Tensor {
	ffnInp := x
	ffnWeights := make(map[string]ggml.Tensor)
	for k, v := range lt {
		if after, ok := strings.CutPrefix(k, "ffn_"); ok {
			ffnWeights[after] = v
		}
	}
	if anyNonNil(ffnWeights) {
		xn2 := rmsNormApply(ctx, x, ffnNorm(lt), rmsEps)
		ffnOut := m.FFNBuilders[il].BuildFFN(ctx, xn2, ffnWeights, m.Params)
		// Optional post-FFN norm
		if !lt["ffn_post_norm"].IsNil() {
			ffnOut = rmsNormApply(ctx, ffnOut, lt["ffn_post_norm"], rmsEps)
		}
		return ggml.Add(ctx, ffnInp, ffnOut)
	}
	return x
}

// anyNonNil returns true if any tensor in the map is non-nil.
func anyNonNil(weights map[string]ggml.Tensor) bool {
	for _, v := range weights {
		if !v.IsNil() {
			return true
		}
	}
	return false
}

// ffnNorm returns the pre-FFN norm weight from the layer tensor map.
func ffnNorm(lt map[string]ggml.Tensor) ggml.Tensor {
	return lt["ffn_norm"]
}

// buildLogits extracts the last token's embedding, applies final norm, and projects
// through the LM head. Returns the logits tensor (already marked as output).
func (m *GenericModel) buildLogits(gctx *ggml.GraphContext, x ggml.Tensor,
	nEmbd int, nSeq int64, rmsEps float32) ggml.Tensor {
	last := ggml.View2D(gctx, x, int64(nEmbd), 1, x.Nb(1), int(nSeq-1)*x.Nb(1))
	xn := rmsNormApply(gctx, last, m.Store.Global("output_norm"), rmsEps)
	logits := ggml.MulMat(gctx, m.Store.Global("output"), xn)

	// Logit softcapping: cap * tanh(logits / cap)
	if cap, ok := m.Params.Floats["logit_softcapping"]; ok && cap > 0 {
		logits = ggml.Scale(gctx, logits, 1.0/cap)
		logits = ggml.Tanh(gctx, logits)
		logits = ggml.Scale(gctx, logits, cap)
	}

	ggml.SetOutput(logits)
	return logits
}

// readLogits copies the logits tensor from GPU to a float32 slice.
func readLogits(logits ggml.Tensor, nVocab int) []float32 {
	result := make([]float32, nVocab)
	ggml.TensorGet(logits, unsafe.Pointer(&result[0]), 0, logits.Nbytes())
	return result
}

// ForwardStateless runs a full-sequence forward pass without caching.
func (m *GenericModel) ForwardStateless(tokenIDs []int32) ([]float32, error) {
	nLayers := m.Params.Ints["n_layers"]
	nEmbd := m.Params.Ints["n_embd"]
	nVocab := m.Params.Ints["n_vocab"]
	rmsEps := m.Params.Floats["rms_eps"]
	nTokens := int64(len(tokenIDs))

	gctx := ggml.NewGraphContext(graphCtxSize(nLayers))
	if gctx == nil {
		return nil, fmt.Errorf("failed to create graph context")
	}
	defer gctx.Free()

	sched := ggml.NewSched(m.Store.GPU, m.Store.CPU, 16384)
	if sched == nil {
		return nil, fmt.Errorf("failed to create scheduler")
	}
	defer sched.Free()

	// Inputs
	inpTokens := ggml.NewTensor1D(gctx, ggml.TypeI32, nTokens)
	ggml.SetInput(inpTokens)
	inpPos := ggml.NewTensor1D(gctx, ggml.TypeI32, nTokens)
	ggml.SetInput(inpPos)
	inpMask := ggml.NewTensor2D(gctx, ggml.TypeF32, nTokens, nTokens)
	ggml.SetInput(inpMask)

	x := ggml.GetRows(gctx, m.Store.Global("token_embd"), inpTokens)

	// Embedding scaling: multiply by sqrt(n_embd)
	if m.Def.Architecture.EmbedScale {
		x = ggml.Scale(gctx, x, float32(math.Sqrt(float64(nEmbd))))
	}

	// SWA mask (only if sliding_window param exists)
	swaWindow, hasSWA := m.Params.Ints["sliding_window"]
	var inpMaskSWA ggml.Tensor
	if hasSWA && swaWindow > 0 {
		inpMaskSWA = ggml.NewTensor2D(gctx, ggml.TypeF32, nTokens, nTokens)
		ggml.SetInput(inpMaskSWA)
	}

	var debugTensors []DebugTensor
	if os.Getenv("DUMP_TENSORS") == "1" {
		debugTensors = make([]DebugTensor, 0, 64)
	}

	inputs := &GraphInputs{
		InpPos:     inpPos,
		InpMask:    inpMask,
		InpMaskSWA: inpMaskSWA,
		NTokens:    nTokens,
		NKV:        nTokens,
		SeqPos:     0,
		SharedKV:   &SharedKVState{K: make(map[string]ggml.Tensor), V: make(map[string]ggml.Tensor)},
	}
	if debugTensors != nil {
		inputs.DebugTensors = &debugTensors
	}

	debugCapture(inputs, "embd_scaled", x)

	// Per-layer embedding preparation
	perLayerEmbd := m.buildPerLayerEmbedSetup(gctx, inpTokens, x, nTokens, rmsEps)

	var zeroFill []ggml.Tensor

	blkFn := func(gctx *ggml.GraphContext, cur ggml.Tensor,
		lt map[string]ggml.Tensor, il int, config map[string]any,
		inputs *GraphInputs) ggml.Tensor {
		return m.BlockBuilders[il].BuildStateless(gctx, cur, lt, m.Params, config, inputs, &zeroFill)
	}
	x = m.runLayers(gctx, x, inputs, rmsEps, blkFn, perLayerEmbd)

	logits := m.buildLogits(gctx, x, nEmbd, nTokens, rmsEps)

	// Build graph
	gf := ggml.NewGraph(gctx, 16384)
	gf.BuildForwardExpand(logits)

	if !sched.AllocGraph(gf) {
		return nil, fmt.Errorf("failed to allocate graph")
	}

	// Zero-fill SSM state tensors.
	for _, t := range zeroFill {
		zeros := make([]byte, t.Nbytes())
		ggml.TensorSetBytes(t, zeros, 0)
	}

	// Set inputs
	ggml.TensorSet(inpTokens, unsafe.Pointer(&tokenIDs[0]), 0, inpTokens.Nbytes())

	positions := make([]int32, nTokens)
	for i := range positions {
		positions[i] = int32(i)
	}
	ggml.TensorSet(inpPos, unsafe.Pointer(&positions[0]), 0, inpPos.Nbytes())

	maskData := buildCausalMaskData(nTokens, nTokens, 0, m.Def.Architecture.NonCausal)
	ggml.TensorSet(inpMask, unsafe.Pointer(&maskData[0]), 0, inpMask.Nbytes())

	if hasSWA && swaWindow > 0 {
		swaMaskData := buildSWAMaskData(nTokens, nTokens, 0, swaWindow)
		ggml.TensorSet(inpMaskSWA, unsafe.Pointer(&swaMaskData[0]), 0, inpMaskSWA.Nbytes())
	}

	// Execute
	status := sched.Compute(gf)
	if status != ggml.StatusSuccess {
		return nil, fmt.Errorf("graph compute failed: %d", status)
	}

	if len(debugTensors) > 0 {
		DumpDebugTensors(debugTensors)
	}

	return readLogits(logits, nVocab), nil
}

// ForwardCached runs a cached forward pass, processing only new tokens.
func (m *GenericModel) ForwardCached(gc *GenericCache, tokenIDs []int32) ([]float32, error) {
	nLayers := m.Params.Ints["n_layers"]
	nEmbd := m.Params.Ints["n_embd"]
	nVocab := m.Params.Ints["n_vocab"]
	rmsEps := m.Params.Floats["rms_eps"]
	nNew := int64(len(tokenIDs))
	seqPos := gc.SeqPos
	nKV := int64(seqPos) + nNew

	if int(nKV) > gc.MaxSeqLen {
		return nil, fmt.Errorf("cache overflow: %d > %d", nKV, gc.MaxSeqLen)
	}

	gctx := ggml.NewGraphContext(graphCtxSize(nLayers))
	if gctx == nil {
		return nil, fmt.Errorf("failed to create graph context")
	}
	defer gctx.Free()

	sched := ggml.NewSched(m.Store.GPU, m.Store.CPU, 16384)
	if sched == nil {
		return nil, fmt.Errorf("failed to create scheduler")
	}
	defer sched.Free()

	// Inputs (only new tokens)
	inpTokens := ggml.NewTensor1D(gctx, ggml.TypeI32, nNew)
	ggml.SetInput(inpTokens)
	inpPos := ggml.NewTensor1D(gctx, ggml.TypeI32, nNew)
	ggml.SetInput(inpPos)
	inpMask := ggml.NewTensor2D(gctx, ggml.TypeF32, nKV, nNew)
	ggml.SetInput(inpMask)

	x := ggml.GetRows(gctx, m.Store.Global("token_embd"), inpTokens)

	// Embedding scaling
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

	var debugTensorsCached []DebugTensor
	if os.Getenv("DUMP_TENSORS") == "1" {
		debugTensorsCached = make([]DebugTensor, 0, 64)
	}

	inputs := &GraphInputs{
		InpPos:     inpPos,
		InpMask:    inpMask,
		InpMaskSWA: inpMaskSWA,
		NTokens:    nNew,
		NKV:        nKV,
		SeqPos:     seqPos,
		SharedKV:   &SharedKVState{K: make(map[string]ggml.Tensor), V: make(map[string]ggml.Tensor)},
	}
	if debugTensorsCached != nil {
		inputs.DebugTensors = &debugTensorsCached
	}

	debugCapture(inputs, "embd_scaled", x)

	// Per-layer embedding preparation
	perLayerEmbd := m.buildPerLayerEmbedSetup(gctx, inpTokens, x, nNew, rmsEps)

	var writebacks []CacheWriteback

	blkFn := func(gctx *ggml.GraphContext, cur ggml.Tensor,
		lt map[string]ggml.Tensor, il int, config map[string]any,
		inputs *GraphInputs) ggml.Tensor {
		return m.BlockBuilders[il].BuildCached(gctx, cur, lt, m.Params, config, inputs,
			&gc.Layers[il], &writebacks)
	}
	x = m.runLayers(gctx, x, inputs, rmsEps, blkFn, perLayerEmbd)

	logits := m.buildLogits(gctx, x, nEmbd, nNew, rmsEps)

	gf := ggml.NewGraph(gctx, 16384)
	gf.BuildForwardExpand(logits)
	for i := range writebacks {
		gf.BuildForwardExpand(writebacks[i].Src)
	}

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

	maskData := buildCausalMaskData(nNew, nKV, seqPos, m.Def.Architecture.NonCausal)
	ggml.TensorSet(inpMask, unsafe.Pointer(&maskData[0]), 0, inpMask.Nbytes())

	if hasSWA && swaWindow > 0 {
		swaMaskData := buildSWAMaskData(nNew, nKV, seqPos, swaWindow)
		ggml.TensorSet(inpMaskSWA, unsafe.Pointer(&swaMaskData[0]), 0, inpMaskSWA.Nbytes())
	}

	status := sched.Compute(gf)
	if status != ggml.StatusSuccess {
		return nil, fmt.Errorf("graph compute failed: %d", status)
	}

	if len(debugTensorsCached) > 0 {
		fmt.Fprintf(os.Stderr, "[DUMP-CACHED seqPos=%d nNew=%d nKV=%d]\n", seqPos, nNew, nKV)
		DumpDebugTensors(debugTensorsCached)
	}

	// Post-compute writebacks (per-head strided copy)
	for _, wb := range writebacks {
		for h := range wb.NHeads {
			tmp := make([]byte, wb.HeadBytes)
			ggml.TensorGet(wb.Src, unsafe.Pointer(&tmp[0]), h*wb.HeadSrc, wb.HeadBytes)
			ggml.TensorSet(wb.Dst, unsafe.Pointer(&tmp[0]), h*wb.HeadDst+wb.HeadOffset, wb.HeadBytes)
		}
	}

	gc.SeqPos += int(nNew)

	return readLogits(logits, nVocab), nil
}

// buildPerLayerEmbedSetup prepares the combined per-layer embedding tensor
// used by perLayerEmbedInject inside the layer loop. Returns NilTensor if unused.
func (m *GenericModel) buildPerLayerEmbedSetup(gctx *ggml.GraphContext,
	inpTokens, x ggml.Tensor, nTokens int64, rmsEps float32) ggml.Tensor {
	tokEmbdPL := m.Store.Global("tok_embd_per_layer")
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
	proj := ggml.MulMat(gctx, m.Store.Global("per_layer_model_proj"), x)
	proj = ggml.Scale(gctx, proj, float32(1.0/math.Sqrt(float64(nEmbd))))
	proj = ggml.Reshape3D(gctx, proj, nEmbdPL, nLayers, nTokens)
	proj = rmsNormApply(gctx, proj, m.Store.Global("per_layer_proj_norm"), rmsEps)

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

	nEmbdPL := perLayerEmbd.Ne(0)
	nTokens := perLayerEmbd.Ne(1)

	// gate = gelu(pe_inp_gate @ x) → [n_embd_per_layer, n_tokens]
	gate := ggml.MulMat(gctx, lt["pe_inp_gate"], x)
	gate = ggml.Gelu(gctx, gate)

	// Slice this layer's embedding: [n_embd_per_layer, n_tokens]
	nb1 := int(nEmbdPL) * 4 // F32 element size
	offset := il * int(nEmbdPL*nTokens) * 4
	peSlice := ggml.View2D(gctx, perLayerEmbd, nEmbdPL, nTokens, nb1, offset)

	// Gated element-wise multiplication
	gate = ggml.Mul(gctx, gate, peSlice)

	// Project back to model dimension and normalize
	out := ggml.MulMat(gctx, lt["pe_proj"], gate)
	out = rmsNormApply(gctx, out, lt["pe_post_norm"], rmsEps)

	return ggml.Add(gctx, x, out)
}
