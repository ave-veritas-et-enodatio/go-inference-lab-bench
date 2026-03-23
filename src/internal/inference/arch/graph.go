package arch

import (
	"fmt"
	"math"
	"strings"
	"unsafe"

	ggml "inference-lab-bench/internal/inference/ggml"
)

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
	inputs *GraphInputs, rmsEps float32, blkFn blockFunc) ggml.Tensor {
	for il := range m.Params.Ints["n_layers"] {
		lt := m.Store.Layer(il)
		blockName := m.LayerBlockNames[il]
		blockDef := m.Def.Blocks[blockName]

		if !lt["attn_norm"].IsNil() {
			cur := rmsNormApply(gctx, x, lt["attn_norm"], rmsEps)
			cur = blkFn(gctx, cur, lt, il, blockDef.Config, inputs)
			x = ggml.Add(gctx, x, cur)
		}

		x = m.buildFFNBlock(gctx, x, lt, il, rmsEps)
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

	inputs := &GraphInputs{
		InpPos:  inpPos,
		InpMask: inpMask,
		NTokens: nTokens,
		NKV:     nTokens,
		SeqPos:  0,
	}

	var zeroFill []ggml.Tensor

	blkFn := func(gctx *ggml.GraphContext, cur ggml.Tensor,
		lt map[string]ggml.Tensor, il int, config map[string]any,
		inputs *GraphInputs) ggml.Tensor {
		return m.BlockBuilders[il].BuildStateless(gctx, cur, lt, m.Params, config, inputs, &zeroFill)
	}
	x = m.runLayers(gctx, x, inputs, rmsEps, blkFn)

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

	maskData := make([]float32, nTokens*nTokens)
	if !m.Def.Architecture.NonCausal {
		for qi := int64(0); qi < nTokens; qi++ {
			for kj := int64(0); kj < nTokens; kj++ {
				if kj > qi {
					maskData[qi*nTokens+kj] = float32(math.Inf(-1))
				}
			}
		}
	}
	ggml.TensorSet(inpMask, unsafe.Pointer(&maskData[0]), 0, inpMask.Nbytes())

	// Execute
	status := sched.Compute(gf)
	if status != ggml.StatusSuccess {
		return nil, fmt.Errorf("graph compute failed: %d", status)
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

	inputs := &GraphInputs{
		InpPos:  inpPos,
		InpMask: inpMask,
		NTokens: nNew,
		NKV:     nKV,
		SeqPos:  seqPos,
	}

	var writebacks []CacheWriteback

	blkFn := func(gctx *ggml.GraphContext, cur ggml.Tensor,
		lt map[string]ggml.Tensor, il int, config map[string]any,
		inputs *GraphInputs) ggml.Tensor {
		return m.BlockBuilders[il].BuildCached(gctx, cur, lt, m.Params, config, inputs,
			&gc.Layers[il], &writebacks)
	}
	x = m.runLayers(gctx, x, inputs, rmsEps, blkFn)

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

	maskData := make([]float32, nKV*nNew)
	if !m.Def.Architecture.NonCausal {
		for qi := int64(0); qi < nNew; qi++ {
			absQI := int64(seqPos) + qi
			for kj := int64(0); kj < nKV; kj++ {
				if kj > absQI {
					maskData[qi*nKV+kj] = float32(math.Inf(-1))
				}
			}
		}
	}
	ggml.TensorSet(inpMask, unsafe.Pointer(&maskData[0]), 0, inpMask.Nbytes())

	status := sched.Compute(gf)
	if status != ggml.StatusSuccess {
		return nil, fmt.Errorf("graph compute failed: %d", status)
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
