package arch

import (
	"fmt"
	"strings"

	"inference-lab-bench/internal/ggml"
)

// Transforms — operations applied to tensor data after reading from
// safetensors, before storing as the final ggml type.
//
// The transform pipeline mirrors the numeric transforms done by the
// HuggingFace → GGUF converter (tools/convert_hf_to_gguf.py) so that
// safetensors loading yields tensors that are numerically identical to what a
// llama.cpp-converted GGUF would contain. This keeps the safetensors path
// data-driven: everything specific to a given architecture lives in the
// .arch.stmap.toml file under [[transforms]], not in Go.
//
// Two transform families:
//
//   - Element-wise (chunkable, executed inside ggml graphs):
//   - add_scalar: f32[i] += value             → ScaleBias(x, 1.0, value)
//   - neg_exp:    f32[i] = -exp(f32[i])       → Neg(Exp(x))
//
//   - Structural (whole-tensor, executed in Go on F32 host buffers):
//   - reorder_v_heads: permute V-head groups from HF grouped layout to
//     ggml tiled layout along either row or column axis, with optional
//     row offset for partial-tensor reorders.
//
// The split exists because chunked element-wise pipelines must be safe to
// process one slice at a time, while structural permutations need the whole
// tensor in scope.

// stElemwiseOp is a single element-wise transform expressed as a ggml node
// builder. Given an F32 tensor x, it returns an F32 tensor representing
// op(x). Pure — no side effects beyond adding nodes to ctx's graph.
type stElemwiseOp func(ctx *ggml.GraphContext, x ggml.Tensor) ggml.Tensor

// stXformBatch buckets transforms applicable to a single tensor by execution
// stage relative to the structural pass.
type stXformBatch struct {
	pre        []*TransformSpec // element-wise transforms before any structural pass
	structural []*TransformSpec // structural transforms (currently only reorder_v_heads)
	post       []*TransformSpec // element-wise transforms after the structural pass
}

// hasWork reports whether the batch contains any applicable transforms.
func (b *stXformBatch) hasWork() bool {
	return len(b.pre) > 0 || len(b.structural) > 0 || len(b.post) > 0
}

// hasStructural reports whether the batch contains a structural transform.
func (b *stXformBatch) hasStructural() bool {
	return len(b.structural) > 0
}

// transformIsStructural reports whether the given op operates on tensor
// structure (and therefore cannot be chunked along the flat element axis).
func transformIsStructural(op string) bool {
	return op == "reorder_v_heads" || op == "permute_qk_rope"
}

// splitTensorTransforms walks the declared transform list once and partitions
// the transforms applicable to ggufName into (pre, structural, post) buckets.
// The buckets preserve declared order so that semantics remain identical to
// the unified F32 walker — only the execution venue (ggml graphs vs Go loops)
// differs across element-wise vs structural transforms.
func splitTensorTransforms(transforms []TransformSpec, ggufName string) stXformBatch {
	var b stXformBatch
	for i := range transforms {
		t := &transforms[i]
		if !tensorMatchesApply(ggufName, t.Apply) {
			continue
		}
		if transformIsStructural(t.Op) {
			b.structural = append(b.structural, t)
			continue
		}
		if len(b.structural) == 0 {
			b.pre = append(b.pre, t)
		} else {
			b.post = append(b.post, t)
		}
	}
	return b
}

// buildElemwiseOps converts element-wise transform specs into ggml node
// builders. Each builder takes an F32 tensor and returns a transformed F32
// tensor. Errors if any spec is structural or unrecognized.
func buildElemwiseOps(specs []*TransformSpec) ([]stElemwiseOp, error) {
	if len(specs) == 0 {
		return nil, nil
	}
	ops := make([]stElemwiseOp, 0, len(specs))
	for _, t := range specs {
		switch t.Op {
		case "add_scalar":
			v, err := paramFloat(t.Params, "value")
			if err != nil {
				return nil, fmt.Errorf("add_scalar: %w", err)
			}
			// ScaleBias(x, 1.0, v) = 1.0*x + v = x + v.
			ops = append(ops, func(ctx *ggml.GraphContext, x ggml.Tensor) ggml.Tensor {
				return ggml.ScaleBias(ctx, x, 1.0, v)
			})
		case "neg_exp":
			ops = append(ops, func(ctx *ggml.GraphContext, x ggml.Tensor) ggml.Tensor {
				return ggml.Neg(ctx, ggml.Exp(ctx, x))
			})
		default:
			return nil, fmt.Errorf("buildElemwiseOps: %q is not an element-wise op", t.Op)
		}
	}
	return ops, nil
}

// applyStructuralTransforms applies structural (whole-tensor) transforms to a
// full F32 buffer in place. Handles reorder_v_heads and permute_qk_rope.
func applyStructuralTransforms(specs []*TransformSpec, ggufName string, f32 []float32, spec TensorSpec) error {
	for _, t := range specs {
		switch t.Op {
		case "reorder_v_heads":
			if err := applyReorderVHeads(t, f32, spec); err != nil {
				return fmt.Errorf("transform reorder_v_heads on %s: %w", ggufName, err)
			}
		case "permute_qk_rope":
			if err := applyPermuteQKRope(t, f32, spec); err != nil {
				return fmt.Errorf("transform permute_qk_rope on %s: %w", ggufName, err)
			}
		default:
			return fmt.Errorf("applyStructuralTransforms: %q is not a structural op", t.Op)
		}
	}
	return nil
}

// tensorMatchesAnyTransform reports whether any transform in the list applies
// to the given GGUF tensor name. Used by ReadTensor to decide if the slow
// (graph-based) path is required.
func tensorMatchesAnyTransform(transforms []TransformSpec, ggufName string) bool {
	for i := range transforms {
		if tensorMatchesApply(ggufName, transforms[i].Apply) {
			return true
		}
	}
	return false
}

// tensorMatchesApply matches a full GGUF tensor name against an apply list of
// short tensor names. A tensor matches iff either:
//   - the full name equals the apply entry (global tensors like "output_norm.weight"), or
//   - the full name ends in "." + entry (per-layer like "blk.5.attn_norm.weight").
//
// The "." separator prevents false-positive partial suffix matches (e.g.
// "attn_norm.weight" does not match "post_attention_norm.weight").
func tensorMatchesApply(ggufName string, apply []string) bool {
	for _, a := range apply {
		if ggufName == a {
			return true
		}
		if strings.HasSuffix(ggufName, "."+a) {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// Per-op implementations
// ---------------------------------------------------------------------------

// applyReorderVHeads permutes V heads from HF grouped layout [G0_v0..v{r-1},
// G1_v0..v{r-1}, ...] to ggml tiled layout [v0 for each K, v1 for each K, ...].
// Mirrors Qwen3NextModel._reorder_v_heads in convert_hf_to_gguf.py.
//
// Params:
//
//	axis         : "row" (permute outer axis) or "col" (permute inner axis
//	               within each row; 2D only)
//	num_k_heads  : number of K heads (int)
//	num_v_per_k  : number of V heads per K head (int); num_v_heads = K * num_v_per_k
//	head_dim     : elements per V head along the permuted axis (int; 1 for
//	               scalar-per-head tensors like A_log or in_proj_a)
//	offset_rows  : (optional, default 0) number of rows to skip before the
//	               permuted region starts — used for partial-tensor reorders
//	               like attn_qkv (Q+K rows preceding the V block) and conv1d
//	               (Q+K channels preceding V channels).
func applyReorderVHeads(t *TransformSpec, f32 []float32, spec TensorSpec) error {
	axis, err := paramString(t.Params, "axis")
	if err != nil {
		return err
	}
	numKHeads, err := paramInt(t.Params, "num_k_heads")
	if err != nil {
		return err
	}
	numVPerK, err := paramInt(t.Params, "num_v_per_k")
	if err != nil {
		return err
	}
	headDim, err := paramInt(t.Params, ParamHeadDim)
	if err != nil {
		return err
	}
	offsetRows := paramIntDefault(t.Params, "offset_rows", 0)

	switch axis {
	case "row":
		return reorderRowsF32(f32, spec, numKHeads, numVPerK, headDim, offsetRows)
	case "col":
		return reorderColsF32(f32, spec, numKHeads, numVPerK, headDim)
	default:
		return fmt.Errorf("reorder_v_heads: axis must be 'row' or 'col', got %q", axis)
	}
}

// applyPermuteQKRope permutes the rows of a Q or K projection weight matrix
// to convert from HF's NeoX-style head layout (rows 0..half-1 = first half of
// each head, rows half..head_dim-1 = second half) to the interleaved layout
// that standard (llama.cpp "standard") RoPE expects.
//
// Mirrors llama.cpp's tools/convert_hf_to_gguf.py:
//
//	permute(weights, n_head):
//	  weights.reshape(n_head, 2, head_dim/2, in_dim)
//	         .swapaxes(1, 2)
//	         .reshape(n_head * head_dim, in_dim)
//
// For each head block of head_dim contiguous rows the new layout is
// new_row[2k]   = old_row[k]
// new_row[2k+1] = old_row[head_dim/2 + k]
// for k in [0, head_dim/2). Rows from different heads do not mix.
//
// Params:
//
//	n_heads : number of heads along the outer axis (ne[1]) — n_head for Q,
//	          n_kv_head for K when they differ.
//
// The weight tensor must be 2D with ne[1] = n_heads * head_dim and head_dim
// even.
func applyPermuteQKRope(t *TransformSpec, f32 []float32, spec TensorSpec) error {
	nHeads, err := paramInt(t.Params, ParamNHeads)
	if err != nil {
		return err
	}
	if nHeads <= 0 {
		return fmt.Errorf("permute_qk_rope: n_heads must be positive, got %d", nHeads)
	}
	if spec.Ne[1] <= 1 {
		return fmt.Errorf("permute_qk_rope: expected 2D tensor, got ne=%v", spec.Ne)
	}
	rowLen := int(spec.Ne[0])
	numRows := int(spec.Ne[1])
	if numRows%nHeads != 0 {
		return fmt.Errorf("permute_qk_rope: outer dim %d not divisible by n_heads=%d", numRows, nHeads)
	}
	headDim := numRows / nHeads
	if headDim%2 != 0 {
		return fmt.Errorf("permute_qk_rope: head_dim=%d must be even", headDim)
	}
	half := headDim / 2
	rowsPerHead := headDim * rowLen
	if int64(nHeads)*int64(rowsPerHead) > int64(len(f32)) {
		return fmt.Errorf("permute_qk_rope: buffer too small (%d heads × %d elems > len=%d)",
			nHeads, rowsPerHead, len(f32))
	}
	tmp := make([]float32, rowsPerHead)
	for h := range nHeads {
		headBase := h * rowsPerHead
		src := f32[headBase : headBase+rowsPerHead]
		for k := range half {
			copy(tmp[(2*k)*rowLen:(2*k+1)*rowLen], src[k*rowLen:(k+1)*rowLen])
			copy(tmp[(2*k+1)*rowLen:(2*k+2)*rowLen], src[(half+k)*rowLen:(half+k+1)*rowLen])
		}
		copy(src, tmp)
	}
	return nil
}

// ---------------------------------------------------------------------------
// Core permutation primitives
// ---------------------------------------------------------------------------

// reorderRowsF32 permutes V-head groups along the tensor's "outer" axis.
//
// For 1D tensors (ne[1] == 1) the outer axis is ne[0] and each "head" is
// headDim contiguous scalars. For 2D tensors the outer axis is ne[1] and each
// head spans headDim rows of ne[0] elements.
//
// The permuted region starts at row offsetRows and covers numKHeads*numVPerK*headDim
// outer-axis entries. Entries outside the region are left unchanged.
func reorderRowsF32(f32 []float32, spec TensorSpec, numKHeads, numVPerK, headDim, offsetRows int) error {
	var rowStride, axisLen int64
	if spec.Ne[1] <= 1 {
		rowStride = 1
		axisLen = spec.Ne[0]
	} else {
		rowStride = spec.Ne[0]
		axisLen = spec.Ne[1]
	}

	numVHeads := int64(numKHeads) * int64(numVPerK)
	need := int64(offsetRows) + numVHeads*int64(headDim)
	if need > axisLen {
		return fmt.Errorf("reorder rows out of range: offset_rows=%d + num_v_heads=%d * head_dim=%d = %d > axis_len=%d (ne=%v)",
			offsetRows, numVHeads, headDim, need, axisLen, spec.Ne)
	}

	startElems := int64(offsetRows) * rowStride
	chunkElems := int64(headDim) * rowStride
	totalElems := numVHeads * chunkElems
	if startElems+totalElems > int64(len(f32)) {
		return fmt.Errorf("reorder rows: buffer too small (start=%d + total=%d > len=%d)",
			startElems, totalElems, len(f32))
	}
	permuteChunks(f32[startElems:startElems+totalElems], int(chunkElems), numKHeads, numVPerK)
	return nil
}

// reorderColsF32 permutes V-head groups along the tensor's inner axis (ne[0]),
// independently for each row in ne[1]. Each row must contain exactly
// numKHeads*numVPerK*headDim elements.
//
// Used by Qwen3.5 out_proj: the input dimension (ne[0]) is V-head-grouped and
// needs to be de-grouped for ggml's tiled layout.
func reorderColsF32(f32 []float32, spec TensorSpec, numKHeads, numVPerK, headDim int) error {
	numVHeads := numKHeads * numVPerK
	rowLen := int(spec.Ne[0])
	numRows := int(spec.Ne[1])
	if numVHeads*headDim != rowLen {
		return fmt.Errorf("reorder cols: num_v_heads=%d * head_dim=%d = %d != ne[0]=%d",
			numVHeads, headDim, numVHeads*headDim, rowLen)
	}
	if int64(numRows)*int64(rowLen) > int64(len(f32)) {
		return fmt.Errorf("reorder cols: buffer too small (%d rows * %d cols > len=%d)",
			numRows, rowLen, len(f32))
	}
	for r := range numRows {
		start := r * rowLen
		permuteChunks(f32[start:start+rowLen], headDim, numKHeads, numVPerK)
	}
	return nil
}

// permuteChunks permutes numKHeads*numVPerK contiguous chunks of size chunkSize
// within buf, remapping HF grouped layout to ggml tiled layout.
//
// Grouped layout (HF, input):
//
//	[K0_v0 K0_v1 ... K0_v{r-1}  K1_v0 K1_v1 ... K1_v{r-1}  ...]
//
// Tiled layout (ggml, output):
//
//	[K0_v0 K1_v0 ... K{K-1}_v0  K0_v1 K1_v1 ... K{K-1}_v1  ...]
//
// Derivation: _reorder_v_heads reshapes the axis as [K, V, D] and swaps the
// first two dims to [V, K, D]. After reshape-back, the element at logical
// output index (v, k, d) reads from input (k, v, d), i.e. for output flat
// index i = v*K*D + k*D + d (head index h_out = v*K + k), the corresponding
// input flat index is k*V*D + v*D + d (head index h_in = k*V + v).
func permuteChunks(buf []float32, chunkSize, numKHeads, numVPerK int) {
	numVHeads := numKHeads * numVPerK
	total := chunkSize * numVHeads
	if total == 0 || total > len(buf) {
		return
	}
	tmp := make([]float32, total)
	for outHead := range numVHeads {
		v := outHead / numKHeads
		k := outHead % numKHeads
		inHead := k*numVPerK + v
		copy(
			tmp[outHead*chunkSize:(outHead+1)*chunkSize],
			buf[inHead*chunkSize:(inHead+1)*chunkSize],
		)
	}
	copy(buf[:total], tmp)
}

// ---------------------------------------------------------------------------
// Param readers (TOML-decoded map[string]any)
// ---------------------------------------------------------------------------

func paramFloat(p map[string]any, key string) (float32, error) {
	v, ok := p[key]
	if !ok {
		return 0, fmt.Errorf("missing param %q", key)
	}
	switch x := v.(type) {
	case float64:
		return float32(x), nil
	case float32:
		return x, nil
	case int64:
		return float32(x), nil
	case int:
		return float32(x), nil
	default:
		return 0, fmt.Errorf("param %q must be a number, got %T", key, v)
	}
}

func paramInt(p map[string]any, key string) (int, error) {
	v, ok := p[key]
	if !ok {
		return 0, fmt.Errorf("missing param %q", key)
	}
	switch x := v.(type) {
	case int64:
		return int(x), nil
	case int:
		return x, nil
	case float64:
		return int(x), nil
	default:
		return 0, fmt.Errorf("param %q must be an integer, got %T", key, v)
	}
}

func paramIntDefault(p map[string]any, key string, dflt int) int {
	if _, ok := p[key]; !ok {
		return dflt
	}
	n, err := paramInt(p, key)
	if err != nil {
		return dflt
	}
	return n
}

func paramString(p map[string]any, key string) (string, error) {
	v, ok := p[key]
	if !ok {
		return "", fmt.Errorf("missing param %q", key)
	}
	s, ok := v.(string)
	if !ok {
		return "", fmt.Errorf("param %q must be a string, got %T", key, v)
	}
	return s, nil
}
