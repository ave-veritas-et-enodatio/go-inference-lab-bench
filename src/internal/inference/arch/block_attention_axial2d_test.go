package arch

import (
	"math"
	"slices"
	"unsafe"

	"testing"

	ggml "inference-lab-bench/internal/ggml"
)

// fillAxialF32 writes vals into an arena-allocated F32 tensor.
func fillAxialF32(t ggml.Tensor, vals []float32) {
	dst := ggml.TensorData(t)
	b := unsafe.Slice((*byte)(dst), len(vals)*4)
	for i, v := range vals {
		bits := math.Float32bits(v)
		b[i*4+0] = byte(bits)
		b[i*4+1] = byte(bits >> 8)
		b[i*4+2] = byte(bits >> 16)
		b[i*4+3] = byte(bits >> 24)
	}
}

// fillAxialI32 writes vals into an arena-allocated I32 tensor.
func fillAxialI32(t ggml.Tensor, vals []int32) {
	dst := ggml.TensorData(t)
	b := unsafe.Slice((*byte)(dst), len(vals)*4)
	for i, v := range vals {
		u := uint32(v)
		b[i*4+0] = byte(u)
		b[i*4+1] = byte(u >> 8)
		b[i*4+2] = byte(u >> 16)
		b[i*4+3] = byte(u >> 24)
	}
}

// readAxialF32 reads numEls F32 values back from an arena-allocated tensor.
func readAxialF32(t ggml.Tensor, numEls int) []float32 {
	src := ggml.TensorData(t)
	b := unsafe.Slice((*byte)(src), numEls*4)
	out := make([]float32, numEls)
	for i := range out {
		bits := uint32(b[i*4+0]) | uint32(b[i*4+1])<<8 |
			uint32(b[i*4+2])<<16 | uint32(b[i*4+3])<<24
		out[i] = math.Float32frombits(bits)
	}
	return out
}

// TestApplyRope2D_ShapeAndPositionBehavior exercises the axial (2D) RoPE math
// reused by the AttentionBuilder rope="axial2d" dispatch. It builds a known q
// tensor shaped [head_dim, n_heads, n_patches] and asserts:
//   - output shape == input shape (Concat rejoins the two rotated halves),
//   - at position 0 the rotation is identity (NeoX at pos 0 is the identity),
//   - with posX != posY the first half (rotated by X) and second half (rotated
//     by Y) diverge as expected.
func TestApplyRope2D_ShapeAndPositionBehavior(t *testing.T) {
	const (
		headDim   = 8
		nHeads    = 2
		nPatches  = 4
		ropeTheta = 10000.0
	)
	total := headDim * nHeads * nPatches

	ctx := ggml.NewGraphContext(8*1024*1024, ggml.AllocPermAllow)
	if ctx == nil {
		t.Fatal("NewGraphContext returned nil")
	}
	defer ctx.Free()

	// Known, distinct values so a no-op rotation is detectable.
	q := ggml.NewTensor3D(ctx, ggml.TypeF32, headDim, nHeads, nPatches)
	if q.IsNil() {
		t.Fatal("q allocation returned nil")
	}
	vals := make([]float32, total)
	for i := range vals {
		vals[i] = float32(i+1) * 0.25
	}
	fillAxialF32(q, vals)

	// --- Case A: posX == posY == 0 → identity rotation. ---
	posZero := ggml.NewTensor1D(ctx, ggml.TypeI32, nPatches)
	if posZero.IsNil() {
		t.Fatal("posZero allocation returned nil")
	}
	fillAxialI32(posZero, make([]int32, nPatches))

	outA := applyRope2D(ctx, q, posZero, posZero, headDim, ropeTheta)
	if outA.IsNil() {
		t.Fatal("applyRope2D returned nil")
	}
	if got := []int64{outA.Ne(0), outA.Ne(1), outA.Ne(2)}; got[0] != headDim || got[1] != nHeads || got[2] != nPatches {
		t.Fatalf("output ne = %v, want [%d %d %d]", got, headDim, nHeads, nPatches)
	}

	gf := ggml.NewGraph(ctx, 64)
	gf.BuildForwardExpand(outA)
	ggml.GraphCompute(ctx, gf, 1)

	gotA := readAxialF32(outA, total)
	const tol = 1e-4
	for i := range vals {
		if math.Abs(float64(gotA[i]-vals[i])) > tol {
			t.Errorf("pos-0 rotation should be identity: out[%d]=%v want %v", i, gotA[i], vals[i])
		}
	}

	// --- Case B: posX != posY → the two half-blocks diverge from input and
	// from each other's rotation, confirming each half is rotated by its own
	// position axis. ---
	posX := ggml.NewTensor1D(ctx, ggml.TypeI32, nPatches)
	posY := ggml.NewTensor1D(ctx, ggml.TypeI32, nPatches)
	if posX.IsNil() || posY.IsNil() {
		t.Fatal("posX/posY allocation returned nil")
	}
	// Distinct, nonzero positions; X and Y differ per patch.
	fillAxialI32(posX, []int32{1, 2, 3, 4})
	fillAxialI32(posY, []int32{4, 3, 2, 1})

	outB := applyRope2D(ctx, q, posX, posY, headDim, ropeTheta)
	if outB.IsNil() {
		t.Fatal("applyRope2D (case B) returned nil")
	}
	gfB := ggml.NewGraph(ctx, 64)
	gfB.BuildForwardExpand(outB)
	ggml.GraphCompute(ctx, gfB, 1)
	gotB := readAxialF32(outB, total)

	// The rotated output must differ from the input (nonzero positions rotate).
	diffs := 0
	for i := range vals {
		if math.Abs(float64(gotB[i]-vals[i])) > tol {
			diffs++
		}
	}
	if diffs == 0 {
		t.Error("nonzero positions produced no rotation — axial RoPE is a no-op")
	}

	// Per patch, the first half (rotated by posX) and second half (rotated by
	// posY) use different angles whenever posX != posY, so the half-vs-half
	// relationship must not be the trivial pos-0 identity. Concretely, swapping
	// posX and posY must change the output (it cannot if both halves ignored
	// position).
	outSwap := applyRope2D(ctx, q, posY, posX, headDim, ropeTheta)
	gfS := ggml.NewGraph(ctx, 64)
	gfS.BuildForwardExpand(outSwap)
	ggml.GraphCompute(ctx, gfS, 1)
	gotSwap := readAxialF32(outSwap, total)
	swapDiffs := 0
	for i := range gotB {
		if math.Abs(float64(gotB[i]-gotSwap[i])) > tol {
			swapDiffs++
		}
	}
	if swapDiffs == 0 {
		t.Error("swapping posX/posY left output unchanged — the two halves are not " +
			"rotated by distinct position axes")
	}
}

// TestAttentionContract_AcceptsAxial2D confirms rope="axial2d" is an allowed
// value in the AttentionBuilder ConfigSchema so TOML loaders accept it.
func TestAttentionContract_AcceptsAxial2D(t *testing.T) {
	b, ok := GetBlockBuilder("attention")
	if !ok {
		t.Fatal("attention builder not registered")
	}
	allowed, ok := b.Contract().ConfigSchema[ConfigRope]
	if !ok {
		t.Fatalf("ConfigRope (%q) missing from attention ConfigSchema", ConfigRope)
	}
	if !slices.Contains(allowed, RopeAxial2D) {
		t.Errorf("attention rope schema %v missing %q", allowed, RopeAxial2D)
	}
}

// TestPrepareQKV_Axial2DPanicsWithoutPositions proves the dispatch routes
// rope="axial2d" into the axial2d branch and that the nil-PosX/PosY guard
// fires. Reaching applyRope2D requires PosX/PosY; their absence is an
// unrecoverable wiring bug, surfaced as a panic (prepareQKV has no error
// channel). A minimal weights/params scaffold is enough to reach the RoPE
// step — Q projection then NeoX-rotate.
func TestPrepareQKV_Axial2DPanicsWithoutPositions(t *testing.T) {
	const (
		nEmbd    = 8
		headDim  = 8
		nHeads   = 1
		nKVHeads = 1
		nTokens  = 4
	)
	ctx := ggml.NewGraphContext(8*1024*1024, ggml.AllocPermAllow)
	if ctx == nil {
		t.Fatal("NewGraphContext returned nil")
	}
	defer ctx.Free()

	// Q projection weight [nEmbd, headDim*nHeads] and hidden state [nEmbd, nTokens].
	wq := ggml.NewTensor2D(ctx, ggml.TypeF32, nEmbd, headDim*nHeads)
	cur := ggml.NewTensor2D(ctx, ggml.TypeF32, nEmbd, nTokens)
	if wq.IsNil() || cur.IsNil() {
		t.Fatal("tensor allocation returned nil")
	}
	weights := map[string]ggml.Tensor{
		WeightAttnQ:      wq,
		WeightAttnOutput: ggml.NilTensor(),
	}
	params := &ResolvedParams{
		Ints:   map[string]int{},
		Floats: map[string]float32{},
	}
	config := map[string]any{ConfigRope: RopeAxial2D}
	// No PosX/PosY supplied — guard must fire.
	inputs := &GraphInputs{NTokens: nTokens}

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic on rope=axial2d with nil PosX/PosY, got none")
		}
	}()

	b := &AttentionBuilder{}
	_ = b.prepareQKV(ctx, cur, weights, params, config, inputs,
		headDim, nHeads, nKVHeads, 0, ggml.RopeTypeNeoX, 10000.0)
}
