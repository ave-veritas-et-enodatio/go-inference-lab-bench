package ggml

import (
	"math"
	"unsafe"
	"testing"
)

// fillF32 writes vals into the arena-allocated tensor t. Caller is
// responsible for ensuring t was allocated in an AllocPermAllow context.
func fillF32(t Tensor, vals []float32) {
	dst := TensorData(t)
	if dst == nil {
		panic("fillF32: nil data pointer (context not AllocPermAllow?)")
	}
	dstBytes := unsafe.Slice((*byte)(dst), len(vals)*4)
	for i, v := range vals {
		bits := math.Float32bits(v)
		dstBytes[i*4+0] = byte(bits)
		dstBytes[i*4+1] = byte(bits >> 8)
		dstBytes[i*4+2] = byte(bits >> 16)
		dstBytes[i*4+3] = byte(bits >> 24)
	}
}

// readF32 reads numEls F32 values back from the arena-allocated tensor t.
func readF32(t Tensor, numEls int) []float32 {
	src := TensorData(t)
	if src == nil {
		panic("readF32: nil data pointer")
	}
	srcBytes := unsafe.Slice((*byte)(src), numEls*4)
	out := make([]float32, numEls)
	for i := range out {
		bits := uint32(srcBytes[i*4+0]) |
			uint32(srcBytes[i*4+1])<<8 |
			uint32(srcBytes[i*4+2])<<16 |
			uint32(srcBytes[i*4+3])<<24
		out[i] = math.Float32frombits(bits)
	}
	return out
}

// TestConv2D_AllOnesPatchSum is the simplest end-to-end check that the
// Conv2D wrapper binds the cgo entry point and computes correctly. A 2x2
// all-ones kernel applied to a 4x4 all-ones input with stride=2,
// padding=0 partitions the input into four non-overlapping 2x2 patches.
// Each output cell is the sum of its 2x2 patch (= 4).
func TestConv2D_AllOnesPatchSum(t *testing.T) {
	ctx := NewGraphContext(8*1024*1024, AllocPermAllow)
	if ctx == nil {
		t.Fatal("NewGraphContext returned nil")
	}
	defer ctx.Free()

	// Kernel ne layout: [kw, kh, ic, oc]
	kernel := NewTensor4D(ctx, TypeF32, 2, 2, 1, 1)
	// Data ne layout: [iw, ih, ic, n]
	data := NewTensor4D(ctx, TypeF32, 4, 4, 1, 1)
	if kernel.IsNil() || data.IsNil() {
		t.Fatal("tensor allocation returned nil")
	}
	fillF32(kernel, []float32{1, 1, 1, 1})
	fillF32(data, []float32{
		1, 1, 1, 1,
		1, 1, 1, 1,
		1, 1, 1, 1,
		1, 1, 1, 1,
	})

	out := Conv2D(ctx, kernel, data, 2, 2, 0, 0, 1, 1)
	if out.IsNil() {
		t.Fatal("Conv2D returned nil tensor")
	}

	// Output ne should be [2, 2, 1, 1].
	if got := []int64{out.Ne(0), out.Ne(1), out.Ne(2), out.Ne(3)}; got[0] != 2 || got[1] != 2 || got[2] != 1 || got[3] != 1 {
		t.Errorf("output ne = %v, want [2 2 1 1]", got)
	}

	gf := NewGraph(ctx, 16)
	gf.BuildForwardExpand(out)
	GraphCompute(ctx, gf, 1)

	got := readF32(out, 4)
	want := []float32{4, 4, 4, 4}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("output[%d] = %v, want %v (full: %v)", i, got[i], want[i], got)
			break
		}
	}
}

// TestConv2D_PatchEmbedShape verifies that the output shape for the
// Gemma 4 vision-tower patch-embedding configuration (kernel 16x16,
// stride 16, 3 input channels, 768 output channels, 224x224 input)
// resolves to the expected 14x14x768 grid. No numerical assertion — this
// is a pure shape gate so that the upstream consumer (Phase 4 encoder
// graph) can rely on the conv producing the right downstream dims.
func TestConv2D_PatchEmbedShape(t *testing.T) {
	ctx := NewGraphContext(64*1024*1024, AllocPermAllow)
	if ctx == nil {
		t.Fatal("NewGraphContext returned nil")
	}
	defer ctx.Free()

	kernel := NewTensor4D(ctx, TypeF32, 16, 16, 3, 768) // [kw, kh, ic, oc]
	data := NewTensor4D(ctx, TypeF32, 224, 224, 3, 1)   // [iw, ih, ic, n]
	if kernel.IsNil() || data.IsNil() {
		t.Fatal("tensor allocation returned nil")
	}

	out := Conv2D(ctx, kernel, data, 16, 16, 0, 0, 1, 1)
	if out.IsNil() {
		t.Fatal("Conv2D returned nil tensor")
	}
	if got := []int64{out.Ne(0), out.Ne(1), out.Ne(2), out.Ne(3)}; got[0] != 14 || got[1] != 14 || got[2] != 768 || got[3] != 1 {
		t.Errorf("output ne = %v, want [14 14 768 1]", got)
	}
}

// TestConv2D_NilGuards confirms the wrapper short-circuits on nil inputs
// without invoking cgo (matches the pattern used by other op wrappers).
func TestConv2D_NilGuards(t *testing.T) {
	ctx := NewGraphContext(1024*1024, AllocPermAllow)
	if ctx == nil {
		t.Fatal("NewGraphContext returned nil")
	}
	defer ctx.Free()

	good := NewTensor4D(ctx, TypeF32, 2, 2, 1, 1)
	cases := []struct {
		name           string
		kernel, data   Tensor
	}{
		{"nil_kernel", NilTensor(), good},
		{"nil_data", good, NilTensor()},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			out := Conv2D(ctx, c.kernel, c.data, 1, 1, 0, 0, 1, 1)
			if !out.IsNil() {
				t.Errorf("expected NilTensor for %s, got non-nil", c.name)
			}
		})
	}
}
