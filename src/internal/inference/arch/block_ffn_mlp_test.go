package arch

import (
	"math"
	"slices"
	"testing"

	ggml "inference-lab-bench/internal/ggml"
)

// TestMLPBuilderRegistered confirms "mlp" resolves to a non-nil FFN builder so
// TOML [ffn]/[vision.ffn] sections naming it load.
func TestMLPBuilderRegistered(t *testing.T) {
	b, ok := GetFFNBuilder("mlp")
	if !ok || b == nil {
		t.Fatal(`GetFFNBuilder("mlp") not found`)
	}
}

// TestMLPContract pins the contract: required up/down, optional up/down biases,
// and an activation config schema. These are what Validate() checks against the
// TOML, so a drift here silently breaks load-time validation.
func TestMLPContract(t *testing.T) {
	b, ok := GetFFNBuilder("mlp")
	if !ok {
		t.Fatal("mlp builder not registered")
	}
	c := b.Contract()

	if c.Kind != KindFFN {
		t.Errorf("Kind = %v, want KindFFN", c.Kind)
	}
	if c.ExpertRouted {
		t.Error("mlp must not be ExpertRouted")
	}
	for _, w := range []string{MoEUp, MoEDown} {
		if !slices.Contains(c.RequiredWeights, w) {
			t.Errorf("RequiredWeights %v missing %q", c.RequiredWeights, w)
		}
	}
	if slices.Contains(c.RequiredWeights, MoEGate) {
		t.Error("plain MLP must not require a gate weight")
	}
	for _, w := range []string{MoEUpBias, MoEDownBias} {
		if !slices.Contains(c.OptionalWeights, w) {
			t.Errorf("OptionalWeights %v missing %q", c.OptionalWeights, w)
		}
	}
	allowed, ok := c.ConfigSchema[ConfigActivation]
	if !ok {
		t.Fatalf("ConfigSchema missing %q", ConfigActivation)
	}
	for _, v := range []string{ActivationSiLU, ActivationGELU} {
		if !slices.Contains(allowed, v) {
			t.Errorf("ConfigActivation schema %v missing %q", allowed, v)
		}
	}
}

// newFilledF32 allocates and fills an arena F32 tensor (reusing the axial test
// helpers fillAxialF32 / readAxialF32 from this package).
func newFilledF32(t *testing.T, ctx *ggml.GraphContext, vals []float32, ne ...int64) ggml.Tensor {
	t.Helper()
	d := [4]int64{1, 1, 1, 1}
	copy(d[:], ne)
	tensor := makeTensorFromSpec(ctx, ggml.TypeF32, d[0], d[1], d[2], d[3])
	if tensor.IsNil() {
		t.Fatal("tensor allocation returned nil")
	}
	fillAxialF32(tensor, vals)
	return tensor
}

func computeFlat(t *testing.T, ctx *ggml.GraphContext, out ggml.Tensor, n int) []float32 {
	t.Helper()
	if out.IsNil() {
		t.Fatal("output tensor is nil")
	}
	gf := ggml.NewGraph(ctx, 256)
	gf.BuildForwardExpand(out)
	ggml.GraphCompute(ctx, gf, 1)
	return readAxialF32(out, n)
}

// TestMLPBuildFFN_ShapeAndBiasNoop builds a tiny MLP graph and asserts:
//   - output shape is [n_embd, n_tokens] (down maps ff → n_embd),
//   - a nil bias map yields identical output to an all-zero bias (proving the
//     bias add is a true no-op when the weight is absent).
//
// Weights are identity-ish so the result is independently checkable: with up =
// I (n_embd→ff square, ff==n_embd) and down = I, the FFN reduces to act(x).
func TestMLPBuildFFN_ShapeAndBiasNoop(t *testing.T) {
	const (
		nEmbd   = 4
		ff      = 4
		nTokens = 3
	)
	ctx := ggml.NewGraphContext(8*1024*1024, ggml.AllocPermAllow)
	if ctx == nil {
		t.Fatal("NewGraphContext returned nil")
	}
	defer ctx.Free()

	// Input [n_embd, n_tokens], distinct positive values.
	inVals := make([]float32, nEmbd*nTokens)
	for i := range inVals {
		inVals[i] = float32(i+1) * 0.1
	}
	// Identity matrices stored row-major as [in_features, out_features] (ggml
	// MulMat: w is [n_embd, ff], x is [n_embd, n_tokens] → [ff, n_tokens]).
	identity := func() []float32 {
		m := make([]float32, nEmbd*ff)
		for i := 0; i < nEmbd && i < ff; i++ {
			m[i*ff+i] = 1
		}
		return m
	}

	build := func(withBias bool) ggml.Tensor {
		input := newFilledF32(t, ctx, inVals, nEmbd, nTokens)
		wUp := newFilledF32(t, ctx, identity(), nEmbd, ff)
		wDown := newFilledF32(t, ctx, identity(), ff, nEmbd)
		weights := map[string]ggml.Tensor{
			MoEUp:       wUp,
			MoEDown:     wDown,
			MoEUpBias:   ggml.NilTensor(),
			MoEDownBias: ggml.NilTensor(),
		}
		if withBias {
			weights[MoEUpBias] = newFilledF32(t, ctx, make([]float32, ff), ff)
			weights[MoEDownBias] = newFilledF32(t, ctx, make([]float32, nEmbd), nEmbd)
		}
		b := &mlpBuilder{}
		return b.BuildFFN(ctx, input, weights, &ResolvedParams{}, map[string]any{}, &GraphInputs{})
	}

	out := build(false)
	if out.Ne(0) != nEmbd || out.Ne(1) != nTokens {
		t.Fatalf("output ne = [%d %d], want [%d %d]", out.Ne(0), out.Ne(1), nEmbd, nTokens)
	}
	got := computeFlat(t, ctx, out, nEmbd*nTokens)

	// Expected = gelu(input) since up/down are identity. Just verify the
	// activation moved the values (sanity) and that they are finite.
	moved := false
	for i := range inVals {
		if math.Abs(float64(got[i]-inVals[i])) > 1e-4 {
			moved = true
		}
		if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
			t.Fatalf("output[%d] not finite: %v", i, got[i])
		}
	}
	if !moved {
		t.Error("activation appears to be a no-op (gelu should change positive inputs)")
	}

	// Zero-bias output must match the nil-bias output exactly.
	outZeroBias := build(true)
	gotZero := computeFlat(t, ctx, outZeroBias, nEmbd*nTokens)
	for i := range got {
		if math.Abs(float64(got[i]-gotZero[i])) > 1e-6 {
			t.Errorf("nil-bias and zero-bias diverge at %d: %v vs %v", i, got[i], gotZero[i])
		}
	}
}

// TestLayerNormApply_AffineAndBiasNoop checks the LayerNorm helper:
//   - weight=1, bias=nil normalizes to mean 0 (within tol) over ne[0],
//   - a nil bias is a no-op vs a zero bias.
func TestLayerNormApply_AffineAndBiasNoop(t *testing.T) {
	const (
		dim     = 4
		nTokens = 2
		eps     = 1e-6
	)
	ctx := ggml.NewGraphContext(8*1024*1024, ggml.AllocPermAllow)
	if ctx == nil {
		t.Fatal("NewGraphContext returned nil")
	}
	defer ctx.Free()

	xVals := []float32{1, 2, 3, 4, 10, 20, 30, 40}
	ones := []float32{1, 1, 1, 1}

	run := func(withBias bool) []float32 {
		x := newFilledF32(t, ctx, xVals, dim, nTokens)
		w := newFilledF32(t, ctx, ones, dim)
		bias := ggml.NilTensor()
		if withBias {
			bias = newFilledF32(t, ctx, make([]float32, dim), dim)
		}
		out := layerNormApply(ctx, x, w, bias, eps)
		if out.Ne(0) != dim || out.Ne(1) != nTokens {
			t.Fatalf("layerNorm output ne = [%d %d], want [%d %d]", out.Ne(0), out.Ne(1), dim, nTokens)
		}
		return computeFlat(t, ctx, out, dim*nTokens)
	}

	got := run(false)
	// Each row (dim elements) should be zero-mean after LayerNorm.
	for row := 0; row < nTokens; row++ {
		var sum float64
		for i := 0; i < dim; i++ {
			sum += float64(got[row*dim+i])
		}
		if math.Abs(sum/float64(dim)) > 1e-4 {
			t.Errorf("row %d mean = %v, want ~0", row, sum/float64(dim))
		}
	}

	gotZero := run(true)
	for i := range got {
		if math.Abs(float64(got[i]-gotZero[i])) > 1e-6 {
			t.Errorf("nil-bias and zero-bias LayerNorm diverge at %d: %v vs %v", i, got[i], gotZero[i])
		}
	}
}
