package arch

import (
	"math"
	"slices"
	"testing"

	ggml "inference-lab-bench/internal/ggml"
)

// TestAttentionContract_AcceptsMRopeVision confirms rope="mrope_vision" is an
// allowed ConfigRope value and the four optional bias keys are declared, so the
// TOML loader accepts the Qwen3-VL vision tower's attention block.
func TestAttentionContract_AcceptsMRopeVision(t *testing.T) {
	b, ok := GetBlockBuilder("attention")
	if !ok {
		t.Fatal("attention builder not registered")
	}
	c := b.Contract()

	allowed, ok := c.ConfigSchema[ConfigRope]
	if !ok {
		t.Fatalf("ConfigRope (%q) missing from attention ConfigSchema", ConfigRope)
	}
	if !slices.Contains(allowed, RopeMRopeVision) {
		t.Errorf("attention rope schema %v missing %q", allowed, RopeMRopeVision)
	}

	for _, w := range []string{WeightAttnQBias, WeightAttnKBias, WeightAttnVBias, WeightAttnOutputBias} {
		if !slices.Contains(c.OptionalWeights, w) {
			t.Errorf("OptionalWeights %v missing bias key %q", c.OptionalWeights, w)
		}
	}
}

// TestPrepareQKV_MRopeVisionShapePreserved proves the mrope_vision branch runs
// (RopeMulti applied to both Q and K) and preserves the [head_dim, n_heads,
// n_tokens] shape. head_dim=8 → n_dims=4, sections={2,2,2,2}; positions are the
// 4-channel buffer (nTokens entries per channel).
func TestPrepareQKV_MRopeVisionShapePreserved(t *testing.T) {
	const (
		nEmbd    = 16
		headDim  = 8
		nHeads   = 2
		nKVHeads = 2
		nTokens  = 4
	)
	ctx := ggml.NewGraphContext(8*1024*1024, ggml.AllocPermAllow)
	if ctx == nil {
		t.Fatal("NewGraphContext returned nil")
	}
	defer ctx.Free()

	wq := ggml.NewTensor2D(ctx, ggml.TypeF32, nEmbd, headDim*nHeads)
	wk := ggml.NewTensor2D(ctx, ggml.TypeF32, nEmbd, headDim*nKVHeads)
	wv := ggml.NewTensor2D(ctx, ggml.TypeF32, nEmbd, headDim*nKVHeads)
	cur := ggml.NewTensor2D(ctx, ggml.TypeF32, nEmbd, nTokens)
	if wq.IsNil() || wk.IsNil() || wv.IsNil() || cur.IsNil() {
		t.Fatal("tensor allocation returned nil")
	}
	fillAxialF32(wq, make([]float32, nEmbd*headDim*nHeads))
	fillAxialF32(wk, make([]float32, nEmbd*headDim*nKVHeads))
	fillAxialF32(wv, make([]float32, nEmbd*headDim*nKVHeads))
	curVals := make([]float32, nEmbd*nTokens)
	for i := range curVals {
		curVals[i] = float32(i+1) * 0.1
	}
	fillAxialF32(cur, curVals)

	weights := map[string]ggml.Tensor{
		WeightAttnQ:      wq,
		WeightAttnK:      wk,
		WeightAttnV:      wv,
		WeightAttnOutput: ggml.NilTensor(),
	}
	params := &ResolvedParams{Ints: map[string]int{}, Floats: map[string]float32{}}
	config := map[string]any{ConfigRope: RopeMRopeVision}

	// 4-channel position buffer: 4*nTokens entries, channel-major [y,x,y,x].
	posBuf := VisionMRopePositions(2, 2, 2) // 2x2 grid → 4 patches == nTokens
	if len(posBuf) != 4*nTokens {
		t.Fatalf("position buffer length %d, want %d", len(posBuf), 4*nTokens)
	}
	pos := ggml.NewTensor1D(ctx, ggml.TypeI32, int64(len(posBuf)))
	if pos.IsNil() {
		t.Fatal("position tensor allocation returned nil")
	}
	fillAxialI32(pos, posBuf)

	inputs := &GraphInputs{NTokens: nTokens, InpPosVision: pos}

	b := &AttentionBuilder{}
	kv := b.prepareQKV(ctx, cur, weights, params, config, inputs,
		headDim, nHeads, nKVHeads, 0, 0, 10000.0)

	for _, tc := range []struct {
		name string
		t    ggml.Tensor
		d1   int64
	}{
		{"Q", kv.Q, nHeads},
		{"K", kv.K, nKVHeads},
	} {
		if tc.t.IsNil() {
			t.Fatalf("%s is nil after mrope_vision", tc.name)
		}
		if got := []int64{tc.t.Ne(0), tc.t.Ne(1), tc.t.Ne(2)}; got[0] != headDim || got[1] != tc.d1 || got[2] != nTokens {
			t.Errorf("%s ne = %v, want [%d %d %d]", tc.name, got, headDim, tc.d1, nTokens)
		}
	}

	// Confirm the graph computes without error.
	gf := ggml.NewGraph(ctx, 128)
	gf.BuildForwardExpand(kv.Q)
	gf.BuildForwardExpand(kv.K)
	ggml.GraphCompute(ctx, gf, 1)
}

// TestPrepareQKV_MRopeVisionPanicsWithoutPositions proves the dispatch routes
// rope="mrope_vision" into its branch and the nil-InpPosVision guard fires.
func TestPrepareQKV_MRopeVisionPanicsWithoutPositions(t *testing.T) {
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

	wq := ggml.NewTensor2D(ctx, ggml.TypeF32, nEmbd, headDim*nHeads)
	cur := ggml.NewTensor2D(ctx, ggml.TypeF32, nEmbd, nTokens)
	if wq.IsNil() || cur.IsNil() {
		t.Fatal("tensor allocation returned nil")
	}
	weights := map[string]ggml.Tensor{
		WeightAttnQ:      wq,
		WeightAttnOutput: ggml.NilTensor(),
	}
	params := &ResolvedParams{Ints: map[string]int{}, Floats: map[string]float32{}}
	config := map[string]any{ConfigRope: RopeMRopeVision}
	// No InpPosVision supplied — guard must fire.
	inputs := &GraphInputs{NTokens: nTokens}

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic on rope=mrope_vision with nil InpPosVision, got none")
		}
	}()

	b := &AttentionBuilder{}
	_ = b.prepareQKV(ctx, cur, weights, params, config, inputs,
		headDim, nHeads, nKVHeads, 0, 0, 10000.0)
}

// TestAddProjBias_BroadcastsOverTokens verifies the optional q/k/v bias is added
// per out-feature and broadcast across tokens: with an identity-ish projection
// of zero input, the output equals the bias replicated per token.
func TestAddProjBias_BroadcastsOverTokens(t *testing.T) {
	const (
		headDim = 4
		nHeads  = 2
		nTokens = 3
	)
	ctx := ggml.NewGraphContext(8*1024*1024, ggml.AllocPermAllow)
	if ctx == nil {
		t.Fatal("NewGraphContext returned nil")
	}
	defer ctx.Free()

	// Zero-valued 3D tensor [head_dim, n_heads, n_tokens] so the result is the
	// bias alone (broadcast over tokens).
	q := ggml.NewTensor3D(ctx, ggml.TypeF32, headDim, nHeads, nTokens)
	bias := ggml.NewTensor1D(ctx, ggml.TypeF32, headDim*nHeads)
	if q.IsNil() || bias.IsNil() {
		t.Fatal("tensor allocation returned nil")
	}
	fillAxialF32(q, make([]float32, headDim*nHeads*nTokens))
	biasVals := make([]float32, headDim*nHeads)
	for i := range biasVals {
		biasVals[i] = float32(i+1)
	}
	fillAxialF32(bias, biasVals)

	out := addProjBias(ctx, q, bias, headDim, nHeads)
	if out.IsNil() {
		t.Fatal("addProjBias returned nil")
	}
	gf := ggml.NewGraph(ctx, 64)
	gf.BuildForwardExpand(out)
	ggml.GraphCompute(ctx, gf, 1)

	got := readAxialF32(out, headDim*nHeads*nTokens)
	for tok := 0; tok < nTokens; tok++ {
		for f := 0; f < headDim*nHeads; f++ {
			want := biasVals[f]
			g := got[tok*headDim*nHeads+f]
			if math.Abs(float64(g-want)) > 1e-5 {
				t.Errorf("token %d feature %d: got %v want %v (bias not broadcast per-feature)", tok, f, g, want)
			}
		}
	}

	// Nil bias must be a no-op (byte-identical default-off path).
	if noop := addProjBias(ctx, q, ggml.NilTensor(), headDim, nHeads); noop != q {
		t.Error("addProjBias with nil bias must return the input tensor unchanged")
	}
}
