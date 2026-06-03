package arch

import (
	"testing"

	"inference-lab-bench/internal/ggml"
)

// TestSelectMask_NonCausal verifies that ConfigNonCausal short-circuits
// mask selection. Used by the vision-tower attention block (Phase 2 of the
// vision-input plan) to suppress the causal triangle. SoftMaxExt and
// FlashAttnExt both tolerate a nil mask, so callers don't need any
// further guard.
func TestSelectMask_NonCausal(t *testing.T) {
	// Use a real (non-nil) tensor for InpMask so a regression that returns
	// it instead of NilTensor is detectable.
	ctx := ggml.NewGraphContext(1024*1024, ggml.AllocPermAllow)
	if ctx == nil {
		t.Fatal("NewGraphContext returned nil")
	}
	defer ctx.Free()
	realMask := ggml.NewTensor1D(ctx, ggml.TypeF32, 4)
	if realMask.IsNil() {
		t.Fatal("tensor allocation returned nil")
	}
	inputs := &GraphInputs{InpMask: realMask}

	got := selectMask(map[string]any{ConfigNonCausal: true}, inputs)
	if !got.IsNil() {
		t.Errorf("non_causal=true should return NilTensor, got non-nil")
	}

	// String form is what TOML round-trips deliver; configBoolOr accepts
	// either, so this asserts the typed path too.
	got = selectMask(map[string]any{ConfigNonCausal: "true"}, inputs)
	if !got.IsNil() {
		t.Errorf("non_causal=\"true\" should return NilTensor, got non-nil")
	}

	// Without the flag (or with it explicitly false) the standard mask wins.
	got = selectMask(map[string]any{}, inputs)
	if got.IsNil() {
		t.Errorf("default mask selection should return InpMask, got NilTensor")
	}
	got = selectMask(map[string]any{ConfigNonCausal: false}, inputs)
	if got.IsNil() {
		t.Errorf("non_causal=false should return InpMask, got NilTensor")
	}
}

// TestSelectMask_NonCausalBeatsSWA confirms the precedence order: a block
// declaring both non_causal and sliding_window resolves to NilTensor. The
// TOML loader is expected to reject this combination at parse time, but
// the runtime tie-breaker keeps behavior well-defined regardless.
func TestSelectMask_NonCausalBeatsSWA(t *testing.T) {
	ctx := ggml.NewGraphContext(1024*1024, ggml.AllocPermAllow)
	if ctx == nil {
		t.Fatal("NewGraphContext returned nil")
	}
	defer ctx.Free()
	inpMask := ggml.NewTensor1D(ctx, ggml.TypeF32, 4)
	swaMask := ggml.NewTensor1D(ctx, ggml.TypeF32, 4)
	inputs := &GraphInputs{InpMask: inpMask, InpMaskSWA: swaMask}

	got := selectMask(map[string]any{
		ConfigNonCausal:    true,
		ParamSlidingWindow: true,
	}, inputs)
	if !got.IsNil() {
		t.Errorf("non_causal should beat sliding_window, got non-nil mask")
	}
}

// TestAttentionContract_AcceptsNewConfig confirms the AttentionBuilder
// Contract surfaces the new ConfigNonCausal and RopeNone entries so that
// TOML loaders accept them in .arch.toml [blocks.*.config] sections.
func TestAttentionContract_AcceptsNewConfig(t *testing.T) {
	b, ok := GetBlockBuilder("attention")
	if !ok {
		t.Fatal("attention builder not registered")
	}
	schema := b.Contract().ConfigSchema

	if _, ok := schema[ConfigNonCausal]; !ok {
		t.Errorf("ConfigNonCausal (%q) missing from attention ConfigSchema", ConfigNonCausal)
	}
	ropeAllowed, ok := schema[ConfigRope]
	if !ok {
		t.Fatalf("ConfigRope (%q) missing from attention ConfigSchema", ConfigRope)
	}
	hasNone := false
	for _, v := range ropeAllowed {
		if v == RopeNone {
			hasNone = true
			break
		}
	}
	if !hasNone {
		t.Errorf("attention rope schema %v missing %q", ropeAllowed, RopeNone)
	}
}

// TestAttentionContract_KQPrec confirms the ConfigKQPrec knob is surfaced in
// the AttentionBuilder ConfigSchema with the expected allowed values, so TOML
// loaders accept kq_prec in [blocks.*.config] sections.
func TestAttentionContract_KQPrec(t *testing.T) {
	b, ok := GetBlockBuilder("attention")
	if !ok {
		t.Fatal("attention builder not registered")
	}
	allowed, ok := b.Contract().ConfigSchema[ConfigKQPrec]
	if !ok {
		t.Fatalf("ConfigKQPrec (%q) missing from attention ConfigSchema", ConfigKQPrec)
	}
	want := map[string]bool{KQPrecNative: false, "": false}
	for _, v := range allowed {
		if _, expected := want[v]; !expected {
			t.Errorf("ConfigKQPrec schema has unexpected value %q (allowed: %v)", v, allowed)
			continue
		}
		want[v] = true
	}
	for v, seen := range want {
		if !seen {
			t.Errorf("ConfigKQPrec schema %v missing expected value %q", allowed, v)
		}
	}
}

// TestKQForceF32 verifies the pure knob resolver: the default (unset / empty /
// any non-native value) forces F32 accumulation — preserving today's decoder
// behavior — while kq_prec="native" opts out.
func TestKQForceF32(t *testing.T) {
	tests := []struct {
		name   string
		config map[string]any
		want   bool
	}{
		{"nil config defaults to F32-on", nil, true},
		{"empty config defaults to F32-on", map[string]any{}, true},
		{"unrelated key defaults to F32-on", map[string]any{ConfigRope: RopeNeox}, true},
		{"explicit empty string is F32-on", map[string]any{ConfigKQPrec: ""}, true},
		{"native opts out of F32 force", map[string]any{ConfigKQPrec: KQPrecNative}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := kqForceF32(tt.config); got != tt.want {
				t.Errorf("kqForceF32(%v) = %v, want %v", tt.config, got, tt.want)
			}
		})
	}
}
