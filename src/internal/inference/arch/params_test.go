package arch

import (
	"testing"
)

// mockGGUF simulates a Qwen3.5-4B GGUF file.
type mockGGUF struct {
	u32s    map[string]uint32
	f32s    map[string]float32
	arrs    map[string][]int
	bools   map[string][]bool
	tensors map[string][]int64 // tensor name → ne dims
}

func (m *mockGGUF) GetU32(key string) (uint32, bool)     { v, ok := m.u32s[key]; return v, ok }
func (m *mockGGUF) GetF32(key string) (float32, bool)    { v, ok := m.f32s[key]; return v, ok }
func (m *mockGGUF) GetArrInts(key string) ([]int, bool)  { v, ok := m.arrs[key]; return v, ok }
func (m *mockGGUF) GetArrBools(key string) ([]bool, bool) { v, ok := m.bools[key]; return v, ok }
func (m *mockGGUF) GetTensorDim(name string, dim int) (int64, bool) {
	dims, ok := m.tensors[name]
	if !ok || dim >= len(dims) {
		return 0, false
	}
	return dims[dim], true
}

func newQwen35MockGGUF() *mockGGUF {
	return &mockGGUF{
		u32s: map[string]uint32{
			"qwen35.block_count":              32,
			"qwen35.attention.head_count":     16,
			"qwen35.attention.head_count_kv":  4,
			"qwen35.embedding_length":         2560,
			"qwen35.feed_forward_length":      9216,
			"qwen35.attention.key_length":     256,
			"qwen35.ssm.conv_kernel":          4,
			"qwen35.ssm.inner_size":           4096,
			"qwen35.ssm.state_size":           128,
			"qwen35.ssm.time_step_rank":       32,
			"qwen35.ssm.group_count":          16,
			"qwen35.full_attention_interval":   4,
			"qwen35.rope.dimension_count":     64,
		},
		f32s: map[string]float32{
			"qwen35.attention.layer_norm_rms_epsilon": 1e-6,
			"qwen35.rope.freq_base":                   10000000.0,
		},
		arrs: map[string][]int{
			"qwen35.rope.dimension_sections": {11, 11, 10, 0},
		},
		tensors: map[string][]int64{
			"token_embd": {2560, 248320},
		},
	}
}

func TestResolveParams(t *testing.T) {
	def, err := Load(findArchDir(t), "qwen35")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	rp, err := ResolveParams(def, newQwen35MockGGUF())
	if err != nil {
		t.Fatalf("ResolveParams: %v", err)
	}

	// Check GGUF-sourced integer params
	wantInts := map[string]int{
		"n_layers":           32,
		"n_heads":            16,
		"n_kv_heads":         4,
		"n_embd":             2560,
		"n_ff":               9216,
		"head_dim":           256,
		"ssm_d_conv":         4,
		"ssm_d_inner":        4096,
		"ssm_d_state":        128,
		"ssm_dt_rank":        32,
		"ssm_n_group":        16,
		"full_attn_interval": 4,
		"rope_n_rot":         64,
	}
	for name, want := range wantInts {
		got, err := rp.GetInt(name)
		if err != nil {
			t.Errorf("GetInt(%q): %v", name, err)
			continue
		}
		if got != want {
			t.Errorf("GetInt(%q) = %d, want %d", name, got, want)
		}
	}

	// Check float params
	wantFloats := map[string]float32{
		"rms_eps":        1e-6,
		"rope_freq_base": 10000000.0,
	}
	for name, want := range wantFloats {
		got, err := rp.GetFloat(name)
		if err != nil {
			t.Errorf("GetFloat(%q): %v", name, err)
			continue
		}
		if got != want {
			t.Errorf("GetFloat(%q) = %v, want %v", name, got, want)
		}
	}

	// Check string params
	ropeMode, err := rp.GetString("rope_mode")
	if err != nil {
		t.Errorf("GetString(rope_mode): %v", err)
	} else if ropeMode != "neox" {
		t.Errorf("rope_mode = %q, want neox", ropeMode)
	}

	// Check int array
	sections, err := rp.GetIntArr("rope_sections")
	if err != nil {
		t.Errorf("GetIntArr(rope_sections): %v", err)
	} else if len(sections) != 4 || sections[0] != 11 || sections[1] != 11 || sections[2] != 10 || sections[3] != 0 {
		t.Errorf("rope_sections = %v, want [11 11 10 0]", sections)
	}

	// Check derived params
	wantDerived := map[string]int{
		"n_vocab":       248320,
		"head_v_dim":    128,  // 4096 / 32
		"conv_channels": 8192, // 4096 + 2*16*128
	}
	for name, want := range wantDerived {
		got, err := rp.GetInt(name)
		if err != nil {
			t.Errorf("GetInt(%q): %v", name, err)
			continue
		}
		if got != want {
			t.Errorf("derived %q = %d, want %d", name, got, want)
		}
	}
}

func TestEvalRoutingRule(t *testing.T) {
	rp := &ResolvedParams{
		Ints: map[string]int{"full_attn_interval": 4},
	}

	// Qwen3.5 routing: (@{layer_idx} + 1) % ${full_attn_interval} != 0
	// Full attention at layers 3, 7, 11, 15, 19, 23, 27, 31 (rule=false → if_false)
	// SSM at all other layers (rule=true → if_true)
	rule := "(@{layer_idx} + 1) % ${full_attn_interval} != 0"

	for i := 0; i < 32; i++ {
		isRecurrent, err := EvalRoutingRule(rule, i, rp)
		if err != nil {
			t.Fatalf("layer %d: %v", i, err)
		}
		wantRecurrent := (i+1)%4 != 0
		if isRecurrent != wantRecurrent {
			t.Errorf("layer %d: isRecurrent=%v, want %v", i, isRecurrent, wantRecurrent)
		}
	}
}

func TestEvalExpr(t *testing.T) {
	rp := &ResolvedParams{
		Ints: map[string]int{"a": 10, "b": 3},
	}

	tests := []struct {
		expr string
		want int
	}{
		{"a + b", 13},
		{"a - b", 7},
		{"a * b", 30},
		{"a / b", 3},
		{"a % b", 1},
		{"(a + 1) * 2", 22},
		{"a + 2 * b", 16},
	}
	for _, tt := range tests {
		got, err := evalExpr(tt.expr, rp)
		if err != nil {
			t.Errorf("evalExpr(%q): %v", tt.expr, err)
			continue
		}
		if got != tt.want {
			t.Errorf("evalExpr(%q) = %d, want %d", tt.expr, got, tt.want)
		}
	}
}
