package arch

import (
	"os"
	"path/filepath"
	"testing"
)

// findArchDir walks up from the test directory to find models/arch/.
func findArchDir(t *testing.T) string {
	t.Helper()
	dir, _ := os.Getwd()
	for range 5 {
		d := filepath.Join(dir, "models", "arch")
		if info, err := os.Stat(d); err == nil && info.IsDir() {
			return d
		}
		dir = filepath.Dir(dir)
	}
	t.Skip("models/arch/ not found")
	return ""
}

func TestLoadQwen35(t *testing.T) {
	def, err := Load(findArchDir(t), "qwen35")
	if err != nil {
		t.Fatalf("Load(qwen35): %v", err)
	}

	// Architecture metadata
	if def.Architecture.Name != "qwen35" {
		t.Errorf("name = %q, want qwen35", def.Architecture.Name)
	}
	if !def.Architecture.TiedEmbeddings {
		t.Error("tied_embeddings should be true")
	}

	// Params — GGUF keys
	wantParams := map[string]string{
		"n_layers":           "qwen35.block_count",
		"n_heads":            "qwen35.attention.head_count",
		"n_kv_heads":         "qwen35.attention.head_count_kv",
		"n_embd":             "qwen35.embedding_length",
		"n_ff":               "qwen35.feed_forward_length",
		"rms_eps":            "qwen35.attention.layer_norm_rms_epsilon",
		"head_dim":           "qwen35.attention.key_length",
		"ssm_d_conv":         "qwen35.ssm.conv_kernel",
		"ssm_d_inner":        "qwen35.ssm.inner_size",
		"ssm_d_state":        "qwen35.ssm.state_size",
		"ssm_dt_rank":        "qwen35.ssm.time_step_rank",
		"ssm_n_group":        "qwen35.ssm.group_count",
		"full_attn_interval": "qwen35.full_attention_interval",
		"rope_n_rot":         "qwen35.rope.dimension_count",
		"rope_freq_base":     "qwen35.rope.freq_base",
		"rope_sections":      "qwen35.rope.dimension_sections",
		"rope_mode":          "neox",
	}
	for k, want := range wantParams {
		got, ok := def.Params.Keys[k]
		if !ok {
			t.Errorf("params.Keys missing %q", k)
		} else if got != want {
			t.Errorf("params.Keys[%q] = %q, want %q", k, got, want)
		}
	}
	if len(def.Params.Keys) != len(wantParams) {
		t.Errorf("params.Keys has %d entries, want %d", len(def.Params.Keys), len(wantParams))
	}

	// Derived params
	wantDerived := map[string]string{
		"n_vocab":       "token_embd.ne[1]",
		"head_v_dim":    "ssm_d_inner / ssm_dt_rank",
		"conv_channels": "ssm_d_inner + 2 * ssm_n_group * ssm_d_state",
	}
	for k, want := range wantDerived {
		got, ok := def.Params.Derived[k]
		if !ok {
			t.Errorf("params.Derived missing %q", k)
		} else if got != want {
			t.Errorf("params.Derived[%q] = %q, want %q", k, got, want)
		}
	}

	// Global weights
	wantGlobal := map[string]string{
		"token_embd":  "token_embd.weight",
		"output_norm": "output_norm.weight",
		"output":      "output.weight",
	}
	for k, want := range wantGlobal {
		got, ok := def.Weights.Global[k]
		if !ok {
			t.Errorf("weights.Global missing %q", k)
		} else if got != want {
			t.Errorf("weights.Global[%q] = %q, want %q", k, got, want)
		}
	}

	// Layers
	if def.Layers.Count != "n_layers" {
		t.Errorf("layers.Count = %q, want n_layers", def.Layers.Count)
	}
	if def.Layers.Prefix != "blk.@{layer_idx}." {
		t.Errorf("layers.Prefix = %q, want blk.@{layer_idx}.", def.Layers.Prefix)
	}
	if def.Layers.Routing.Rule != "(@{layer_idx} + 1) % ${full_attn_interval} != 0" {
		t.Errorf("layers.Routing.Rule = %q", def.Layers.Routing.Rule)
	}
	if def.Layers.Routing.IfTrue != "recurrent_ssm" {
		t.Errorf("layers.Routing.IfTrue = %q", def.Layers.Routing.IfTrue)
	}
	if def.Layers.Routing.IfFalse != "full_attention" {
		t.Errorf("layers.Routing.IfFalse = %q", def.Layers.Routing.IfFalse)
	}

	// Common weights
	if def.Layers.CommonWeights["attn_norm"] != "attn_norm.weight" {
		t.Errorf("common_weights.attn_norm = %q", def.Layers.CommonWeights["attn_norm"])
	}
	if def.Layers.CommonWeights["ffn_norm"] != "post_attention_norm.weight" {
		t.Errorf("common_weights.ffn_norm = %q", def.Layers.CommonWeights["ffn_norm"])
	}

	// Blocks
	if len(def.Blocks) != 2 {
		t.Fatalf("expected 2 blocks, got %d", len(def.Blocks))
	}

	fa := def.Blocks["full_attention"]
	if fa.Builder != "full_attention_gated" {
		t.Errorf("full_attention.Builder = %q", fa.Builder)
	}
	if fa.Weights["attn_q"] != "attn_q.weight" {
		t.Errorf("full_attention.Weights[attn_q] = %q", fa.Weights["attn_q"])
	}
	if fa.Config["q_has_gate"] != true {
		t.Errorf("full_attention.Config[q_has_gate] = %v", fa.Config["q_has_gate"])
	}
	if len(fa.Cache) != 2 {
		t.Errorf("full_attention.Cache has %d entries, want 2", len(fa.Cache))
	}
	kCache := fa.Cache["k"]
	if len(kCache.Dims) != 3 || kCache.Dims[0] != "head_dim" || kCache.Dims[1] != "max_seq_len" || kCache.Dims[2] != "n_kv_heads" {
		t.Errorf("k cache dims = %v", kCache.Dims)
	}
	if kCache.Dtype != "f32" {
		t.Errorf("k cache dtype = %q", kCache.Dtype)
	}

	ssm := def.Blocks["recurrent_ssm"]
	if ssm.Builder != "gated_delta_net" {
		t.Errorf("recurrent_ssm.Builder = %q", ssm.Builder)
	}
	if ssm.Weights["ssm_a"] != "ssm_a" {
		t.Errorf("recurrent_ssm.Weights[ssm_a] = %q", ssm.Weights["ssm_a"])
	}
	convCache := ssm.Cache["conv_state"]
	if len(convCache.Dims) != 2 || convCache.Dims[0] != "ssm_d_conv - 1" {
		t.Errorf("conv_state dims = %v", convCache.Dims)
	}

	// FFN
	if def.FFN.Builder != "swiglu" {
		t.Errorf("ffn.Builder = %q", def.FFN.Builder)
	}
	if def.FFN.Weights["gate"] != "ffn_gate.weight" {
		t.Errorf("ffn.Weights[gate] = %q", def.FFN.Weights["gate"])
	}
}

func TestListArchitectures(t *testing.T) {
	names, err := ListArchitectures(findArchDir(t))
	if err != nil {
		t.Fatalf("ListArchitectures: %v", err)
	}
	found := false
	for _, n := range names {
		if n == "qwen35" {
			found = true
		}
	}
	if !found {
		t.Errorf("qwen35 not found in architectures: %v", names)
	}
}

func TestParseInvalid(t *testing.T) {
	_, err := Parse([]byte(`[architecture]`))
	if err == nil {
		t.Error("expected error for missing architecture.name")
	}
}
