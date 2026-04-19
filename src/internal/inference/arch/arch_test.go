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
		ParamNLayers:          "qwen35.block_count",
		ParamNHeads:           "qwen35.attention.head_count",
		ParamNKVHeads:         "qwen35.attention.head_count_kv",
		ParamNEmbd:            "qwen35.embedding_length",
		"n_ff":                "qwen35.feed_forward_length",
		ParamRMSEps:           "qwen35.attention.layer_norm_rms_epsilon",
		ParamHeadDim:          "qwen35.attention.key_length",
		ParamSSMDConv:         "qwen35.ssm.conv_kernel",
		ParamSSMDInner:        "qwen35.ssm.inner_size",
		ParamSSMDState:        "qwen35.ssm.state_size",
		ParamSSMDTRank:        "qwen35.ssm.time_step_rank",
		ParamSSMNGroup:        "qwen35.ssm.group_count",
		ParamFullAttnInterval: "qwen35.full_attention_interval",
		ParamRoPENRot:         "qwen35.rope.dimension_count",
		ParamRoPEFreqBase:     "qwen35.rope.freq_base",
		ParamRoPESections:     "qwen35.rope.dimension_sections",
		"rope_mode":           RopeNeox,
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
		ParamNVocab:       "token_embd.ne[1]",
		ParamHeadVDim:     "ssm_d_inner / ssm_dt_rank",
		ParamConvChannels: "ssm_d_inner + 2 * ssm_n_group * ssm_d_state",
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
		WeightTokenEmbd:  "token_embd.weight",
		WeightOutputNorm: "output_norm.weight",
		WeightOutput:     "output.weight",
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
	if def.Layers.Count != ParamNLayers {
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
	if def.Layers.CommonWeights[WeightAttnNorm] != "attn_norm.weight" {
		t.Errorf("common_weights.attn_norm = %q", def.Layers.CommonWeights[WeightAttnNorm])
	}
	if def.Layers.CommonWeights[WeightFFNNorm] != "post_attention_norm.weight" {
		t.Errorf("common_weights.ffn_norm = %q", def.Layers.CommonWeights[WeightFFNNorm])
	}

	// Blocks
	if len(def.Blocks) != 2 {
		t.Fatalf("expected 2 blocks, got %d", len(def.Blocks))
	}

	fa := def.Blocks["full_attention"]
	if fa.Builder != "full_attention_gated" {
		t.Errorf("full_attention.Builder = %q", fa.Builder)
	}
	if fa.Weights[WeightAttnQ] != "attn_q.weight" {
		t.Errorf("full_attention.Weights[attn_q] = %q", fa.Weights[WeightAttnQ])
	}
	if fa.Config[ConfigQHasGate] != true {
		t.Errorf("full_attention.Config[q_has_gate] = %v", fa.Config[ConfigQHasGate])
	}
	if len(fa.Cache) != 2 {
		t.Errorf("full_attention.Cache has %d entries, want 2", len(fa.Cache))
	}
	kCache := fa.Cache[CacheK]
	if len(kCache.Dims) != 3 || kCache.Dims[0] != ParamHeadDim || kCache.Dims[1] != CacheDimMaxSeqLen || kCache.Dims[2] != ParamNKVHeads {
		t.Errorf("k cache dims = %v", kCache.Dims)
	}
	if kCache.Dtype != "f32" {
		t.Errorf("k cache dtype = %q", kCache.Dtype)
	}

	ssm := def.Blocks["recurrent_ssm"]
	if ssm.Builder != "gated_delta_net" {
		t.Errorf("recurrent_ssm.Builder = %q", ssm.Builder)
	}
	if ssm.Weights[WeightSSMA] != "ssm_a" {
		t.Errorf("recurrent_ssm.Weights[ssm_a] = %q", ssm.Weights[WeightSSMA])
	}
	convCache := ssm.Cache[CacheConvState]
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
