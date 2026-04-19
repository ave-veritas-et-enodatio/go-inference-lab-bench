package arch

import (
	"testing"
)

func TestResolveWeights(t *testing.T) {
	def, err := Load(findArchDir(t), "qwen35")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	rp, err := ResolveParams(def, newQwen35MockGGUF())
	if err != nil {
		t.Fatalf("ResolveParams: %v", err)
	}

	rw, err := ResolveWeights(def, rp)
	if err != nil {
		t.Fatalf("ResolveWeights: %v", err)
	}

	// Global weights
	if rw.Global[WeightTokenEmbd] != "token_embd.weight" {
		t.Errorf("global token_embd = %q", rw.Global[WeightTokenEmbd])
	}
	if rw.Global[WeightOutputNorm] != "output_norm.weight" {
		t.Errorf("global output_norm = %q", rw.Global[WeightOutputNorm])
	}
	if rw.Global[WeightOutput] != "output.weight" {
		t.Errorf("global output = %q", rw.Global[WeightOutput])
	}

	// Should have 32 layers
	if len(rw.Layers) != 32 {
		t.Fatalf("expected 32 layers, got %d", len(rw.Layers))
	}

	// Layer 0: SSM (recurrent)
	l0 := rw.Layers[0]
	if l0.BlockName != "recurrent_ssm" {
		t.Errorf("layer 0 block = %q, want recurrent_ssm", l0.BlockName)
	}
	if l0.Common[WeightAttnNorm] != "blk.0.attn_norm.weight" {
		t.Errorf("layer 0 attn_norm = %q", l0.Common[WeightAttnNorm])
	}
	if l0.Common[WeightFFNNorm] != "blk.0.post_attention_norm.weight" {
		t.Errorf("layer 0 ffn_norm = %q", l0.Common[WeightFFNNorm])
	}
	if l0.Block[WeightAttnQKV] != "blk.0.attn_qkv.weight" {
		t.Errorf("layer 0 attn_qkv = %q", l0.Block[WeightAttnQKV])
	}
	if l0.Block[WeightSSMA] != "blk.0.ssm_a" {
		t.Errorf("layer 0 ssm_a = %q", l0.Block[WeightSSMA])
	}
	if l0.FFN["gate"] != "blk.0.ffn_gate.weight" {
		t.Errorf("layer 0 ffn_gate = %q", l0.FFN["gate"])
	}

	// Layer 3: full attention
	l3 := rw.Layers[3]
	if l3.BlockName != "full_attention" {
		t.Errorf("layer 3 block = %q, want full_attention", l3.BlockName)
	}
	if l3.Block[WeightAttnQ] != "blk.3.attn_q.weight" {
		t.Errorf("layer 3 attn_q = %q", l3.Block[WeightAttnQ])
	}
	if l3.Block[WeightAttnK] != "blk.3.attn_k.weight" {
		t.Errorf("layer 3 attn_k = %q", l3.Block[WeightAttnK])
	}

	// Verify block assignment pattern across all 32 layers
	for i := range 32 {
		lw := rw.Layers[i]
		expectRecurrent := (i+1)%4 != 0
		if expectRecurrent {
			if lw.BlockName != "recurrent_ssm" {
				t.Errorf("layer %d: block = %q, want recurrent_ssm", i, lw.BlockName)
			}
		} else {
			if lw.BlockName != "full_attention" {
				t.Errorf("layer %d: block = %q, want full_attention", i, lw.BlockName)
			}
		}
	}

	// Layer 31: full attention (last layer)
	l31 := rw.Layers[31]
	if l31.BlockName != "full_attention" {
		t.Errorf("layer 31 block = %q, want full_attention", l31.BlockName)
	}
	if l31.Block[WeightAttnOutput] != "blk.31.attn_output.weight" {
		t.Errorf("layer 31 attn_output = %q", l31.Block[WeightAttnOutput])
	}
	if l31.FFN["down"] != "blk.31.ffn_down.weight" {
		t.Errorf("layer 31 ffn_down = %q", l31.FFN["down"])
	}
}
