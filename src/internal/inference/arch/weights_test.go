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
	if rw.Global["token_embd"] != "token_embd.weight" {
		t.Errorf("global token_embd = %q", rw.Global["token_embd"])
	}
	if rw.Global["output_norm"] != "output_norm.weight" {
		t.Errorf("global output_norm = %q", rw.Global["output_norm"])
	}
	if rw.Global["output"] != "output.weight" {
		t.Errorf("global output = %q", rw.Global["output"])
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
	if l0.Common["attn_norm"] != "blk.0.attn_norm.weight" {
		t.Errorf("layer 0 attn_norm = %q", l0.Common["attn_norm"])
	}
	if l0.Common["ffn_norm"] != "blk.0.post_attention_norm.weight" {
		t.Errorf("layer 0 ffn_norm = %q", l0.Common["ffn_norm"])
	}
	if l0.Block["attn_qkv"] != "blk.0.attn_qkv.weight" {
		t.Errorf("layer 0 attn_qkv = %q", l0.Block["attn_qkv"])
	}
	if l0.Block["ssm_a"] != "blk.0.ssm_a" {
		t.Errorf("layer 0 ssm_a = %q", l0.Block["ssm_a"])
	}
	if l0.FFN["gate"] != "blk.0.ffn_gate.weight" {
		t.Errorf("layer 0 ffn_gate = %q", l0.FFN["gate"])
	}

	// Layer 3: full attention
	l3 := rw.Layers[3]
	if l3.BlockName != "full_attention" {
		t.Errorf("layer 3 block = %q, want full_attention", l3.BlockName)
	}
	if l3.Block["attn_q"] != "blk.3.attn_q.weight" {
		t.Errorf("layer 3 attn_q = %q", l3.Block["attn_q"])
	}
	if l3.Block["attn_k"] != "blk.3.attn_k.weight" {
		t.Errorf("layer 3 attn_k = %q", l3.Block["attn_k"])
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
	if l31.Block["attn_output"] != "blk.31.attn_output.weight" {
		t.Errorf("layer 31 attn_output = %q", l31.Block["attn_output"])
	}
	if l31.FFN["down"] != "blk.31.ffn_down.weight" {
		t.Errorf("layer 31 ffn_down = %q", l31.FFN["down"])
	}
}
