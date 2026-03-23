package arch

import "testing"

func TestBuilderRegistry(t *testing.T) {
	// Block builders
	for _, name := range []string{"attention", "full_attention_gated", "mla_attention", "gated_delta_net"} {
		b, ok := GetBlockBuilder(name)
		if !ok {
			t.Errorf("GetBlockBuilder(%q) not found", name)
		}
		if b == nil {
			t.Errorf("GetBlockBuilder(%q) returned nil", name)
		}
	}

	_, ok := GetBlockBuilder("nonexistent")
	if ok {
		t.Error("GetBlockBuilder(nonexistent) should return false")
	}

	// FFN builders
	for _, name := range []string{"swiglu", "moe_with_shared"} {
		b, ok := GetFFNBuilder(name)
		if !ok || b == nil {
			t.Errorf("GetFFNBuilder(%q) not found", name)
		}
	}

	_, ok = GetFFNBuilder("nonexistent")
	if ok {
		t.Error("GetFFNBuilder(nonexistent) should return false")
	}
}
