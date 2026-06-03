package arch

import (
	"testing"
)

// TestParseVisionAndProjector covers the stmap vision extension: a stmap
// file with [vision] and [projector] blocks parses into matching ArchSTMap
// fields, and the decoder-side fields stay unaffected. Schema details
// follow ARCHITECTURE.md "Vision / Multimodal Subsystem → Construction
// Across Two Formats".
func TestParseVisionAndProjector(t *testing.T) {
	dir, name := writeStmap(t, `
[architecture]
hf_class = "FooVLMForConditionalGeneration"

[params]
"text_config.num_hidden_layers" = "foo.block_count"

[layer_prefix]
hf = "model.language_model.layers.{N}."

[tensors]
"attn_q.weight" = "self_attn.q_proj.weight"

[tensors.global]
"token_embd.weight" = "model.language_model.embed_tokens.weight"

[vision]

[vision.layer_prefix]
hf   = "model.vision_tower.encoder.layers.{N}."
gguf = "v.blk.@{layer_idx}."

[vision.tensors]
"attn_q.weight" = "self_attn.q_proj.linear.weight"
"ln1.weight"    = "input_layernorm.weight"

[vision.tensors.global]
"v.patch_embd.weight" = "model.vision_tower.patch_embedder.input_proj.weight"

[vision.params]
"vision_config.num_hidden_layers" = "vision.block_count"
"vision_config.hidden_size"       = "vision.embedding_length"

[projector]

[projector.tensors.global]
"mm.input_projection.weight" = "model.embed_vision.embedding_projection.weight"
`)

	sm, err := LoadArchSTMap(dir, name)
	if err != nil {
		t.Fatalf("LoadArchSTMap: %v", err)
	}
	if sm == nil {
		t.Fatal("LoadArchSTMap returned nil")
	}

	// Decoder side untouched.
	if got := sm.LayerPrefixHF; got != "model.language_model.layers.{N}." {
		t.Errorf("decoder LayerPrefixHF = %q, want decoder template", got)
	}
	if sm.Tensors["attn_q.weight"] != "self_attn.q_proj.weight" {
		t.Errorf("decoder tensors map: got %q", sm.Tensors["attn_q.weight"])
	}

	// Vision block present and structurally correct.
	if sm.Vision == nil {
		t.Fatal("Vision is nil; expected populated block")
	}
	if got := sm.Vision.LayerPrefixHF; got != "model.vision_tower.encoder.layers.{N}." {
		t.Errorf("vision LayerPrefixHF = %q", got)
	}
	if got := sm.Vision.LayerPrefixGGUF; got != "v.blk.@{layer_idx}." {
		t.Errorf("vision LayerPrefixGGUF = %q", got)
	}
	if got := sm.Vision.Tensors["attn_q.weight"]; got != "self_attn.q_proj.linear.weight" {
		t.Errorf("vision attn_q map: got %q", got)
	}
	if got := sm.Vision.Tensors["ln1.weight"]; got != "input_layernorm.weight" {
		t.Errorf("vision ln1 map: got %q", got)
	}
	if got := sm.Vision.GlobalTensors["v.patch_embd.weight"]; got != "model.vision_tower.patch_embedder.input_proj.weight" {
		t.Errorf("vision patch_embd global: got %q", got)
	}
	if got := sm.Vision.Params["vision_config.num_hidden_layers"]; got != "vision.block_count" {
		t.Errorf("vision params: got %q", got)
	}

	// Projector block.
	if sm.Projector == nil {
		t.Fatal("Projector is nil; expected populated block")
	}
	if got := sm.Projector.GlobalTensors["mm.input_projection.weight"]; got != "model.embed_vision.embedding_projection.weight" {
		t.Errorf("projector mapping: got %q", got)
	}
}

// TestParseVisionGGUFPrefixDefault verifies the default vision GGUF prefix
// kicks in when [vision.layer_prefix.gguf] is omitted from the TOML.
func TestParseVisionGGUFPrefixDefault(t *testing.T) {
	dir, name := writeStmap(t, `
[architecture]
hf_class = "FooVLM"

[vision]

[vision.layer_prefix]
hf = "model.vision_tower.encoder.layers.{N}."

[vision.tensors]
"attn_q.weight" = "self_attn.q_proj.weight"
`)
	sm, err := LoadArchSTMap(dir, name)
	if err != nil {
		t.Fatalf("LoadArchSTMap: %v", err)
	}
	if sm.Vision == nil {
		t.Fatal("Vision nil")
	}
	if got := sm.Vision.LayerPrefixGGUF; got != "v.blk.@{layer_idx}." {
		t.Errorf("vision LayerPrefixGGUF default = %q, want %q", got, "v.blk.@{layer_idx}.")
	}
}

// TestParseVisionMissingPrefixIsError confirms a [vision.tensors] table
// with no [vision.layer_prefix.hf] surfaces a named error rather than
// silently dropping the per-layer entries.
func TestParseVisionMissingPrefixIsError(t *testing.T) {
	dir, name := writeStmap(t, `
[architecture]
hf_class = "FooVLM"

[vision]

[vision.tensors]
"attn_q.weight" = "self_attn.q_proj.weight"
`)
	_, err := LoadArchSTMap(dir, name)
	if err == nil {
		t.Fatal("expected error for missing vision.layer_prefix.hf, got nil")
	}
}

// TestParseUnimodalUnchanged confirms a stmap without [vision] or
// [projector] still parses cleanly and leaves both fields nil.
func TestParseUnimodalUnchanged(t *testing.T) {
	dir, name := writeStmap(t, `
[architecture]
hf_class = "FooForCausalLM"

[layer_prefix]
hf = "model.layers.{N}."

[tensors]
"attn_q.weight" = "self_attn.q_proj.weight"
`)
	sm, err := LoadArchSTMap(dir, name)
	if err != nil {
		t.Fatalf("LoadArchSTMap: %v", err)
	}
	if sm.Vision != nil {
		t.Errorf("Vision should be nil for unimodal stmap")
	}
	if sm.Projector != nil {
		t.Errorf("Projector should be nil for unimodal stmap")
	}
}
