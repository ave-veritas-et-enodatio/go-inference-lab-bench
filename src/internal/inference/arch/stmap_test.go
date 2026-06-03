package arch

import (
	"os"
	"path/filepath"
	"testing"
)

// writeStmap writes a minimal stmap TOML and returns the temp directory and
// arch name.
func writeStmap(t *testing.T, content string) (archDir, archName string) {
	t.Helper()
	dir := t.TempDir()
	archName = "testarch"
	path := filepath.Join(dir, archName+extArchSTMapToml)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write stmap: %v", err)
	}
	return dir, archName
}

func TestParseDerivedMetadata(t *testing.T) {
	dir, name := writeStmap(t, `
[architecture]
hf_class = "TestForCausalLM"

[layer_prefix]
hf = "model.layers.{N}."

[[derived_metadata]]
target = "testarch.attention.sliding_window_pattern"
op     = "string_array_eq"
source = "text_config.layer_types"
match  = "sliding_attention"
`)
	got, err := LoadArchSTMap(dir, name)
	if err != nil {
		t.Fatalf("LoadArchSTMap: %v", err)
	}
	if len(got.DerivedMetadata) != 1 {
		t.Fatalf("DerivedMetadata len = %d, want 1", len(got.DerivedMetadata))
	}
	d := got.DerivedMetadata[0]
	if d.Target != "testarch.attention.sliding_window_pattern" {
		t.Errorf("Target = %q", d.Target)
	}
	if d.Op != "string_array_eq" {
		t.Errorf("Op = %q", d.Op)
	}
	if d.Params["source"] != "text_config.layer_types" {
		t.Errorf("Params[source] = %v", d.Params["source"])
	}
	if d.Params["match"] != "sliding_attention" {
		t.Errorf("Params[match] = %v", d.Params["match"])
	}
	// op and target must not leak into Params.
	if _, ok := d.Params["op"]; ok {
		t.Error("Params should not contain 'op'")
	}
	if _, ok := d.Params["target"]; ok {
		t.Error("Params should not contain 'target'")
	}
}

func TestParseDerivedTensors(t *testing.T) {
	dir, name := writeStmap(t, `
[architecture]
hf_class = "TestForCausalLM"

[layer_prefix]
hf = "model.layers.{N}."

[[derived_tensors]]
target = "rope_freqs.weight"
op     = "rope_freqs_proportional"
head_dim_source       = "text_config.global_head_dim"
partial_rotary_source = "text_config.rope_parameters.full_attention.partial_rotary_factor"
`)
	got, err := LoadArchSTMap(dir, name)
	if err != nil {
		t.Fatalf("LoadArchSTMap: %v", err)
	}
	if len(got.DerivedTensors) != 1 {
		t.Fatalf("DerivedTensors len = %d, want 1", len(got.DerivedTensors))
	}
	d := got.DerivedTensors[0]
	if d.Target != "rope_freqs.weight" {
		t.Errorf("Target = %q", d.Target)
	}
	if d.Op != "rope_freqs_proportional" {
		t.Errorf("Op = %q", d.Op)
	}
	if d.Params["head_dim_source"] != "text_config.global_head_dim" {
		t.Errorf("Params[head_dim_source] = %v", d.Params["head_dim_source"])
	}
}

func TestParseDerivedMetadata_MissingOp(t *testing.T) {
	dir, name := writeStmap(t, `
[architecture]
hf_class = "TestForCausalLM"

[layer_prefix]
hf = "model.layers.{N}."

[[derived_metadata]]
target = "testarch.foo"
`)
	_, err := LoadArchSTMap(dir, name)
	if err == nil {
		t.Fatal("expected error for missing op, got nil")
	}
}

func TestParseDerivedMetadata_MissingTarget(t *testing.T) {
	dir, name := writeStmap(t, `
[architecture]
hf_class = "TestForCausalLM"

[layer_prefix]
hf = "model.layers.{N}."

[[derived_metadata]]
op = "string_array_eq"
`)
	_, err := LoadArchSTMap(dir, name)
	if err == nil {
		t.Fatal("expected error for missing target, got nil")
	}
}

func TestParseDerivedTensors_EmptyOp(t *testing.T) {
	dir, name := writeStmap(t, `
[architecture]
hf_class = "TestForCausalLM"

[layer_prefix]
hf = "model.layers.{N}."

[[derived_tensors]]
target = "foo.weight"
op     = ""
`)
	_, err := LoadArchSTMap(dir, name)
	if err == nil {
		t.Fatal("expected error for empty op, got nil")
	}
}

func TestParseStmap_NoDerivedBlocks(t *testing.T) {
	// Existing stmaps without [[derived_*]] blocks must still parse cleanly
	// with the new fields nil.
	dir, name := writeStmap(t, `
[architecture]
hf_class = "TestForCausalLM"

[layer_prefix]
hf = "model.layers.{N}."

[params]
"text_config.num_hidden_layers" = "testarch.block_count"

[tensors]
"attn_q.weight" = "self_attn.q_proj.weight"
`)
	got, err := LoadArchSTMap(dir, name)
	if err != nil {
		t.Fatalf("LoadArchSTMap: %v", err)
	}
	if got.DerivedMetadata != nil {
		t.Errorf("DerivedMetadata = %v, want nil", got.DerivedMetadata)
	}
	if got.DerivedTensors != nil {
		t.Errorf("DerivedTensors = %v, want nil", got.DerivedTensors)
	}
}
