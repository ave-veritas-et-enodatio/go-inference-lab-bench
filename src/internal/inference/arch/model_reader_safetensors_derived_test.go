package arch

import (
	"bytes"
	"encoding/json"
	"testing"
)

// configJSONFromString parses a JSON snippet via the same UseNumber decoder
// path that the real safetensors loader uses, so numeric handling matches.
func configJSONFromString(t *testing.T, s string) map[string]any {
	t.Helper()
	dec := json.NewDecoder(bytes.NewReader([]byte(s)))
	dec.UseNumber()
	var out map[string]any
	if err := dec.Decode(&out); err != nil {
		t.Fatalf("config JSON parse: %v", err)
	}
	return out
}

func TestStringArrayEq_Gemma4LayerTypes(t *testing.T) {
	cfg := configJSONFromString(t, `{
		"text_config": {
			"layer_types": [
				"sliding_attention",
				"sliding_attention",
				"full_attention",
				"sliding_attention",
				"full_attention"
			]
		}
	}`)
	spec := DerivedMetadataSpec{
		Target: "gemma4.attention.sliding_window_pattern",
		Op:     "string_array_eq",
		Params: map[string]any{
			"source": "text_config.layer_types",
			"match":  "sliding_attention",
		},
	}
	got, err := handleStringArrayEq(spec, cfg)
	if err != nil {
		t.Fatalf("handler error: %v", err)
	}
	arr, ok := got.([]any)
	if !ok {
		t.Fatalf("output is %T, want []any", got)
	}
	want := []bool{true, true, false, true, false}
	if len(arr) != len(want) {
		t.Fatalf("len = %d, want %d", len(arr), len(want))
	}
	for i := range want {
		b, ok := arr[i].(bool)
		if !ok {
			t.Fatalf("arr[%d] is %T, want bool", i, arr[i])
		}
		if b != want[i] {
			t.Errorf("arr[%d] = %v, want %v", i, b, want[i])
		}
	}
	// Cross-check: the result must round-trip through toBoolArr, which is
	// the path GetArrBools uses.
	if bools, ok := toBoolArr(got); !ok {
		t.Error("toBoolArr rejected the handler output")
	} else if len(bools) != len(want) {
		t.Errorf("toBoolArr len = %d, want %d", len(bools), len(want))
	}
}

func TestStringArrayEq_MissingSourceKey(t *testing.T) {
	cfg := configJSONFromString(t, `{"text_config": {}}`)
	spec := DerivedMetadataSpec{
		Target: "foo",
		Op:     "string_array_eq",
		Params: map[string]any{"source": "text_config.layer_types", "match": "x"},
	}
	if _, err := handleStringArrayEq(spec, cfg); err == nil {
		t.Fatal("expected error for missing source key")
	}
}

func TestStringArrayEq_NonStringElement(t *testing.T) {
	cfg := configJSONFromString(t, `{"a": ["x", 5, "y"]}`)
	spec := DerivedMetadataSpec{
		Op: "string_array_eq",
		Params: map[string]any{"source": "a", "match": "x"},
	}
	if _, err := handleStringArrayEq(spec, cfg); err == nil {
		t.Fatal("expected error for non-string element")
	}
}

func TestRopeFreqsProportional_Gemma4E4B(t *testing.T) {
	// Gemma 4 E4B: global_head_dim=512, partial_rotary_factor=0.25
	// Expected: 64 × 1.0 then 192 × 1e30, total length 256.
	cfg := configJSONFromString(t, `{
		"text_config": {
			"global_head_dim": 512,
			"rope_parameters": {
				"full_attention": {
					"partial_rotary_factor": 0.25
				}
			}
		}
	}`)
	spec := DerivedTensorSpec{
		Target: "rope_freqs.weight",
		Op:     "rope_freqs_proportional",
		Params: map[string]any{
			"head_dim_source":       "text_config.global_head_dim",
			"partial_rotary_source": "text_config.rope_parameters.full_attention.partial_rotary_factor",
		},
	}
	data, ne, err := handleRopeFreqsProportional(spec, cfg)
	if err != nil {
		t.Fatalf("handler error: %v", err)
	}
	if ne != [4]int64{256, 1, 1, 1} {
		t.Errorf("ne = %v, want [256 1 1 1]", ne)
	}
	if len(data) != 256 {
		t.Fatalf("len(data) = %d, want 256", len(data))
	}
	for i := range 64 {
		if data[i] != 1.0 {
			t.Errorf("data[%d] = %v, want 1.0", i, data[i])
		}
	}
	for i := 64; i < 256; i++ {
		if data[i] != 1e30 {
			t.Errorf("data[%d] = %v, want 1e30", i, data[i])
		}
	}
}

func TestRopeFreqsProportional_MissingHeadDim(t *testing.T) {
	cfg := configJSONFromString(t, `{}`)
	spec := DerivedTensorSpec{
		Op: "rope_freqs_proportional",
		Params: map[string]any{
			"head_dim_source":       "x",
			"partial_rotary_source": "y",
		},
	}
	if _, _, err := handleRopeFreqsProportional(spec, cfg); err == nil {
		t.Fatal("expected error for missing head_dim_source key")
	}
}

func TestRopeFreqsProportional_OddHeadDim(t *testing.T) {
	cfg := configJSONFromString(t, `{"hd": 7, "pf": 0.5}`)
	spec := DerivedTensorSpec{
		Op:     "rope_freqs_proportional",
		Params: map[string]any{"head_dim_source": "hd", "partial_rotary_source": "pf"},
	}
	if _, _, err := handleRopeFreqsProportional(spec, cfg); err == nil {
		t.Fatal("expected error for odd head_dim")
	}
}

func TestCopyParam_Scalar(t *testing.T) {
	cfg := configJSONFromString(t, `{"text_config": {"global_head_dim": 512}}`)
	spec := DerivedMetadataSpec{
		Target: "gemma4.rope.dimension_count",
		Op:     "copy_param",
		Params: map[string]any{"source": "text_config.global_head_dim"},
	}
	got, err := handleCopyParam(spec, cfg)
	if err != nil {
		t.Fatalf("handler error: %v", err)
	}
	// Value passes through as the JSON-decoded type (json.Number with UseNumber()).
	n, ok := toUint32(got)
	if !ok {
		t.Fatalf("returned value %v (%T) doesn't pass toUint32", got, got)
	}
	if n != 512 {
		t.Errorf("got %d, want 512", n)
	}
}

func TestCopyParam_MissingSource(t *testing.T) {
	cfg := configJSONFromString(t, `{}`)
	spec := DerivedMetadataSpec{
		Op:     "copy_param",
		Params: map[string]any{"source": "missing.key"},
	}
	if _, err := handleCopyParam(spec, cfg); err == nil {
		t.Fatal("expected error for missing source")
	}
}

func TestRegistryLookup(t *testing.T) {
	if _, err := resolveDerivedMetadataOp("string_array_eq"); err != nil {
		t.Errorf("string_array_eq not registered: %v", err)
	}
	if _, err := resolveDerivedMetadataOp("copy_param"); err != nil {
		t.Errorf("copy_param not registered: %v", err)
	}
	if _, err := resolveDerivedMetadataOp("no_such_op"); err == nil {
		t.Error("expected error for unknown metadata op")
	}
	if _, err := resolveDerivedTensorOp("rope_freqs_proportional"); err != nil {
		t.Errorf("rope_freqs_proportional not registered: %v", err)
	}
	if _, err := resolveDerivedTensorOp("no_such_op"); err == nil {
		t.Error("expected error for unknown tensor op")
	}
}
