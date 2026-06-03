package arch

import "testing"

func TestShardNamePattern(t *testing.T) {
	match := []string{
		"model-00001-of-00004.safetensors",             // standard HF
		"model.safetensors-00001-of-00004.safetensors", // model.safetensors base variant
		"model-1-of-2.safetensors",                      // unpadded width
		"model.safetensors-12-of-34.safetensors",
	}
	noMatch := []string{
		"model.safetensors",            // single-file (handled by exact-name check)
		"model-00001.safetensors",      // no -of-M
		"model-00001-of.safetensors",   // malformed
		"foo-00001-of-00004.safetensors", // wrong base
		"model.bin-00001-of-00004.safetensors",
		"model-00001-of-00004.bin",
	}
	for _, n := range match {
		if shardNamePattern.FindStringSubmatch(n) == nil {
			t.Errorf("expected %q to match shardNamePattern", n)
		}
	}
	for _, n := range noMatch {
		if shardNamePattern.FindStringSubmatch(n) != nil {
			t.Errorf("expected %q NOT to match shardNamePattern", n)
		}
	}
}
