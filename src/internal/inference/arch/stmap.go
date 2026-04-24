package arch

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/BurntSushi/toml"
	log "inference-lab-bench/internal/log"
)

const extArchSTMapToml = ".arch.stmap.toml"

// ArchSTMap is the safetensors → GGUF mapping for an architecture: it
// translates HuggingFace tensor/param names to our canonical (GGUF) names,
// and declares any per-tensor numeric/structural transforms needed so that
// safetensors-loaded weights end up byte-for-byte equivalent to what a
// llama.cpp-converted GGUF would contain.
type ArchSTMap struct {
	// HFClass is the config.json "architectures"[0] value that maps to this architecture.
	HFClass string
	// Params maps HF param names to GGUF-equivalent keys (e.g., "num_hidden_layers" → "llama.block_count").
	Params map[string]string
	// LayerPrefixHF is the HF layer prefix with {N} substitution (e.g., "model.layers.{N}.").
	LayerPrefixHF string
	// LayerPrefixGGUF is the GGUF layer prefix with {N} substitution.
	// Defaults to "blk.@{layer_idx}." when not specified.
	LayerPrefixGGUF string
	// Tensors maps our per-layer short names to HF short names.
	Tensors map[string]string
	// GlobalTensors maps our global short names to HF full names (no prefix expansion).
	GlobalTensors map[string]string
	// ParamDefaults maps param names to param references for default values.
	ParamDefaults map[string]string
	// Metadata provides literal GGUF metadata key → value pairs for keys that
	// only exist in the GGUF metadata section (injected by llama.cpp conversion)
	// and have no config.json equivalent. Only used by safetensors loading.
	Metadata map[string]any
	// Transforms lists tensor post-read transforms (e.g. norm+1, -exp on A_log,
	// SSM V-head reordering). Applied to matching tensors at load time in
	// declared order. Only used by safetensors loading.
	Transforms []TransformSpec
}

// TransformSpec is a single [[transforms]] entry from a stmap TOML.
// Op is the transform kind; Apply lists GGUF short tensor names to match
// (matched against the last path component of each tensor name). Params
// holds op-specific arguments as raw TOML values.
type TransformSpec struct {
	Op     string
	Apply  []string
	Params map[string]any
}

// stmapRaw matches the TOML structure for unmarshaling.
// Uses map[string]any for [params] and [tensors] because they can contain
// sub-tables (e.g. [params.defaults], [tensors.global]).
type stmapRaw struct {
	Architecture struct {
		HFClass string `toml:"hf_class"`
	} `toml:"architecture"`
	Params      map[string]any `toml:"params"`
	LayerPrefix struct {
		HF   string `toml:"hf"`
		GGUF string `toml:"gguf"`
	} `toml:"layer_prefix"`
	Tensors    map[string]any   `toml:"tensors"`
	GGUFMeta   map[string]any   `toml:"gguf_metadata"` // literal GGUF metadata key/value (no config.json equivalent)
	Transforms []map[string]any `toml:"transforms"`
}

// LoadArchSTMap loads <archDir>/<archName>.arch.stmap.toml.
// Returns (nil, nil) if the file does not exist — not all architectures need stmaps.
// Returns an error on TOML parse failure.
func LoadArchSTMap(archDir, archName string) (*ArchSTMap, error) {
	filename := filepath.Join(archDir, archName+extArchSTMapToml)
	data, err := os.ReadFile(filename)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("reading stmap def %q: %w", archName, err)
	}

	var raw stmapRaw
	if _, err := toml.Decode(string(data), &raw); err != nil {
		return nil, fmt.Errorf("parsing stmap def %q: %w", archName, err)
	}

	if raw.Architecture.HFClass == "" {
		return nil, fmt.Errorf("stmap def %q: architecture.hf_class is required", archName)
	}

	// Extract string-valued params; sub-tables like [param.defaults] handled separately.
	params := make(map[string]string)
	paramDefaults := make(map[string]string)
	for k, v := range raw.Params {
		if s, ok := v.(string); ok {
			params[k] = s
		} else if m, ok := v.(map[string]any); ok {
			// Sub-table (e.g. "defaults") — extract string values.
			for dk, dv := range m {
				if s, ok := dv.(string); ok {
					paramDefaults[dk] = s
				}
			}
		}
	}

	tensors := make(map[string]string)
	globalTensors := make(map[string]string)

	for k, v := range raw.Tensors {
		if k == "global" {
			if m, ok := v.(map[string]any); ok {
				for gk, gv := range m {
					if s, ok := gv.(string); ok {
						globalTensors[gk] = s
					}
				}
			}
			continue
		}
		s, ok := v.(string)
		if !ok {
			return nil, fmt.Errorf("stmap def %q: tensors.%s must be a string", archName, k)
		}
		tensors[k] = s
	}

	// Use default GGUF prefix if not overridden (matches all arch TOML prefix conventions).
	ggufPrefix := raw.LayerPrefix.GGUF
	if ggufPrefix == "" {
		ggufPrefix = BuiltinLayerPrefix
	}

	transforms, err := parseTransformSpecs(archName, raw.Transforms)
	if err != nil {
		return nil, err
	}

	return &ArchSTMap{
		HFClass:         raw.Architecture.HFClass,
		Params:          params,
		LayerPrefixHF:   raw.LayerPrefix.HF,
		LayerPrefixGGUF: ggufPrefix,
		Tensors:         tensors,
		GlobalTensors:   globalTensors,
		ParamDefaults:   paramDefaults,
		Metadata:        raw.GGUFMeta,
		Transforms:      transforms,
	}, nil
}

// parseTransformSpecs converts the raw TOML [[transforms]] array into
// TransformSpec values. Each entry must have `op` (string) and `apply`
// (array of strings). All other keys are collected into Params as-is.
func parseTransformSpecs(archName string, raw []map[string]any) ([]TransformSpec, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	specs := make([]TransformSpec, 0, len(raw))
	for i, entry := range raw {
		opAny, ok := entry["op"]
		if !ok {
			return nil, fmt.Errorf("stmap %q: transforms[%d] missing 'op'", archName, i)
		}
		op, ok := opAny.(string)
		if !ok || op == "" {
			return nil, fmt.Errorf("stmap %q: transforms[%d].op must be a non-empty string", archName, i)
		}
		applyAny, ok := entry["apply"]
		if !ok {
			return nil, fmt.Errorf("stmap %q: transforms[%d] missing 'apply'", archName, i)
		}
		applyArr, ok := applyAny.([]any)
		if !ok || len(applyArr) == 0 {
			return nil, fmt.Errorf("stmap %q: transforms[%d].apply must be a non-empty array of strings", archName, i)
		}
		apply := make([]string, 0, len(applyArr))
		for j, v := range applyArr {
			s, ok := v.(string)
			if !ok {
				return nil, fmt.Errorf("stmap %q: transforms[%d].apply[%d] must be a string", archName, i, j)
			}
			apply = append(apply, s)
		}
		params := make(map[string]any, len(entry))
		for k, v := range entry {
			if k == "op" || k == "apply" {
				continue
			}
			params[k] = v
		}
		specs = append(specs, TransformSpec{Op: op, Apply: apply, Params: params})
	}
	return specs, nil
}

// FindSTMapByHFClass scans all *.arch.stmap.toml files in archDir and returns
// the one with matching hf_class. Returns ("", nil, nil) if none found.
func FindSTMapByHFClass(archDir, hfClass string) (string, *ArchSTMap, error) {
	entries, err := os.ReadDir(archDir)
	if err != nil {
		return "", nil, fmt.Errorf("scanning arch dir for stmaps: %w", err)
	}
	log.Debug("FindSTMapByHFClass: scanning %s for hf_class=%q, found %d entries", archDir, hfClass, len(entries))

	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), extArchSTMapToml) {
			continue
		}

		archName := strings.TrimSuffix(e.Name(), extArchSTMapToml)
		stmap, err := LoadArchSTMap(archDir, archName)
		if err != nil {
			log.Debug("FindSTMapByHFClass: error loading %s: %v", archName, err)
			return "", nil, err
		}
		if stmap != nil {
			log.Debug("FindSTMapByHFClass: %s has hf_class=%q", archName, stmap.HFClass)
		}
		if stmap != nil && stmap.HFClass == hfClass {
			return archName, stmap, nil
		}
	}

	log.Debug("FindSTMapByHFClass: no match found for hf_class=%q", hfClass)
	return "", nil, nil
}
