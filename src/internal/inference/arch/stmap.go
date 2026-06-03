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
	// DerivedMetadata lists GGUF metadata keys whose values are computed at
	// load time from one or more config.json fields. Used when the llama.cpp
	// converter emits a GGUF metadata key via a non-trivial computation that
	// has no 1:1 config.json mapping (e.g. Gemma 4's sliding_window_pattern
	// is computed from text_config.layer_types). Each entry names an op
	// handler from the derivedMetadataOps registry.
	DerivedMetadata []DerivedMetadataSpec
	// DerivedTensors lists GGUF tensor names whose data is synthesized at
	// load time (no source tensor in safetensors). Used when the converter
	// generates extra tensors procedurally (e.g. Gemma 4's rope_freqs.weight
	// is computed from rope params, not stored in the model file). Each
	// entry names an op handler from the derivedTensorOps registry.
	DerivedTensors []DerivedTensorSpec

	// Vision and Projector describe the second tower for multimodal models.
	// Nil for unimodal text models. The vision block mirrors the top-level
	// shape (per-layer prefix + tensor map, global tensors, params); the
	// projector block carries only globals + params since it has no per-
	// layer structure. See ARCHITECTURE.md "Vision / Multimodal Subsystem →
	// Construction Across Two Formats" for the schema rationale and the
	// Gemma 4 worked example.
	Vision    *STMapVision
	Projector *STMapProjector
}

// STMapVision is the [vision] block of an .arch.stmap.toml file.
// It mirrors the top-level layout for the vision tower: HF↔GGUF tensor
// name maps + a params map sourcing from `vision_config.*` in config.json.
type STMapVision struct {
	// LayerPrefixHF is the HF per-layer prefix template (e.g.
	// "model.vision_tower.encoder.layers.{N}.").
	LayerPrefixHF string
	// LayerPrefixGGUF is the GGUF per-layer prefix template
	// (e.g. "v.blk.@{layer_idx}.").
	LayerPrefixGGUF string
	// Tensors maps our per-layer short names to HF short names within the
	// vision tower (e.g. "attn_q.weight" → "self_attn.q_proj.linear.weight").
	Tensors map[string]string
	// GlobalTensors maps vision-tower global short names to HF full names
	// (e.g. "v.patch_embd.weight" → "model.vision_tower.patch_embedder.input_proj.weight").
	GlobalTensors map[string]string
	// Params maps HF config keys (typically under `vision_config.*`) to
	// GGUF-equivalent keys under the `vision.*` namespace.
	Params map[string]string
}

// STMapProjector is the [projector] block of an .arch.stmap.toml file.
// The projector has no per-layer structure — it's a small global graph
// (typically one linear matmul) that bridges the vision tower's output
// embedding space to the decoder's token-embedding space.
type STMapProjector struct {
	GlobalTensors map[string]string
	Params        map[string]string
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

// DerivedMetadataSpec is a single [[derived_metadata]] entry. It declares a
// GGUF metadata key whose value is computed at safetensors-load time from
// other config.json fields (not a 1:1 mapping or a literal). Op names a
// handler in the derivedMetadataOps registry; Params carries op-specific
// arguments (typically including a `source` config.json key path).
type DerivedMetadataSpec struct {
	Target string
	Op     string
	Params map[string]any
}

// DerivedTensorSpec is a single [[derived_tensors]] entry. It declares a
// GGUF tensor name whose data is synthesized at load time — no source
// tensor exists in safetensors. The llama.cpp converter generates such
// tensors via generate_extra_tensors (e.g. Gemma 4's rope_freqs.weight).
// Op names a handler in the derivedTensorOps registry; Params carries
// op-specific arguments (typically source param key paths).
type DerivedTensorSpec struct {
	Target string
	Op     string
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
	Tensors         map[string]any   `toml:"tensors"`
	GGUFMeta        map[string]any   `toml:"gguf_metadata"` // literal GGUF metadata key/value (no config.json equivalent)
	Transforms      []map[string]any `toml:"transforms"`
	DerivedMetadata []map[string]any `toml:"derived_metadata"`
	DerivedTensors  []map[string]any `toml:"derived_tensors"`

	// Vision and Projector are the multimodal extension blocks. See
	// ARCHITECTURE.md "Vision / Multimodal Subsystem → Construction Across
	// Two Formats" for the schema. Both nil for unimodal models.
	Vision    *visionRaw    `toml:"vision"`
	Projector *projectorRaw `toml:"projector"`
}

// visionRaw mirrors the top-level layout for the vision tower's stmap
// block. Sub-tables ([vision.params], [vision.tensors], [vision.tensors.global],
// [vision.layer_prefix]) use map[string]any so we can dispatch the same
// way the top-level path does.
type visionRaw struct {
	Params      map[string]any `toml:"params"`
	LayerPrefix struct {
		HF   string `toml:"hf"`
		GGUF string `toml:"gguf"`
	} `toml:"layer_prefix"`
	Tensors map[string]any `toml:"tensors"`
}

// projectorRaw carries the [projector] block. No per-layer structure —
// just globals and params.
type projectorRaw struct {
	Params  map[string]any `toml:"params"`
	Tensors map[string]any `toml:"tensors"`
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

	// Extract string-valued params; sub-tables like [params.defaults]
	// handled separately.
	params, paramDefaults := extractParams(raw.Params)

	tensors, globalTensors, err := extractTensors(archName, "tensors", raw.Tensors)
	if err != nil {
		return nil, err
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

	derivedMeta, err := parseDerivedMetadataSpecs(archName, raw.DerivedMetadata)
	if err != nil {
		return nil, err
	}

	derivedTensors, err := parseDerivedTensorSpecs(archName, raw.DerivedTensors)
	if err != nil {
		return nil, err
	}

	vision, err := parseSTMapVision(archName, raw.Vision)
	if err != nil {
		return nil, err
	}
	projector, err := parseSTMapProjector(archName, raw.Projector)
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
		DerivedMetadata: derivedMeta,
		DerivedTensors:  derivedTensors,
		Vision:          vision,
		Projector:       projector,
	}, nil
}

// extractParams pulls scalar string entries out of a raw [params] (or
// [vision.params]) table and any nested "defaults" sub-table. Returns
// (params, defaults). Non-string, non-table values are silently
// ignored — strict-decoding is the top-level TOML decoder's job; this
// helper is only concerned with the supported shapes.
func extractParams(raw map[string]any) (params, defaults map[string]string) {
	params = make(map[string]string)
	defaults = make(map[string]string)
	for k, v := range raw {
		if s, ok := v.(string); ok {
			params[k] = s
			continue
		}
		if m, ok := v.(map[string]any); ok && k == "defaults" {
			for dk, dv := range m {
				if s, ok := dv.(string); ok {
					defaults[dk] = s
				}
			}
		}
	}
	return params, defaults
}

// extractTensors splits a raw [tensors] (or [vision.tensors]) table into
// the per-layer short-name map and the [tensors.global] full-name map.
// archName and section are used only for error messages.
func extractTensors(archName, section string, raw map[string]any) (perLayer, globals map[string]string, err error) {
	perLayer = make(map[string]string)
	globals = make(map[string]string)
	for k, v := range raw {
		if k == "global" {
			m, ok := v.(map[string]any)
			if !ok {
				return nil, nil, fmt.Errorf("stmap def %q: %s.global must be a table", archName, section)
			}
			for gk, gv := range m {
				s, ok := gv.(string)
				if !ok {
					return nil, nil, fmt.Errorf("stmap def %q: %s.global.%s must be a string", archName, section, gk)
				}
				globals[gk] = s
			}
			continue
		}
		s, ok := v.(string)
		if !ok {
			return nil, nil, fmt.Errorf("stmap def %q: %s.%s must be a string", archName, section, k)
		}
		perLayer[k] = s
	}
	return perLayer, globals, nil
}

// parseSTMapVision converts the raw [vision] block into STMapVision.
// Returns (nil, nil) when the block is absent (unimodal models).
func parseSTMapVision(archName string, raw *visionRaw) (*STMapVision, error) {
	if raw == nil {
		return nil, nil
	}
	params, _ := extractParams(raw.Params) // [vision.params.defaults] not yet supported; intentional
	tensors, globals, err := extractTensors(archName, "vision.tensors", raw.Tensors)
	if err != nil {
		return nil, err
	}
	if raw.LayerPrefix.HF == "" && len(tensors) > 0 {
		return nil, fmt.Errorf("stmap def %q: vision.layer_prefix.hf is required when vision.tensors has per-layer entries", archName)
	}
	ggufPrefix := raw.LayerPrefix.GGUF
	if ggufPrefix == "" {
		ggufPrefix = "v." + BuiltinLayerPrefix // default vision prefix: "v.blk.@{layer_idx}."
	}
	return &STMapVision{
		LayerPrefixHF:   raw.LayerPrefix.HF,
		LayerPrefixGGUF: ggufPrefix,
		Tensors:         tensors,
		GlobalTensors:   globals,
		Params:          params,
	}, nil
}

// parseSTMapProjector converts the raw [projector] block into
// STMapProjector. Returns (nil, nil) when the block is absent.
func parseSTMapProjector(archName string, raw *projectorRaw) (*STMapProjector, error) {
	if raw == nil {
		return nil, nil
	}
	params, _ := extractParams(raw.Params)
	_, globals, err := extractTensors(archName, "projector.tensors", raw.Tensors)
	if err != nil {
		return nil, err
	}
	return &STMapProjector{
		GlobalTensors: globals,
		Params:        params,
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

// parseDerivedMetadataSpecs converts the raw TOML [[derived_metadata]]
// array into DerivedMetadataSpec values. Each entry must have `op` and
// `target` (non-empty strings). All other keys are collected into Params.
func parseDerivedMetadataSpecs(archName string, raw []map[string]any) ([]DerivedMetadataSpec, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	specs := make([]DerivedMetadataSpec, 0, len(raw))
	for i, entry := range raw {
		target, op, params, err := parseDerivedEntry("derived_metadata", archName, i, entry)
		if err != nil {
			return nil, err
		}
		specs = append(specs, DerivedMetadataSpec{Target: target, Op: op, Params: params})
	}
	return specs, nil
}

// parseDerivedTensorSpecs converts the raw TOML [[derived_tensors]] array
// into DerivedTensorSpec values. Same shape as derived_metadata; only the
// op-handler registry differs at consumption time.
func parseDerivedTensorSpecs(archName string, raw []map[string]any) ([]DerivedTensorSpec, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	specs := make([]DerivedTensorSpec, 0, len(raw))
	for i, entry := range raw {
		target, op, params, err := parseDerivedEntry("derived_tensors", archName, i, entry)
		if err != nil {
			return nil, err
		}
		specs = append(specs, DerivedTensorSpec{Target: target, Op: op, Params: params})
	}
	return specs, nil
}

// parseDerivedEntry validates and extracts the common (target, op, params)
// triple from a derived_metadata or derived_tensors TOML entry. Used by both
// parseDerivedMetadataSpecs and parseDerivedTensorSpecs.
func parseDerivedEntry(blockName, archName string, idx int, entry map[string]any) (target, op string, params map[string]any, err error) {
	opAny, ok := entry["op"]
	if !ok {
		return "", "", nil, fmt.Errorf("stmap %q: %s[%d] missing 'op'", archName, blockName, idx)
	}
	op, ok = opAny.(string)
	if !ok || op == "" {
		return "", "", nil, fmt.Errorf("stmap %q: %s[%d].op must be a non-empty string", archName, blockName, idx)
	}
	targetAny, ok := entry["target"]
	if !ok {
		return "", "", nil, fmt.Errorf("stmap %q: %s[%d] missing 'target'", archName, blockName, idx)
	}
	target, ok = targetAny.(string)
	if !ok || target == "" {
		return "", "", nil, fmt.Errorf("stmap %q: %s[%d].target must be a non-empty string", archName, blockName, idx)
	}
	params = make(map[string]any, len(entry))
	for k, v := range entry {
		if k == "op" || k == "target" {
			continue
		}
		params[k] = v
	}
	return target, op, params, nil
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
