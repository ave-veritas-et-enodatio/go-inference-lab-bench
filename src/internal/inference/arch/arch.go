package arch

import (
	"fmt"
	goast "go/ast"
	goparser "go/parser"
	"os"
	"path/filepath"
	"strings"

	"github.com/BurntSushi/toml"

	"inference-lab-bench/internal/util"
)

// ArchDef is the top-level parsed architecture definition.
type ArchDef struct {
	Architecture ArchMeta               `toml:"architecture"`
	Params       ParamsDef              `toml:"params"`
	Weights      WeightsDef             `toml:"weights"`
	Layers       LayersDef              `toml:"layers"`
	Blocks       map[string]BlockDef    `toml:"blocks"`
	FFN          FFNDef                 `toml:"ffn"`
	FFNAlt       *FFNDef                `toml:"ffn_alt"` // optional: layers that have these weights use this FFN instead
	Tokens       TokensDef              `toml:"tokens"`
	Example      ExampleDef             `toml:"example"`
}

// ExampleDef provides diagram-only example values for SVG generation.
// Has no effect on inference. Used when no GGUF model is loaded.
type ExampleDef struct {
	NLayers       int `toml:"n_layers"`        // example layer count for layer-pattern strip
	FullAttnEvery int `toml:"full_attn_every"` // interval: layer (i+1) % N == 0 is full/global attention
}

type TokensDef struct {
	ThinkOpen  string   `toml:"think_open"`   // e.g. "<think>"
	ThinkClose string   `toml:"think_close"`  // e.g. "</think>"
	ExtraEOS []string `toml:"extra_eos"`
}

type ArchMeta struct {
	Name            string `toml:"name"`
	TiedEmbeddings  bool   `toml:"tied_embeddings"`
	NonCausal       bool   `toml:"non_causal"`       // bidirectional attention (no causal mask)
	EmbedScale      bool   `toml:"embed_scale"`      // multiply embeddings by sqrt(n_embd)
	Generation      string `toml:"generation"`       // "" (default autoregressive) or "diffusion"
	ShiftLogits     bool   `toml:"shift_logits"`     // diffusion: use position p-1 logits for output position p
	NoFlashAttn     bool   `toml:"no_flash_attn"`    // disable flash attention; falls back to standard MulMat SDPA
}

// ParamsDef holds GGUF key mappings and derived expressions.
// Each entry in the map is either a GGUF key string (e.g. "qwen35.block_count")
// or a special value like "neox" for enum-like params.
type ParamsDef struct {
	// Flat params parsed from TOML key=value pairs (excluding "derived" and "defaults")
	Keys     map[string]string  `toml:"-"`
	Derived  map[string]string  `toml:"derived"`
	Defaults map[string]string  `toml:"defaults"` // fallback: if param resolves to 0, use this param's value instead
}

// UnmarshalTOML implements custom unmarshalling for ParamsDef.
// The params section has flat key=value pairs plus a nested "derived" table.
func (pd *ParamsDef) UnmarshalTOML(data any) error {
	m, ok := data.(map[string]any)
	if !ok {
		return fmt.Errorf("params: expected table, got %T", data)
	}
	pd.Keys = make(map[string]string)
	pd.Derived = make(map[string]string)
	pd.Defaults = make(map[string]string)
	for k, v := range m {
		if k == "derived" || k == "defaults" {
			sub, ok := v.(map[string]any)
			if !ok {
				return fmt.Errorf("params.%s: expected table, got %T", k, v)
			}
			target := pd.Derived
			if k == "defaults" {
				target = pd.Defaults
			}
			for dk, dv := range sub {
				s, ok := dv.(string)
				if !ok {
					return fmt.Errorf("params.%s.%s: expected string, got %T", k, dk, dv)
				}
				target[dk] = s
			}
			continue
		}
		s, ok := v.(string)
		if !ok {
			return fmt.Errorf("params.%s: expected string, got %T", k, v)
		}
		pd.Keys[k] = s
	}
	return nil
}

type WeightsDef struct {
	Global map[string]string `toml:"global"`
}

type LayersDef struct {
	Count         string                  `toml:"count"`
	Prefix        string                  `toml:"prefix"`
	Routing       RoutingDef              `toml:"routing"`
	CommonWeights map[string]string       `toml:"common_weights"`
}

// RoutingDef defines binary per-layer block routing.
// Either Rule (expression) or Pattern (array param indexed by layer) must be set, not both.
type RoutingDef struct {
	Rule    string `toml:"rule"`
	Pattern string `toml:"pattern"`  // array param name — nonzero → if_true, zero → if_false
	IfTrue  string `toml:"if_true"`
	IfFalse string `toml:"if_false"`
}

type BlockDef struct {
	Builder string                 `toml:"builder"`
	Weights map[string]string      `toml:"weights"`
	Config  map[string]any         `toml:"config"`
	Cache   map[string]CacheDef    `toml:"cache"`
}

type CacheDef struct {
	Dims   []string `toml:"dims"`
	Dtype  string   `toml:"dtype"`
	Shared string   `toml:"shared"` // shared group name — layers with same value share one cache tensor
}

type FFNDef struct {
	Builder string                 `toml:"builder"`
	Weights map[string]string      `toml:"weights"`
	Config  map[string]any         `toml:"config"`
}

// archTomlExt is a package-local alias for the canonical extension constant.
var archTomlExt = util.ExtArchToml

// Load reads an architecture definition from the given directory.
// archDir is the directory containing arch.toml definition files.
// name should be the architecture name without extension (e.g. "qwen35").
func Load(archDir, name string) (*ArchDef, error) {
	filename := filepath.Join(archDir, name+archTomlExt)
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("arch def %q not found at %s: %w", name, filename, err)
	}
	def, err := Parse(data)
	if err != nil {
		return nil, err
	}
	if def.Architecture.Name != name {
		return nil, fmt.Errorf("arch name mismatch: file %s%s declares name %q (must match filename)",
			name, archTomlExt, def.Architecture.Name)
	}
	return def, nil
}

// Parse decodes a TOML architecture definition from raw bytes.
func Parse(data []byte) (*ArchDef, error) {
	def := &ArchDef{}
	md, err := toml.Decode(string(data), def)
	if err != nil {
		return nil, fmt.Errorf("parsing arch def: %w", err)
	}
	if undecoded := md.Undecoded(); len(undecoded) > 0 {
		keys := make([]string, len(undecoded))
		for i, k := range undecoded {
			keys[i] = k.String()
		}
		return nil, fmt.Errorf("unknown arch def keys: %s", strings.Join(keys, ", "))
	}
	if def.Architecture.Name == "" {
		return nil, fmt.Errorf("architecture.name is required")
	}
	if errs := Validate(def); len(errs) > 0 {
		lines := ResolveErrorLines(data, errs)
		return nil, fmt.Errorf("validation errors:\n%s", strings.Join(lines, "\n"))
	}
	return def, nil
}

// ValidationError represents a single validation issue with its TOML key path.
type ValidationError struct {
	KeyPath string
	Message string
}

// Validate checks structural and semantic constraints on a parsed ArchDef.
// Returns all errors found (not just the first).
func Validate(def *ArchDef) []ValidationError {
	var errs []ValidationError
	add := func(keyPath, msg string) {
		errs = append(errs, ValidationError{keyPath, msg})
	}

	// --- Architecture meta checks ---

	if def.Architecture.Generation != "" && def.Architecture.Generation != "diffusion" {
		add("architecture.generation", fmt.Sprintf("unknown generation strategy %q (valid: \"diffusion\")", def.Architecture.Generation))
	}
	if def.Architecture.Generation == "diffusion" && !def.Architecture.NonCausal {
		add("architecture.generation", "generation=diffusion requires non_causal=true")
	}

	// --- Structural checks ---

	if def.Layers.Count == "" {
		add("layers.count", "required")
	}
	if !strings.Contains(def.Layers.Prefix, "@{layer_idx}") {
		add("layers.prefix", "must contain @{layer_idx}")
	}
	r := def.Layers.Routing
	if r.Rule == "" && r.Pattern == "" {
		add("layers.routing", "either rule or pattern is required")
	}
	if r.Rule != "" && r.Pattern != "" {
		add("layers.routing", "rule and pattern are mutually exclusive")
	}
	if r.IfTrue == "" {
		add("layers.routing.if_true", "required")
	}
	if r.IfFalse == "" {
		add("layers.routing.if_false", "required")
	}
	if r.IfTrue != "" {
		if _, ok := def.Blocks[r.IfTrue]; !ok {
			add("layers.routing.if_true", fmt.Sprintf("no such block %q", r.IfTrue))
		}
	}
	if r.IfFalse != "" {
		if _, ok := def.Blocks[r.IfFalse]; !ok {
			add("layers.routing.if_false", fmt.Sprintf("no such block %q", r.IfFalse))
		}
	}

	// --- Block builder contract checks ---

	for name, blk := range def.Blocks {
		bb, ok := GetBlockBuilder(blk.Builder)
		if !ok {
			add(fmt.Sprintf("blocks.%s.builder", name), fmt.Sprintf("unknown builder %q", blk.Builder))
			continue
		}
		if len(blk.Weights) == 0 {
			add(fmt.Sprintf("blocks.%s.weights", name), "must not be empty")
		}
		validateContract(bb.Contract(), blk.Weights, blk.Config, fmt.Sprintf("blocks.%s", name), &errs)

		for cn, cd := range blk.Cache {
			if len(cd.Dims) == 0 {
				add(fmt.Sprintf("blocks.%s.cache.%s", name, cn), "dims must not be empty")
			}
			if !validCacheDtype(cd.Dtype) {
				add(fmt.Sprintf("blocks.%s.cache.%s", name, cn), fmt.Sprintf("invalid dtype %q", cd.Dtype))
			}
		}
	}

	// --- FFN builder contract checks ---

	fb, ok := GetFFNBuilder(def.FFN.Builder)
	if !ok {
		add("ffn.builder", fmt.Sprintf("unknown builder %q", def.FFN.Builder))
	} else {
		if len(def.FFN.Weights) == 0 {
			add("ffn.weights", "must not be empty")
		}
		validateContract(fb.Contract(), def.FFN.Weights, def.FFN.Config, "ffn", &errs)
	}

	// --- FFN alt builder contract checks ---

	if def.FFNAlt != nil {
		ffnAltB, ok := GetFFNBuilder(def.FFNAlt.Builder)
		if !ok {
			add("ffn_alt.builder", fmt.Sprintf("unknown builder %q", def.FFNAlt.Builder))
		} else {
			if len(def.FFNAlt.Weights) == 0 {
				add("ffn_alt.weights", "must not be empty")
			}
			validateContract(ffnAltB.Contract(), def.FFNAlt.Weights, def.FFNAlt.Config, "ffn_alt", &errs)
		}
	}

	// --- norm_w / norm_w_param mutual exclusion ---

	checkFFNNormW := func(config map[string]any, prefix string) {
		_, hasStatic := config["norm_w"]
		_, hasParam := config["norm_w_param"]
		if hasStatic && hasParam {
			add(prefix+".config", "norm_w and norm_w_param are mutually exclusive")
		}
	}
	if def.FFN.Config != nil {
		checkFFNNormW(def.FFN.Config, "ffn")
	}
	if def.FFNAlt != nil && def.FFNAlt.Config != nil {
		checkFFNNormW(def.FFNAlt.Config, "ffn_alt")
	}

	// --- Required global weights ---

	if _, ok := def.Weights.Global["token_embd"]; !ok {
		add("weights.global", "missing required key \"token_embd\"")
	}
	if _, ok := def.Weights.Global["output_norm"]; !ok {
		add("weights.global", "missing required key \"output_norm\"")
	}

	// --- Required common weights ---

	if _, ok := def.Layers.CommonWeights["attn_norm"]; !ok {
		add("layers.common_weights", "missing required key \"attn_norm\"")
	}
	if _, ok := def.Layers.CommonWeights["ffn_norm"]; !ok {
		add("layers.common_weights", "missing required key \"ffn_norm\"")
	}

	// --- Cross-reference checks ---

	declaredParams := collectDeclaredParams(def)

	// norm_w_param must reference a declared param.
	if def.FFN.Config != nil {
		if ref, ok := def.FFN.Config["norm_w_param"].(string); ok && ref != "" {
			if !declaredParams[ref] {
				add("ffn.config.norm_w_param", fmt.Sprintf("references undeclared param %q", ref))
			}
		}
	}
	if def.FFNAlt != nil && def.FFNAlt.Config != nil {
		if ref, ok := def.FFNAlt.Config["norm_w_param"].(string); ok && ref != "" {
			if !declaredParams[ref] {
				add("ffn_alt.config.norm_w_param", fmt.Sprintf("references undeclared param %q", ref))
			}
		}
	}

	// Routing rule: validate @{builtin} refs, ${param} refs, and expression syntax
	if r.Rule != "" {
		for _, ref := range extractRefs(r.Rule, '@') {
			if !routingBuiltins[ref] {
				add("layers.routing.rule", fmt.Sprintf("unknown builtin @{%s}", ref))
			}
		}
		for _, ref := range extractRefs(r.Rule, '$') {
			if !declaredParams[ref] {
				add("layers.routing.rule", fmt.Sprintf("references undeclared param ${%s}", ref))
			}
		}
		for _, exprErr := range ValidateRoutingExpr(r.Rule, declaredParams) {
			add("layers.routing.rule", exprErr)
		}
	}
	if r.Pattern != "" {
		if !declaredParams[r.Pattern] {
			add("layers.routing.pattern", fmt.Sprintf("references undeclared param %q", r.Pattern))
		}
	}

	// layers.count reference
	if def.Layers.Count != "" {
		if _, err := fmt.Sscanf(def.Layers.Count, "%d", new(int)); err != nil {
			if !declaredParams[def.Layers.Count] {
				add("layers.count", fmt.Sprintf("references undeclared param %q", def.Layers.Count))
			}
		}
	}

	// Cache dim expressions
	for name, blk := range def.Blocks {
		for cn, cd := range blk.Cache {
			for _, dimExpr := range cd.Dims {
				for _, ident := range extractIdentifiers(dimExpr) {
					if ident == "max_seq_len" {
						continue
					}
					if !declaredParams[ident] {
						add(fmt.Sprintf("blocks.%s.cache.%s", name, cn),
							fmt.Sprintf("dim expression references undeclared param %q", ident))
					}
				}
			}
		}
	}

	// Derived expression references
	for name, expr := range def.Params.Derived {
		if strings.Contains(expr, ".ne[") {
			continue // tensor dim lookup, not a param reference
		}
		for _, ident := range extractIdentifiers(expr) {
			if !declaredParams[ident] {
				add(fmt.Sprintf("params.derived.%s", name),
					fmt.Sprintf("references undeclared param %q", ident))
			}
		}
	}

	// SharedKV producer check: every non-KV block referencing a shared_kv_group
	// must have at least one KV block (has attn_k) that produces for that group.
	kvGroups := make(map[string]bool)
	for _, blk := range def.Blocks {
		if _, hasK := blk.Weights["attn_k"]; hasK {
			if group, _ := blk.Config["shared_kv_group"].(string); group != "" {
				kvGroups[group] = true
			}
		}
	}
	for name, blk := range def.Blocks {
		if _, hasK := blk.Weights["attn_k"]; !hasK {
			if group, _ := blk.Config["shared_kv_group"].(string); group != "" {
				if !kvGroups[group] {
					add(fmt.Sprintf("blocks.%s.config.shared_kv_group", name),
						fmt.Sprintf("no KV-producing block (with attn_k) found for group %q", group))
				}
			}
		}
	}

	return errs
}

// validateContract checks a builder's contract against the TOML-defined weights and config.
func validateContract(c BuilderContract, weights map[string]string, config map[string]any, prefix string, errs *[]ValidationError) {
	add := func(keyPath, msg string) {
		*errs = append(*errs, ValidationError{keyPath, msg})
	}

	// Check required weights are present
	for _, req := range c.RequiredWeights {
		if _, ok := weights[req]; !ok {
			add(fmt.Sprintf("%s.weights", prefix), fmt.Sprintf("missing required key %q", req))
		}
	}

	// Check for unknown weight keys
	known := make(map[string]bool)
	for _, k := range c.RequiredWeights {
		known[k] = true
	}
	for _, k := range c.OptionalWeights {
		known[k] = true
	}
	for k := range weights {
		if !known[k] {
			add(fmt.Sprintf("%s.weights.%s", prefix, k), fmt.Sprintf("unknown key (valid: %s)",
				strings.Join(append(c.RequiredWeights, c.OptionalWeights...), ", ")))
		}
	}

	// Check config keys and values
	if c.ConfigSchema != nil {
		for k, v := range config {
			validValues, schemaHasKey := c.ConfigSchema[k]
			if !schemaHasKey {
				validKeys := make([]string, 0, len(c.ConfigSchema))
				for sk := range c.ConfigSchema {
					validKeys = append(validKeys, sk)
				}
				add(fmt.Sprintf("%s.config.%s", prefix, k),
					fmt.Sprintf("unknown config key (valid: %s)", strings.Join(validKeys, ", ")))
				continue
			}
			if validValues != nil {
				strVal := fmt.Sprintf("%v", v)
				found := false
				for _, vv := range validValues {
					if strVal == vv {
						found = true
						break
					}
				}
				if !found {
					add(fmt.Sprintf("%s.config.%s", prefix, k),
						fmt.Sprintf("invalid value %q (valid: %s)", strVal, strings.Join(validValues, ", ")))
				}
			}
		}
	}
}

// collectDeclaredParams returns the set of all param names declared in the def.
func collectDeclaredParams(def *ArchDef) map[string]bool {
	m := make(map[string]bool)
	for name := range def.Params.Keys {
		m[name] = true
	}
	for name := range def.Params.Derived {
		m[name] = true
	}
	return m
}

// extractIdentifiers parses a Go-like expression and returns all identifier names.
func extractIdentifiers(expr string) []string {
	node, err := parseExprSafe(expr)
	if err != nil {
		return nil
	}
	var idents []string
	collectIdents(node, &idents)
	return idents
}

func collectIdents(node any, idents *[]string) {
	switch n := node.(type) {
	case *goast.Ident:
		*idents = append(*idents, n.Name)
	case *goast.BinaryExpr:
		collectIdents(n.X, idents)
		collectIdents(n.Y, idents)
	case *goast.ParenExpr:
		collectIdents(n.X, idents)
	}
}

func parseExprSafe(expr string) (goast.Expr, error) {
	return goparser.ParseExpr(expr)
}

// extractRefs returns all names referenced via sigil{name} syntax in an expression.
func extractRefs(expr string, sigil byte) []string {
	var refs []string
	for i := 0; i < len(expr); i++ {
		if i+1 < len(expr) && expr[i] == sigil && expr[i+1] == '{' {
			end := strings.IndexByte(expr[i+2:], '}')
			if end >= 0 {
				refs = append(refs, expr[i+2:i+2+end])
				i += 2 + end
			}
		}
	}
	return refs
}

func validCacheDtype(s string) bool {
	switch s {
	case "f32", "f16", "i32", "i16", "i8":
		return true
	}
	return false
}

// ListArchitectures returns the names of all architecture definitions in the given directory.
func ListArchitectures(archDir string) ([]string, error) {
	entries, err := os.ReadDir(archDir)
	if err != nil {
		return nil, err
	}
	var names []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), archTomlExt) {
			names = append(names, strings.TrimSuffix(e.Name(), archTomlExt))
		}
	}
	return names, nil
}
