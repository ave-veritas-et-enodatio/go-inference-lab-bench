package arch

// Module identifies a named weight module at architectural block boundaries.
//
// Weight names are stored in compact form:
//   - WeightContext holds the shared GGUF prefix (e.g. "blk.5").
//   - Weights holds names relative to WeightContext that take the implicit ".weight" suffix
//     (e.g. "attn_norm" → "blk.5.attn_norm.weight").
//   - Params holds names relative to WeightContext that are used verbatim
//     (e.g. "ssm_dt.bias" → "blk.5.ssm_dt.bias", "ssm_a" → "blk.5.ssm_a").
//   - If WeightContext is empty the context prefix step is skipped.
type Module struct {
	ID            int      `toml:"id"`
	Name          string   `toml:"name"`
	BlockName     string   `toml:"block_name,omitempty"`     // routing-resolved block type; empty for FFN/global
	WeightContext string   `toml:"weight_context,omitempty"`
	Weights       []string `toml:"weights,omitempty"`        // short names, get ".weight" appended on reconstruction
	Params        []string `toml:"params,omitempty"`         // short names used verbatim (bias, bare scalars, etc.)
}

// FullWeightNames reconstructs the complete GGUF tensor names for this module.
func (m *Module) FullWeightNames() []string {
	prefix := ""
	if m.WeightContext != "" {
		prefix = m.WeightContext + "."
	}
	names := make([]string, 0, len(m.Weights)+len(m.Params))
	for _, w := range m.Weights {
		names = append(names, prefix+w+".weight")
	}
	for _, p := range m.Params {
		names = append(names, prefix+p)
	}
	return names
}

// ModuleMap is the static structural map of a model's weight modules.
// Modules is the full structural list built by BuildModuleMap.
type ModuleMap struct {
	Modules []Module `toml:"modules"`
}
