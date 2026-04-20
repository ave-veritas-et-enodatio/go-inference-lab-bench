package arch

import (
	"bytes"
	"math"
	"os"
	"sort"

	"github.com/BurntSushi/toml"

	"inference-lab-bench/internal/util"
)

// ModuleCulled is the sentinel value indicating a module is fully culled.
const ModuleCulled = math.MinInt32

// Module identifies a named weight module at architectural block boundaries.
//
// Weight names are stored in compact form:
//   - WeightContext holds the shared GGUF prefix (e.g. "blk.5").
//   - Weights holds names relative to WeightContext that take the implicit ".weight" suffix
//     (e.g. "attn_norm" -> "blk.5.attn_norm.weight").
//   - Params holds names relative to WeightContext that are used verbatim
//     (e.g. "ssm_dt.bias" -> "blk.5.ssm_dt.bias", "ssm_a" -> "blk.5.ssm_a").
//   - If WeightContext is empty the context prefix step is skipped.
type Module struct {
	ID              int      `toml:"id"`
	Name            string   `toml:"name"`
	BlockName       string   `toml:"block_name,omitempty"`       // routing-resolved block type; empty for FFN/global
	FFNExpertRouted bool     `toml:"ffn_expert_routed,omitempty"` // true for MoE FFN modules (from builder contract)
	WeightContext   string   `toml:"weight_context,omitempty"`
	Weights         []string `toml:"weights,omitempty"`           // short names, get ".weight" appended on reconstruction
	Params          []string `toml:"params,omitempty"`            // short names used verbatim (bias, bare scalars, etc.)
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

// CullingStats holds aggregate counts for culled modules of a given block type.
// Populated by the caller that decides what to cull; used by the SVG summary.
type CullingStats struct {
	Count      int   // number of culled modules of this type
	Tensors    int   // total tensors (weights + params) across culled modules
	Weights    int   // total weight tensors across culled modules
	Parameters int64 // total model parameters across culled modules
	Bytes      int64 // total VRAM bytes across culled modules
}

// ModuleMap is the static structural map of a model's weight modules.
//
// Modules is the full structural list of all modules (built by BuildModuleMap).
//
// CulledIDs lists module IDs that are fully culled (weights absent from VRAM).
// Serialized as culled_modules in TOML.
//
// CulledByType summarises what was culled, keyed by block type name
// (e.g. "full_attention", "recurrent_ssm", "ffn"). Populated by the caller;
// not serialized. Used by the SVG summary box.
type ModuleMap struct {
	Modules      []Module                `toml:"modules"`
	CulledIDs    []int                   `toml:"-"`
	CulledByType map[string]CullingStats `toml:"-"`
	Method       string                  `toml:"-"` // culling method used (diagnostic)
	Prompt       string                  `toml:"-"` // prompt that triggered culling (diagnostic)
	CullLog      []string                `toml:"-"` // diagnostic messages from culling + compilation
}

// Compile produces a CullingMask from culled IDs.
// CulledIDs produce ZeroTensors (whole-tensor culling).
func (mm *ModuleMap) Compile() *CullingMask {
	cm := &CullingMask{}

	// Build ZeroTensors from CulledIDs.
	if len(mm.CulledIDs) > 0 {
		culledSet := make(map[int]bool, len(mm.CulledIDs))
		for _, id := range mm.CulledIDs {
			culledSet[id] = true
		}
		for _, m := range mm.Modules {
			if culledSet[m.ID] {
				cm.ZeroTensors = append(cm.ZeroTensors, m.FullWeightNames()...)
			}
		}
		cm.zeroSet = make(map[string]bool, len(cm.ZeroTensors))
		for _, name := range cm.ZeroTensors {
			cm.zeroSet[name] = true
		}
	}

	return cm
}

// LoadModuleMap reads a module map from a TOML file.
// The returned ModuleMap has Modules and CulledIDs populated.
func LoadModuleMap(path string) (*ModuleMap, error) {
	var loaded struct {
		Modules       []Module `toml:"modules"`
		CulledModules []int    `toml:"culled_modules"`
	}
	if err := util.LoadTOML(path, &loaded); err != nil {
		return nil, err
	}
	mm := &ModuleMap{
		Modules:   loaded.Modules,
		CulledIDs: loaded.CulledModules,
	}
	return mm, nil
}

// Clone returns a deep copy of the ModuleMap suitable for per-query mutation.
func (mm *ModuleMap) Clone() *ModuleMap {
	c := &ModuleMap{
		Modules: make([]Module, len(mm.Modules)),
		Method:  mm.Method,
		Prompt:  mm.Prompt,
	}
	for i, m := range mm.Modules {
		c.Modules[i] = m
		c.Modules[i].Weights = append([]string(nil), m.Weights...)
		c.Modules[i].Params = append([]string(nil), m.Params...)
	}
	if mm.CulledIDs != nil {
		c.CulledIDs = append([]int(nil), mm.CulledIDs...)
	}
	if mm.CulledByType != nil {
		c.CulledByType = make(map[string]CullingStats, len(mm.CulledByType))
		for k, v := range mm.CulledByType {
			c.CulledByType[k] = v
		}
	}
	return c
}

// Save writes culled module IDs as culled_modules, sorted by ascending ID.
func (mm *ModuleMap) Save(path string) error {
	type sparseModuleMap struct {
		Method        string   `toml:"method,omitempty"`
		Prompt        string   `toml:"prompt,omitempty"`
		CullLog       []string `toml:"cull_log,omitempty"`
		CulledModules []int    `toml:"culled_modules"`
	}
	var out sparseModuleMap
	out.Method = mm.Method
	out.Prompt = mm.Prompt
	out.CullLog = mm.CullLog

	// Culled IDs — always present (empty list = explicitly no culling), sorted.
	out.CulledModules = make([]int, len(mm.CulledIDs))
	copy(out.CulledModules, mm.CulledIDs)
	sort.Ints(out.CulledModules)

	var buf bytes.Buffer
	if err := toml.NewEncoder(&buf).Encode(out); err != nil {
		return err
	}
	return os.WriteFile(path, buf.Bytes(), 0644)
}
