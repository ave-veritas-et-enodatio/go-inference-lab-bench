package archeditor

import (
	"fmt"
	"sort"
	"strings"

	"inference-lab-bench/internal/inference/arch"
)

// SerializeToTOML converts an ArchDef to TOML text.
func SerializeToTOML(def *arch.ArchDef) []byte {
	var b strings.Builder

	// [architecture]
	b.WriteString("[architecture]\n")
	writeStr(&b, "name", def.Architecture.Name)
	if def.Architecture.TiedEmbeddings {
		b.WriteString("tied_embeddings = true\n")
	}
	b.WriteString("\n")

	// [params] flat keys
	b.WriteString("[params]\n")
	for _, k := range sortedKeys(def.Params.Keys) {
		writeStr(&b, k, def.Params.Keys[k])
	}
	b.WriteString("\n")

	// [params.derived]
	if len(def.Params.Derived) > 0 {
		b.WriteString("[params.derived]\n")
		for _, k := range sortedKeys(def.Params.Derived) {
			writeStr(&b, k, def.Params.Derived[k])
		}
		b.WriteString("\n")
	}

	// [params.defaults]
	if len(def.Params.Defaults) > 0 {
		b.WriteString("[params.defaults]\n")
		for _, k := range sortedKeys(def.Params.Defaults) {
			writeStr(&b, k, def.Params.Defaults[k])
		}
		b.WriteString("\n")
	}

	// [weights.global]
	b.WriteString("[weights.global]\n")
	for _, k := range sortedKeys(def.Weights.Global) {
		writeStr(&b, k, def.Weights.Global[k])
	}
	b.WriteString("\n")

	// [layers]
	b.WriteString("[layers]\n")
	writeStr(&b, "count", def.Layers.Count)
	writeStr(&b, "prefix", def.Layers.Prefix)
	b.WriteString("\n")

	// [layers.routing]
	b.WriteString("[layers.routing]\n")
	writeStr(&b, "rule", def.Layers.Routing.Rule)
	writeStr(&b, "if_true", def.Layers.Routing.IfTrue)
	writeStr(&b, "if_false", def.Layers.Routing.IfFalse)
	b.WriteString("\n")

	// [layers.common_weights]
	b.WriteString("[layers.common_weights]\n")
	for _, k := range sortedKeys(def.Layers.CommonWeights) {
		writeStr(&b, k, def.Layers.CommonWeights[k])
	}
	b.WriteString("\n")

	// [blocks.*]
	for _, name := range sortedKeys(def.Blocks) {
		blk := def.Blocks[name]
		fmt.Fprintf(&b, "[blocks.%s]\n", name)
		writeStr(&b, "builder", blk.Builder)
		b.WriteString("\n")

		fmt.Fprintf(&b, "  [blocks.%s.weights]\n", name)
		for _, k := range sortedKeys(blk.Weights) {
			fmt.Fprintf(&b, "  ")
			writeStr(&b, k, blk.Weights[k])
		}
		b.WriteString("\n")

		if len(blk.Config) > 0 {
			fmt.Fprintf(&b, "  [blocks.%s.config]\n", name)
			for _, k := range sortedKeys(blk.Config) {
				fmt.Fprintf(&b, "  ")
				writeAny(&b, k, blk.Config[k])
			}
			b.WriteString("\n")
		}

		if len(blk.Cache) > 0 {
			fmt.Fprintf(&b, "  [blocks.%s.cache]\n", name)
			for _, k := range sortedKeys(blk.Cache) {
				cd := blk.Cache[k]
				quoted := make([]string, len(cd.Dims))
				for i, d := range cd.Dims {
					quoted[i] = fmt.Sprintf("%q", d)
				}
				fmt.Fprintf(&b, "  %s = { dims = [%s], dtype = %q }\n",
					k, strings.Join(quoted, ", "), cd.Dtype)
			}
			b.WriteString("\n")
		}
	}

	// [ffn]
	b.WriteString("[ffn]\n")
	writeStr(&b, "builder", def.FFN.Builder)
	b.WriteString("\n")

	b.WriteString("  [ffn.weights]\n")
	for _, k := range sortedKeys(def.FFN.Weights) {
		fmt.Fprintf(&b, "  ")
		writeStr(&b, k, def.FFN.Weights[k])
	}
	b.WriteString("\n")

	if len(def.FFN.Config) > 0 {
		b.WriteString("  [ffn.config]\n")
		for _, k := range sortedKeys(def.FFN.Config) {
			fmt.Fprintf(&b, "  ")
			writeAny(&b, k, def.FFN.Config[k])
		}
		b.WriteString("\n")
	}

	return []byte(b.String())
}

func writeStr(b *strings.Builder, k, v string) {
	fmt.Fprintf(b, "%s = %q\n", k, v)
}

func writeAny(b *strings.Builder, k string, v any) {
	switch val := v.(type) {
	case bool:
		fmt.Fprintf(b, "%s = %v\n", k, val)
	case string:
		fmt.Fprintf(b, "%s = %q\n", k, val)
	case float64:
		if val == float64(int64(val)) {
			fmt.Fprintf(b, "%s = %d\n", k, int64(val))
		} else {
			fmt.Fprintf(b, "%s = %v\n", k, val)
		}
	default:
		fmt.Fprintf(b, "%s = %v\n", k, val)
	}
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
