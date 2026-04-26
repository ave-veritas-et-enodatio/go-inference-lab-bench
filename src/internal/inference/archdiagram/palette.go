package archdiagram

import (
	"fmt"
	"io"

	"inference-lab-bench/internal/inference/arch"
	"inference-lab-bench/internal/log"
)

// pal is the canonical color map for all SVG diagram rendering.
//
// Package-private; mutation is prevented by the absence of any exported
// setter or map reference.
//
// Keys use dot-separated component.property naming:
//
//	element.fill          — light interior
//	element.stroke        — outline
//	element.grad_top      — cell gradient top stop
//	element.grad_bottom   — cell gradient bottom stop
//
// Block-type elements use the GGUF block type name directly where possible:
//
//	full_attention.*      — standard transformer attention blocks
//	recurrent.*           — SSM, delta-net, and other recurrent blocks
//	ffn.*                 — feed-forward layers (dense and MoE)
//	global.*              — global tensors (token_embd, output_norm, output)
//	norm.*                — normalization tensors (attn_norm, ffn_norm, output_norm)
//	ffn_moe.*             — MoE-specific extras (expert box lid)
//	ui.*                  — structural / chrome elements
var pal = map[string]string{
	// full_attention blocks
	"full_attention.stroke":      "#1976D2",
	"full_attention.fill":        "#cce5f6",
	"full_attention.grad_top":    "#e3f2fd",
	"full_attention.grad_bottom": "#bbdefb",

	// swa (sliding-window attention) blocks — uses alt-attention green (same family as recurrent)
	"swa.stroke":      "#4E9650",
	"swa.fill":        "#c8eac8",
	"swa.grad_top":    "#e8f5e9",
	"swa.grad_bottom": "#c8e6c9",

	// recurrent blocks (recurrent_ssm, recurrent_delta_net, etc.)
	"recurrent.stroke":      "#4E9650",
	"recurrent.fill":        "#c8eac8",
	"recurrent.grad_top":    "#e8f5e9",
	"recurrent.grad_bottom": "#c8e6c9",

	// FFN layers (ffn, ffn_moe)
	"ffn.stroke":      "#EF6C00",
	"ffn.fill":        "#ffddb8",
	"ffn.grad_top":    "#fff3e0",
	"ffn.grad_bottom": "#ffe0b2",

	// global module
	"global.stroke":      "#C62828",
	"global.text":        "#b71c1c",
	"global.fill":        "#f8bbd0",
	"global.grad_top":    "#fce4ec",
	"global.grad_bottom": "#f8bbd0",

	// norm tensors (attn_norm, ffn_norm, post_attention_norm, output_norm)
	"norm.stroke":      "#9C27B0",
	"norm.fill":        "#e1bee7",
	"norm.grad_top":    "#f3e5f5",
	"norm.grad_bottom": "#e1bee7",
	"norm.text":        "#6a1b9a",

	// MoE expert box lid
	"ffn_moe.lid": "#ffe8cc",

	// UI / chrome — structural elements
	"ui.spine":   "#AAAAAA",
	"ui.dot":     "#888888",
	"ui.trim":    "#404040",
	"ui.divider": "#DDDDDD",
	"ui.arrow":   "#555555",

	// UI / surfaces
	"ui.canvas_bg":  "#fafafa",
	"ui.box_bg":     "#f5f5f5",
	"ui.box_bg_alt": "#f9f9f9",
	"ui.box_border": "#bdbdbd",
	"ui.box_fill":   "#eeeeee",
	"ui.box_stroke": "#9e9e9e",

	// UI / text hierarchy (darkest → lightest)
	"ui.text_title":  "#111111",
	"ui.text_head":   "#333333",
	"ui.text_body":   "#555555",
	"ui.text_sec":    "#666666",
	"ui.text_sub":    "#777777",
	"ui.text_label":  "#888888",
	"ui.text_hint":   "#999999",
	"ui.text_tensor": "#000000",
}

// kindPalPrefix maps a registered builder's Kind to its palette prefix.
// First-pass lookup in palPrefix.
var kindPalPrefix = map[arch.BuilderKind]string{
	arch.KindAttention:    arch.TypeFullAttention,
	arch.KindSWAAttention: arch.TypeSWA,
	arch.KindRecurrent:    arch.TypeRecurrent,
	arch.KindFFN:          arch.TypeFFN,
}

// moduleTypePalPrefix maps TOML block names and module-type strings that do
// not round-trip through the builder registry to a palette prefix. Extend
// this map when introducing a new .arch.toml block name whose palette color
// is not already determined by its builder kind (e.g. sliding-window
// attention shares the "attention" builder but renders in "swa" green; MLA
// attention shares the "mla_attention" builder kind but the block name
// "mla" is used directly in routing path labels and needs an explicit entry).
var moduleTypePalPrefix = map[string]string{
	// Module types emitted by BuildModuleMap.
	arch.ModuleGlobal: arch.ModuleGlobal,
	arch.TypeFFN:      arch.TypeFFN,
	arch.TypeFFNMoE:   arch.TypeFFN,
	arch.TypeNorm:     arch.TypeNorm,

	// Built-in .arch.toml block names (models/arch/*.arch.toml).
	arch.TypeFullAttention: arch.TypeFullAttention,
	arch.TypeSWA:           arch.TypeSWA,
	"swa_attention":        arch.TypeSWA,
	"recurrent_ssm":        arch.TypeRecurrent,
	"mla":                  arch.TypeFullAttention,
}

// Note: unifying palPrefix and palPrefixBuilder into a single function means
// a block's palette prefix is now computed once, independent of caller site.
// Previously emitLegend had a special fallback that colored "mla" as
// full_attention (blue) while other call sites used palPrefix("mla") → default
// "recurrent" (green), producing an inconsistent diagram. The new behavior
// colors "mla" as full_attention everywhere — a semantic correction bundled
// with the refactor. See BURNDOWN_PLAN §"Item 4" for the design rationale.

// palPrefix maps a TOML block name, module-type string, or registered builder
// name to its palette prefix. Registered builders win first (authoritative by
// Kind); module-type/block-name registry is the fallback for names that do
// not appear as builders. Unknown names warn and return the recurrent prefix.
//
// No substring or prefix matching — adding a new block name is a data
// operation on one of the two maps.
func palPrefix(name string) string {
	if bb, ok := arch.GetBlockBuilder(name); ok {
		if p, ok := kindPalPrefix[bb.Contract().Kind]; ok {
			return p
		}
	}
	if fb, ok := arch.GetFFNBuilder(name); ok {
		if p, ok := kindPalPrefix[fb.Contract().Kind]; ok {
			return p
		}
	}
	if p, ok := moduleTypePalPrefix[name]; ok {
		return p
	}
	log.Warn("palPrefix: unknown type %q — defaulting to recurrent", name)
	return arch.TypeRecurrent
}

// emitGradients writes a self-contained <defs> block of linearGradient elements,
// one per palette family. idPrefix is prepended to each gradient ID.
// Called by both renderers; each uses its own ID prefix to avoid conflicts.
func emitGradients(w io.Writer, idPrefix string) {
	fmt.Fprintf(w, "  <defs>\n")
	for _, entry := range []struct{ suffix, prefix string }{
		{"recurrent", arch.TypeRecurrent},
		{"full_attention", arch.TypeFullAttention},
		{"swa", arch.TypeSWA},
		{"ffn", arch.TypeFFN},
		{"norm", arch.TypeNorm},
		{"global", arch.ModuleGlobal},
	} {
		fmt.Fprintf(w, "    <linearGradient id=\"%s%s\" x1=\"0\" y1=\"0\" x2=\"0\" y2=\"1\">\n", idPrefix, entry.suffix)
		fmt.Fprintf(w, "      <stop offset=\"0%%\" stop-color=\"%s\"/><stop offset=\"100%%\" stop-color=\"%s\"/>\n",
			pal[entry.prefix+".grad_top"], pal[entry.prefix+".grad_bottom"])
		fmt.Fprintf(w, "    </linearGradient>\n")
	}
	fmt.Fprintf(w, "  </defs>\n")
}
