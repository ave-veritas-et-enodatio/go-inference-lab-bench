package arch

import (
	"sort"
	"strings"
)

// DiagramPalette returns the canonical color map for all SVG diagram rendering.
// Keys use dot-separated component.property naming:
//
//	element.fill          — light interior (active)
//	element.stroke        — outline (active)
//	element.fill_de       — desaturated interior (culled)
//	element.stroke_de     — desaturated outline (culled)
//	element.grad_top      — cell gradient top stop (active)
//	element.grad_bottom   — cell gradient bottom stop (active)
//	element.cell_de       — outer cell flat fill (culled)
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
func diagramPalette() map[string]string {
	return map[string]string{
		// full_attention blocks
		"full_attention.stroke":      "#1976D2",
		"full_attention.fill":        "#cce5f6",
		"full_attention.stroke_de":   "#7099B8",
		"full_attention.fill_de":     "#BFD0E0",
		"full_attention.cell_de":     "#CAD8E8",
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
		"recurrent.stroke_de":   "#70A068",
		"recurrent.fill_de":     "#BDDABA",
		"recurrent.cell_de":     "#C8DEC8",
		"recurrent.grad_top":    "#e8f5e9",
		"recurrent.grad_bottom": "#c8e6c9",

		// FFN layers (ffn, ffn_moe)
		"ffn.stroke":      "#EF6C00",
		"ffn.fill":        "#ffddb8",
		"ffn.stroke_de":   "#B08048",
		"ffn.fill_de":     "#E0C9A0",
		"ffn.cell_de":     "#E4D0B8",
		"ffn.grad_top":    "#fff3e0",
		"ffn.grad_bottom": "#ffe0b2",

		// global module
		"global.stroke":      "#C62828",
		"global.text":        "#b71c1c",
		"global.fill":        "#f8bbd0",
		"global.stroke_de":   "#A07080",
		"global.fill_de":     "#DFBFC8",
		"global.grad_top":    "#fce4ec",
		"global.grad_bottom": "#f8bbd0",

		// norm tensors (attn_norm, ffn_norm, post_attention_norm, output_norm)
		"norm.stroke":      "#9C27B0",
		"norm.fill":        "#e1bee7",
		"norm.stroke_de":   "#8868A0",
		"norm.fill_de":     "#C8A8D0",
		"norm.grad_top":    "#f3e5f5",
		"norm.grad_bottom": "#e1bee7",
		"norm.text":        "#6a1b9a",

		// MoE expert box lid
		"ffn_moe.lid":    "#ffe8cc",
		"ffn_moe.lid_de": "#EDD8B8",

		// UI / chrome — structural elements
		"ui.spine":      "#AAAAAA",
		"ui.dot":        "#888888",
		"ui.trim":       "#404040",
		"ui.divider":    "#DDDDDD",
		"ui.arrow":      "#555555",

		// UI / surfaces
		"ui.canvas_bg":  "#fafafa",
		"ui.box_bg":     "#f5f5f5",
		"ui.box_bg_alt": "#f9f9f9",
		"ui.box_border": "#bdbdbd",
		"ui.box_fill":   "#eeeeee",
		"ui.box_stroke": "#9e9e9e",

		// Engagement overlays
		"engage.hot": "#D32F2F", // engagement heat — warm red (Material 700)

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
}

// palPrefixBuilder maps a builder name (e.g. "full_attention_gated", "gated_delta_net")
// to its palette key prefix. Used by arch_diagram.go where the input is a builder name
// rather than a block type.
func palPrefixBuilder(builderName string) string {
	switch {
	case strings.Contains(builderName, "attention"):
		return TypeFullAttention
	case strings.Contains(builderName, "swiglu") || strings.Contains(builderName, "moe") || strings.Contains(builderName, "geglu"):
		return TypeFFN
	default:
		return TypeRecurrent
	}
}

// palPrefix maps a GGUF block type or module type to its palette key prefix.
// swa_* → "swa"; *attention* → "full_attention"; ffn/ffn_moe → "ffn"; global → "global";
// everything else (recurrent_ssm, recurrent_delta_net, …) → "recurrent".
func palPrefix(blockType string) string {
	switch {
	case strings.HasPrefix(blockType, TypeSWA):
		return TypeSWA
	case strings.Contains(blockType, "attention"):
		return TypeFullAttention
	case blockType == TypeFFN || blockType == TypeFFNMoE:
		return TypeFFN
	case blockType == ModuleGlobal:
		return ModuleGlobal
	default:
		return TypeRecurrent
	}
}

// DiagramPalette returns the diagram color palette (exported for external use).
func DiagramPalette() map[string]string { return diagramPalette() }

// PalPrefixBuilder returns the palette prefix for a given builder name (exported).
func PalPrefixBuilder(name string) string { return palPrefixBuilder(name) }

// BlockBuilderNames returns all registered block builder names, sorted.
func BlockBuilderNames() []string {
	names := make([]string, 0, len(blockBuilders))
	for k := range blockBuilders {
		names = append(names, k)
	}
	sort.Strings(names)
	return names
}

// FFNBuilderNames returns all registered FFN builder names, sorted.
func FFNBuilderNames() []string {
	names := make([]string, 0, len(ffnBuilders))
	for k := range ffnBuilders {
		names = append(names, k)
	}
	sort.Strings(names)
	return names
}
