package arch

import "strings"

// DiagramPalette returns the canonical color map for all SVG diagram rendering.
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
func diagramPalette() map[string]string {
	return map[string]string{
		// full_attention blocks
		"full_attention.stroke":      "#1976D2",
		"full_attention.fill":        "#cce5f6",
		"full_attention.grad_top":    "#e3f2fd",
		"full_attention.grad_bottom": "#bbdefb",

		// swa (sliding-window attention) blocks — uses alt-attention green (same family as recurrent)
		"swa.stroke":      "#66BB6A",
		"swa.fill":        "#c8eac8",
		"swa.grad_top":    "#e8f5e9",
		"swa.grad_bottom": "#c8e6c9",

		// recurrent blocks (recurrent_ssm, recurrent_delta_net, etc.)
		"recurrent.stroke":      "#66BB6A",
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
		"ffn_moe.lid":    "#ffe8cc",

		// UI / chrome — structural elements
		"ui.spine":      "#AAAAAA",
		"ui.dot":        "#888888",
		"ui.divider":    "#DDDDDD",
		"ui.arrow":      "#555555",

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
}

// palPrefixBuilder maps a builder name (e.g. "full_attention_gated", "gated_delta_net")
// to its palette key prefix. Used by arch_diagram.go where the input is a builder name
// rather than a block type.
func palPrefixBuilder(builderName string) string {
	switch {
	case strings.Contains(builderName, "attention"):
		return "full_attention"
	case strings.Contains(builderName, "swiglu") || strings.Contains(builderName, "moe") || strings.Contains(builderName, "geglu"):
		return "ffn"
	default:
		return "recurrent"
	}
}

// palPrefix maps a GGUF block type or module type to its palette key prefix.
// swa_* → "swa"; *attention* → "full_attention"; ffn/ffn_moe → "ffn"; global → "global";
// everything else (recurrent_ssm, recurrent_delta_net, …) → "recurrent".
func palPrefix(blockType string) string {
	switch {
	case strings.HasPrefix(blockType, "swa"):
		return "swa"
	case strings.Contains(blockType, "attention"):
		return "full_attention"
	case blockType == "ffn" || blockType == "ffn_moe":
		return "ffn"
	case blockType == "global":
		return "global"
	default:
		return "recurrent"
	}
}
