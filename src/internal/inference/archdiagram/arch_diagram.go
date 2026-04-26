// Package archdiagram generates SVG architecture diagrams from TOML model definitions.
package archdiagram

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"inference-lab-bench/internal/inference/arch"
)

// ArchDiagramOptions controls SVG generation.
type ArchDiagramOptions struct {
	LayerCount int  // if > 0, render layer pattern strip with routing evaluation
	UseFFNAlt  bool // render using FFNAlt builder instead of FFN
}

// blockSVG holds a loaded block fragment.
type blockSVG struct {
	Content string // raw <g>...</g> content
	Width   int
	Height  int
}

// RenderDiagram writes a composite SVG for the given architecture definition.
func RenderArchDiagram(def *arch.ArchDef, blockSVGDir string, w io.Writer, opts ArchDiagramOptions) error {
	// Load block SVG fragments for all blocks and FFN referenced by this def.
	// For each block, try <blockName>.svg first (allows block-specific overrides),
	// then fall back to <builderName>.svg. Map is keyed by display key (block name
	// for blocks, builder name for FFN).
	blocks := make(map[string]*blockSVG)
	displayKeys := collectDisplayKeys(def, opts.UseFFNAlt)
	for _, dk := range displayKeys {
		svg, err := loadBlockSVGWithFallback(blockSVGDir, dk.displayKey, dk.builderName)
		if err != nil {
			return fmt.Errorf("loading block SVG %q (builder %q): %w", dk.displayKey, dk.builderName, err)
		}
		blocks[dk.displayKey] = svg
	}

	// Determine layout
	const (
		svgWidth    = 960
		margin      = 30
		boxGap      = 15
		arrowLen    = 20
		groupGap    = 10 // vertical gap between dashed group boxes
		normHeight  = 34
		normWidth   = 300
		globalBoxH  = 40
		globalBoxW  = 300
		logitsBoxH  = 32
		logitsBoxW  = 200
		patternH    = 50
	)

	routingPaths := collectRoutingPaths(def)
	ffnKey := def.FFN.Builder
	if opts.UseFFNAlt && def.FFNAlt != nil {
		ffnKey = def.FFNAlt.Builder
	}

	bw := bufio.NewWriter(w)

	// Body is written to a buffer so the final SVG height can be computed from
	// the single cursor advance below and emitted into the <svg> header.
	var bodyBuf bytes.Buffer
	body := bufio.NewWriter(&bodyBuf)

	emitGradientDefs(body)

	body.WriteString("  <defs>\n")
	for _, dk := range displayKeys {
		blk := blocks[dk.displayKey]
		fmt.Fprintf(body, "  <g id=\"block_%s\">\n", dk.displayKey)
		body.WriteString(blk.Content)
		body.WriteString("\n  </g>\n")
	}
	body.WriteString("  </defs>\n\n")

	// Title
	cx := svgWidth / 2
	title := capitalizeASCII(def.Architecture.Name) + " Architecture"
	if opts.UseFFNAlt && def.FFNAlt != nil {
		title += " (" + def.FFNAlt.Builder + " variant)"
	}
	fmt.Fprintf(body, "  <text x=\"%d\" y=\"32\" text-anchor=\"middle\" font-size=\"20\" font-weight=\"bold\" fill=\"%s\">%s</text>\n", cx, pal["ui.text_head"], title)

	// Legend (centered)
	emitLegend(body, def, cx, ffnKey)

	// (syntax hint is now inside the legend box)

	// Start layout
	cursor := 110

	// Token embedding
	embLabel := "token_embd.weight"
	if def.Architecture.TiedEmbeddings {
		embLabel += " (tied to output)"
	}
	emitGlobalBox(body, centerX(cx, globalBoxW), cursor, globalBoxW, globalBoxH, "Token Embedding", embLabel)
	cursor += globalBoxH
	emitArrow(body, cx, cursor, arrowLen)
	cursor += arrowLen

	// Layer groups
	for gi, rp := range routingPaths {
		if gi > 0 {
			cursor += groupGap
		}
		blk := blocks[rp.blockName]
		ffnBlk := blocks[ffnKey]
		strokeColor := blockColorByName(rp.blockName)

		groupH := computeLayerGroupHeight(normHeight, boxGap, arrowLen, blk.Height, ffnBlk.Height)
		groupW := svgWidth - 2*margin

		// Dashed group border
		fmt.Fprintf(body, "  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" rx=\"8\" fill=\"none\" stroke=\"%s\" stroke-width=\"1.5\" stroke-dasharray=\"6,3\" opacity=\"0.6\"/>\n",
			margin, cursor, groupW, groupH, strokeColor)

		// Group label
		fmt.Fprintf(body, "  <text x=\"%d\" y=\"%d\" font-size=\"11\" font-weight=\"600\" fill=\"%s\">%s</text>\n",
			margin+12, cursor+20, strokeColor, rp.label)
		cursor += 28

		// Pre-attn norm
		normX := centerX(cx, normWidth)
		cursor += boxGap
		emitRMSNormBox(body, normX, cursor, normWidth, normHeight, weightNameOrDefault(def.Layers.CommonWeights, arch.WeightAttnNorm))
		cursor += normHeight
		emitArrow(body, cx, cursor, arrowLen)
		cursor += arrowLen

		// Block — reference by block name (display key), not builder name
		blkX := centerX(cx, blk.Width)
		fmt.Fprintf(body, "  <use href=\"#block_%s\" transform=\"translate(%d, %d)\"/>\n", rp.blockName, blkX, cursor)

		// Residual label (right of block, inside dashed box)
		fmt.Fprintf(body, "  <text x=\"%d\" y=\"%d\" font-size=\"11\" fill=\"%s\" font-weight=\"600\">+ residual</text>\n",
			blkX+blk.Width+12, cursor+blk.Height/2, pal["ui.text_body"])

		cursor += blk.Height + boxGap

		// Post-attn norm
		emitRMSNormBox(body, normX, cursor, normWidth, normHeight, weightNameOrDefault(def.Layers.CommonWeights, arch.WeightFFNNorm))
		cursor += normHeight
		emitArrow(body, cx, cursor, arrowLen)
		cursor += arrowLen

		// FFN
		ffnX := centerX(cx, ffnBlk.Width)
		fmt.Fprintf(body, "  <use href=\"#block_%s\" transform=\"translate(%d, %d)\"/>\n", ffnKey, ffnX, cursor)
		fmt.Fprintf(body, "  <text x=\"%d\" y=\"%d\" font-size=\"11\" fill=\"%s\" font-weight=\"600\">+ residual</text>\n",
			ffnX+ffnBlk.Width+12, cursor+ffnBlk.Height/2+4, pal["ui.text_body"])
		cursor += ffnBlk.Height + boxGap + 10
	}

	// Repeat annotation
	emitArrow(body, cx, cursor, arrowLen)
	fmt.Fprintf(body, "  <text x=\"%d\" y=\"%d\" font-size=\"10\" fill=\"%s\">x repeats</text>\n", cx+8, cursor+arrowLen-4, pal["ui.text_hint"])
	cursor += arrowLen

	// Layer pattern strip
	if opts.LayerCount > 0 {
		emitLayerPattern(body, def, opts.LayerCount, margin, cursor, svgWidth-2*margin, patternH)
		cursor += patternH
		emitArrow(body, cx, cursor, arrowLen)
		cursor += arrowLen
	}

	// Final norm
	emitRMSNormBox(body, centerX(cx, normWidth), cursor, normWidth, normHeight, weightNameOrDefault(def.Weights.Global, arch.WeightOutputNorm))
	cursor += normHeight
	emitArrow(body, cx, cursor, arrowLen)
	cursor += arrowLen

	// LM Head
	lmLabel := def.Weights.Global[arch.WeightOutput]
	if def.Architecture.TiedEmbeddings {
		lmLabel = "tied: reuses token_embd.weight"
	}
	emitGlobalBox(body, centerX(cx, globalBoxW), cursor, globalBoxW, globalBoxH, "LM Head", lmLabel)
	cursor += globalBoxH
	emitArrow(body, cx, cursor, arrowLen)
	cursor += arrowLen

	// Logits
	fmt.Fprintf(body, "  <g transform=\"translate(%d, %d)\">\n", centerX(cx, logitsBoxW), cursor)
	fmt.Fprintf(body, "    <rect width=\"%d\" height=\"%d\" rx=\"6\" fill=\"%s\" stroke=\"%s\" stroke-width=\"1\"/>\n", logitsBoxW, logitsBoxH, pal["ui.box_fill"], pal["ui.box_stroke"])
	fmt.Fprintf(body, "    <text x=\"%d\" y=\"20\" text-anchor=\"middle\" font-weight=\"600\" fill=\"%s\">Logits [n_vocab]</text>\n", logitsBoxW/2, pal["ui.text_head"])
	body.WriteString("  </g>\n")
	cursor += logitsBoxH + 30

	// Footer: routing rule (omitted when trivial — single block type)
	if !arch.IsTrivialRouting(def) {
		fmt.Fprintf(body, "  <text x=\"%d\" y=\"%d\" font-size=\"11\" fill=\"%s\">", margin, cursor, pal["ui.text_sec"])
		fmt.Fprintf(body, "<tspan font-weight=\"600\">Routing:</tspan>")
		if def.Layers.Routing.Pattern != "" {
			fmt.Fprintf(body, " <tspan font-family=\"monospace\" fill=\"%s\">${%s}[@{layer_idx}]</tspan>",
				pal["ui.text_head"], def.Layers.Routing.Pattern)
		} else {
			fmt.Fprintf(body, " <tspan font-family=\"monospace\" fill=\"%s\">%s</tspan>",
				pal["ui.text_head"], def.Layers.Routing.Rule)
		}
		fmt.Fprintf(body, " -> true: %s / false: %s", def.Layers.Routing.IfTrue, def.Layers.Routing.IfFalse)
		body.WriteString("</text>\n")
	}

	// Tokens info box (bottom-right)
	if def.Tokens.ThinkOpen != "" {
		tokBoxW := 200
		tokBoxH := 38
		tokX := svgWidth - margin - tokBoxW
		tokY := cursor - 20
		fmt.Fprintf(body, "  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" rx=\"5\" fill=\"%s\" stroke=\"%s\" stroke-width=\"0.8\"/>\n",
			tokX, tokY, tokBoxW, tokBoxH, pal["ui.box_bg"], pal["ui.box_border"])
		fmt.Fprintf(body, "  <text x=\"%d\" y=\"%d\" font-size=\"10\" font-weight=\"600\" fill=\"%s\">Tokens</text>\n",
			tokX+8, tokY+15, pal["ui.text_sec"])
		fmt.Fprintf(body, "  <text x=\"%d\" y=\"%d\" font-size=\"9\" fill=\"%s\" font-family=\"monospace\">thinking: %s...%s</text>\n",
			tokX+8, tokY+28, pal["ui.text_hint"], xmlEsc(def.Tokens.ThinkOpen), xmlEsc(def.Tokens.ThinkClose))
	}

	// Flush body and emit <svg> header + background + body + close.
	finalHeight := cursor + 20
	fmt.Fprintf(bw, `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" font-family="system-ui, -apple-system, sans-serif" font-size="13">`, svgWidth, finalHeight)
	bw.WriteString("\n")
	fmt.Fprintf(bw, "  <rect width=\"%d\" height=\"%d\" fill=\"%s\" rx=\"8\"/>\n\n", svgWidth, finalHeight, pal["ui.canvas_bg"])
	body.Flush()
	bw.Write(bodyBuf.Bytes())
	fmt.Fprintf(bw, "</svg>\n")
	return bw.Flush()
}

// routingPath describes one unique block path through the layer stack.
type routingPath struct {
	blockName string // key into def.Blocks
	label     string // human-readable label
}

func collectRoutingPaths(def *arch.ArchDef) []routingPath {
	r := def.Layers.Routing
	var paths []routingPath

	// Uniform routing: single block type, no conditions
	if r.Uniform != "" {
		paths = append(paths, routingPath{blockName: r.Uniform, label: formatBuilderName(r.Uniform)})
		return paths
	}

	// Build condition text depending on routing type.
	// Keep labels short — the full rule/pattern is in the TOML, not the diagram.
	var trueCondition, falseCondition string
	if r.Pattern != "" || r.Rule != "" {
		trueCondition = "routing rule is true"
		falseCondition = "routing rule is false"
	}

	// if_true path first (typically the more common one).
	// When routing is trivial (single block type), omit the condition — it's noise.
	trivial := arch.IsTrivialRouting(def)
	if r.IfTrue != "" {
		label := formatBuilderName(r.IfTrue)
		if !trivial && trueCondition != "" {
			label += " — when: " + trueCondition
		}
		paths = append(paths, routingPath{blockName: r.IfTrue, label: label})
	}
	if r.IfFalse != "" && r.IfFalse != r.IfTrue {
		label := formatBuilderName(r.IfFalse)
		if !trivial && falseCondition != "" {
			label += " — when: " + falseCondition
		}
		paths = append(paths, routingPath{blockName: r.IfFalse, label: label})
	}
	return paths
}

// displayKeyEntry pairs a display key (used in the SVG id and blocks map)
// with the underlying builder name (used as fallback for SVG loading).
type displayKeyEntry struct {
	displayKey  string // block name for blocks, builder name for FFN
	builderName string
}

func collectDisplayKeys(def *arch.ArchDef, useFFNAlt bool) []displayKeyEntry {
	seen := map[string]bool{}
	var entries []displayKeyEntry
	add := func(dk, builder string) {
		if dk != "" && !seen[dk] {
			seen[dk] = true
			entries = append(entries, displayKeyEntry{dk, builder})
		}
	}
	blockKeys := make([]string, 0, len(def.Blocks))
	for k := range def.Blocks {
		blockKeys = append(blockKeys, k)
	}
	sort.Strings(blockKeys)
	for _, k := range blockKeys {
		add(k, def.Blocks[k].Builder)
	}
	ffnBuilder := def.FFN.Builder
	if useFFNAlt && def.FFNAlt != nil {
		ffnBuilder = def.FFNAlt.Builder
	}
	add(ffnBuilder, ffnBuilder)
	return entries
}

// loadBlockSVGWithFallback tries loading <displayKey>.svg first, then falls back to
// <builderName>.svg. This allows block-name-specific SVG overrides (e.g. swa_attention.svg)
// while still falling back to the builder's generic SVG (e.g. attention.svg).
func loadBlockSVGWithFallback(dir, displayKey, builderName string) (*blockSVG, error) {
	svg, err := loadBlockSVG(dir, displayKey)
	if err == nil {
		return svg, nil
	}
	if displayKey == builderName {
		return nil, err // no fallback possible
	}
	return loadBlockSVG(dir, builderName)
}

// blockColorByName returns the stroke color for a block name (e.g. "swa_attention").
func blockColorByName(blockName string) string {
	return pal[palPrefix(blockName)+".stroke"]
}

var bboxRe = regexp.MustCompile(`bbox:\s*(\d+)x(\d+)`)

func loadBlockSVG(dir, builderName string) (*blockSVG, error) {
	path := filepath.Join(dir, builderName+".svg")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	content := string(data)

	// Parse bbox from comment
	w, h := 400, 100 // defaults
	if m := bboxRe.FindStringSubmatch(content); m != nil {
		w, _ = strconv.Atoi(m[1])
		h, _ = strconv.Atoi(m[2])
	}

	// Strip XML comments and leading/trailing whitespace
	lines := strings.Split(content, "\n")
	var filtered []string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "<!--") {
			continue
		}
		if trimmed != "" {
			filtered = append(filtered, line)
		}
	}

	return &blockSVG{
		Content: strings.Join(filtered, "\n"),
		Width:   w,
		Height:  h,
	}, nil
}

func emitArrow(w *bufio.Writer, cx, y, length int) {
	fmt.Fprintf(w, "  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"1.5\" marker-end=\"url(#arrow)\"/>\n",
		cx, y, cx, y+length, pal["ui.arrow"])
}

func emitGradientDefs(w *bufio.Writer) {
	// Gradient defs (shared helper)
	emitGradients(w, "")
	// Marker + filter defs (arch-diagram-specific)
	w.WriteString("  <defs>\n")
	w.WriteString("    <marker id=\"arrow\" markerWidth=\"8\" markerHeight=\"6\" refX=\"8\" refY=\"3\" orient=\"auto\">\n")
	fmt.Fprintf(w, "      <path d=\"M0,0 L8,3 L0,6\" fill=\"%s\"/>\n", pal["ui.arrow"])
	w.WriteString("    </marker>\n")
	w.WriteString("    <filter id=\"shadow\" x=\"-2%\" y=\"-2%\" width=\"104%\" height=\"104%\">\n")
	w.WriteString("      <feDropShadow dx=\"1\" dy=\"1\" stdDeviation=\"2\" flood-opacity=\"0.12\"/>\n")
	w.WriteString("    </filter>\n")
	w.WriteString("  </defs>\n")
}

func emitLegend(w *bufio.Writer, def *arch.ArchDef, cx int, ffnKey string) {
	// First pass: compute total width
	type legendItem struct {
		label, fill, stroke string
	}
	// Gradient ID mapping for arch diagram legend
	gradID := map[string]string{
		arch.TypeFullAttention: "url(#full_attention)",
		arch.TypeSWA:           "url(#swa)",
		arch.TypeRecurrent:     "url(#recurrent)",
		arch.TypeFFN:           "url(#ffn)",
	}
	var items []legendItem
	items = append(items, legendItem{"Global", "url(#global)", pal[arch.ModuleGlobal+".stroke"]})
	items = append(items, legendItem{"RMSNorm", "url(#norm)", pal[arch.TypeNorm+".stroke"]})

	// Check if model has recurrent blocks — if not, simplify attention label.
	hasRecurrent := false
	for _, blk := range def.Blocks {
		if palPrefix(blk.Builder) == arch.TypeRecurrent {
			hasRecurrent = true
			break
		}
	}
	// Check if multiple blocks share the same builder — if so, use block names
	// as legend labels instead of builder names to avoid duplicate entries.
	builderCount := map[string]int{}
	for _, blk := range def.Blocks {
		builderCount[blk.Builder]++
	}
	blkNames := make([]string, 0, len(def.Blocks))
	for name := range def.Blocks {
		blkNames = append(blkNames, name)
	}
	sort.Strings(blkNames)
	for _, name := range blkNames {
		blk := def.Blocks[name]
		pp := palPrefix(name)
		var label string
		if builderCount[blk.Builder] > 1 {
			// Multiple blocks share this builder — use block name to differentiate
			label = formatBuilderName(name)
		} else {
			label = formatBuilderName(blk.Builder)
			if pp == arch.TypeFullAttention && !hasRecurrent {
				label = "attention"
			}
		}
		items = append(items, legendItem{label, gradID[pp], pal[pp+".stroke"]})
	}
	ffnPP := palPrefix(ffnKey)
	items = append(items, legendItem{formatBuilderName(ffnKey), gradID[ffnPP], pal[ffnPP+".stroke"]})

	totalW := 0
	for _, it := range items {
		totalW += 24 + len(it.label)*7 + 16
	}
	boxPad := 10
	boxW := totalW + boxPad*2
	boxX := cx - boxW/2
	fmt.Fprintf(w, "  <rect x=\"%d\" y=\"44\" width=\"%d\" height=\"44\" rx=\"5\" fill=\"%s\" stroke=\"%s\" stroke-width=\"0.8\"/>\n",
		boxX, boxW, pal["ui.box_bg"], pal["ui.box_border"])
	fmt.Fprintf(w, "  <g transform=\"translate(%d, 52)\">\n", boxX+boxPad)
	x := 0
	for _, it := range items {
		fmt.Fprintf(w, "    <rect x=\"%d\" width=\"18\" height=\"14\" rx=\"3\" fill=\"%s\" stroke=\"%s\" stroke-width=\"0.8\"/>\n", x, it.fill, it.stroke)
		fmt.Fprintf(w, "    <text x=\"%d\" y=\"12\" font-size=\"11\" fill=\"%s\">%s</text>\n", x+24, pal["ui.text_body"], it.label)
		x += 24 + len(it.label)*7 + 16
	}
	w.WriteString("  </g>\n")
	fmt.Fprintf(w, "  <text x=\"%d\" y=\"78\" text-anchor=\"middle\" font-size=\"9\" fill=\"%s\" font-family=\"monospace\">@{name} = engine builtin, ${name} = GGUF-resolved param</text>\n", cx, pal["ui.text_hint"])
}

func emitLayerPattern(w *bufio.Writer, def *arch.ArchDef, nLayers, x, y, width, height int) {
	fmt.Fprintf(w, "  <g transform=\"translate(%d, %d)\">\n", x, y)
	fmt.Fprintf(w, "    <rect width=\"%d\" height=\"%d\" rx=\"6\" fill=\"%s\" stroke=\"%s\" stroke-width=\"1\" filter=\"url(#shadow)\"/>\n", width, height, pal["ui.box_bg"], pal["ui.box_border"])
	interval := 0
	title := fmt.Sprintf("Layer Pattern (example: %d layers", nLayers)
	if !arch.IsTrivialRouting(def) {
		if def.Example.FullAttnEvery > 0 {
			title += fmt.Sprintf(", full attention every %d", def.Example.FullAttnEvery)
		} else if def.Example.AttnPatternTrueEvery > 0 {
			interval = def.Example.AttnPatternTrueEvery
			title += fmt.Sprintf(", attention pattern true every %d", interval)
		} else if def.Example.AttnPatternFalseEvery > 0 {
			interval = def.Example.AttnPatternFalseEvery
			title += fmt.Sprintf(", attention pattern false every %d", interval)
		}
	}
	title += ")"
	fmt.Fprintf(w, "    <text x=\"%d\" y=\"19\" text-anchor=\"middle\" font-size=\"12\" font-weight=\"600\" fill=\"%s\">%s</text>\n", width/2, pal["ui.text_head"], title)

	if nLayers > 0 {
		// Build fallback params for diagram rendering (no GGUF available).
		rp := diagramFallbackParams(def, nLayers)

		boxW := min(max((width-60)/nLayers, 6), 18)
		gap := 1
		if boxW > 10 {
			gap = 3
		}
		totalW := nLayers*(boxW+gap) - gap
		startX := (width - totalW) / 2

		fmt.Fprintf(w, "    <g transform=\"translate(%d, 30)\">\n", startX)
		for i := range nLayers {
			blockName := resolveBlockForDiagram(def, i, rp)
			pp := palPrefix(blockName)
			color := pal[pp+".stroke"]
			fill := pal[pp+".grad_bottom"]
			bx := i * (boxW + gap)
			fmt.Fprintf(w, "      <rect x=\"%d\" width=\"%d\" height=\"12\" rx=\"2\" fill=\"%s\" stroke=\"%s\" stroke-width=\"0.6\"/>", bx, boxW, fill, color)
			if boxW >= 14 {
				fmt.Fprintf(w, "<text x=\"%d\" y=\"10\" text-anchor=\"middle\" font-size=\"7\" fill=\"%s\">%d</text>", bx+boxW/2, pal["ui.text_head"], i)
			}
			w.WriteString("\n")
		}
		w.WriteString("    </g>\n")
	}

	w.WriteString("  </g>\n")
}

// --- Helpers ---

func centerX(center, width int) int { return center - width/2 }

func weightNameOrDefault(weights map[string]string, key string) string {
	if name := weights[key]; name != "" {
		return name
	}
	// hard coded magic string is bad.
	return key + ".weight"
}

func computeLayerGroupHeight(normHeight, boxGap, arrowLen, blockH, ffnH int) int {
	return 28 + boxGap + normHeight + arrowLen + blockH + boxGap + normHeight + arrowLen + ffnH + boxGap + 10
}

func emitRMSNormBox(bw *bufio.Writer, x, y, width, height int, paramName string) {
	fmt.Fprintf(bw, "  <g transform=\"translate(%d, %d)\">\n", x, y)
	fmt.Fprintf(bw, "    <rect width=\"%d\" height=\"%d\" rx=\"5\" fill=\"url(#norm)\" stroke=\"%s\" stroke-width=\"1\" filter=\"url(#shadow)\"/>\n", width, height, pal["norm.stroke"])
	fmt.Fprintf(bw, "    <text x=\"%d\" y=\"14\" text-anchor=\"middle\" font-weight=\"600\" fill=\"%s\">RMSNorm</text>\n", width/2, pal["norm.text"])
	fmt.Fprintf(bw, "    <text x=\"%d\" y=\"27\" text-anchor=\"middle\" font-size=\"10\" fill=\"%s\">%s</text>\n", width/2, pal["ui.text_sub"], paramName)
	bw.WriteString("  </g>\n")
}

func emitGlobalBox(bw *bufio.Writer, x, y, width, height int, title, label string) {
	fmt.Fprintf(bw, "  <g transform=\"translate(%d, %d)\">\n", x, y)
	fmt.Fprintf(bw, "    <rect width=\"%d\" height=\"%d\" rx=\"6\" fill=\"url(#global)\" stroke=\"%s\" stroke-width=\"1.2\" filter=\"url(#shadow)\"/>\n", width, height, pal["global.stroke"])
	fmt.Fprintf(bw, "    <text x=\"%d\" y=\"17\" text-anchor=\"middle\" font-weight=\"600\" fill=\"%s\">%s</text>\n", width/2, pal["global.text"], title)
	fmt.Fprintf(bw, "    <text x=\"%d\" y=\"32\" text-anchor=\"middle\" font-size=\"10\" fill=\"%s\">%s</text>\n", width/2, pal["ui.text_sub"], label)
	bw.WriteString("  </g>\n")
}

func formatBuilderName(name string) string {
	return strings.ReplaceAll(name, "_", " ")
}

