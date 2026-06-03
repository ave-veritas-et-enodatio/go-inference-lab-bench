package archdiagram

import (
	"fmt"
	"io"
	"strings"

	"inference-lab-bench/internal/inference/arch"
)

// vision-layers layout constants. Independent of the decoder's layers diagram
// (the vision encoder is uniform — one block repeated — so there is no routing
// or MoE variation to lay out), but the macro 2-column "U" shape is the same:
// it keeps a deep encoder (Qwen3.5 = 27 layers) from being absurdly tall.
const (
	vlCellW   = 150 // attention cell width
	vlFfnW    = 132 // FFN cell width
	vlNormW   = 16  // left norm-slot width inside a cell
	vlCellH   = 46
	vlCellPad = 4
	vlSubH    = 9 // per-tensor sub-rect height in the stacked Q/K/V column
	vlSubGap  = 3
	vlDivW    = 3 // divider gap between logical columns

	vlColLeftX  = 26 // attention cell absolute left in a column
	vlCellGap   = 30 // gap between attention and FFN cells
	vlFfnLeftX  = vlColLeftX + vlCellW + vlCellGap
	vlRowOutX   = vlFfnLeftX + vlFfnW + vlCellGap
	vlTrunkInX  = 10
	vlUturnGap  = 16
	vlColGap    = 36
	vlRightOffX = vlRowOutX + vlColGap

	vlTrunkW = "2.5"
)

// visionLayerSymbols holds the two pre-rendered cell symbols (attention + FFN)
// for the uniform vision encoder, plus the cell height they were rendered at.
// cellH is derived from the attention block's projection-row count so a tall
// separate-Q/K/V/Qn/Kn stack (Gemma, 5 rows) is fully contained, while a fused
// single-QKV stack (Qwen, 1 row) stays compact — no architecture branch.
type visionLayerSymbols struct {
	attnID, ffnID   string
	attnSVG, ffnSVG string
	cellH           int
}

// RenderVisionLayersDiagram renders the vision encoder stack fully exploded:
// every layer drawn end-to-end with its full per-module tensor detail (norm →
// attention → norm → FFN). The tower is UNIFORM — one block repeated — so every
// layer is identical detail; the repetition is intentional (tutorial value).
//
// All per-arch differences (norm_type, fused-vs-separate QKV, FFN type/activation,
// projector shape) are read from def.Vision / def.Projector — no architecture
// branches. Layout: 2-column "U" snake (left descends, right ascends) so a deep
// encoder stays legible.
func RenderVisionLayersDiagram(def *arch.ArchDef, n int, w io.Writer) error {
	if def.Vision == nil {
		return fmt.Errorf("RenderVisionLayersDiagram: def has no [vision] section")
	}
	if n <= 0 {
		return fmt.Errorf("RenderVisionLayersDiagram: example vision layer count must be > 0")
	}
	v := def.Vision
	syms := buildVisionLayerSymbols(def)

	title := capitalizeASCII(def.Architecture.Name) + " Vision Tower — Layers"

	// --- Column split: left gets first half (rounded up), right gets the rest. ---
	leftCount := (n + 1) / 2
	rightCount := n - leftCount

	// --- Vertical layout. ---
	// Layer rows use the symbol cell height (sized to contain the attention
	// projection stack), so the row pitch must track it too — otherwise a tall
	// cell would overlap the next row.
	cellH := syms.cellH
	pitch := cellH + vlUturnGap
	titleBandH := 64
	entryCY := titleBandH + vlCellH/2 + 20           // patch-embed entry cell, with headroom for its IMAGE PATCHES label
	firstRowCY := entryCY + vlCellH/2 + pitch/2 + 16 // first layer row below the entry cell
	leftCY := func(i int) int { return firstRowCY + i*pitch }
	bottomCY := leftCY(leftCount - 1)
	rightCY := func(j int) int { return bottomCY - j*pitch }

	// --- Canvas dimensions. ---
	contentRightEdge := vlRightOffX + vlRowOutX + 16
	legBoxX := contentRightEdge + 8
	legBoxW := 188
	canvasW := legBoxX + legBoxW + 12

	bottomHopY := bottomCY + cellH/2 + vlUturnGap/2
	contentBottom := bottomHopY + 28
	legBoxY := titleBandH + 8
	legBoxH := 10 + 5*18 + 10
	gutterBottom := legBoxY + legBoxH + 16
	canvasH := max(contentBottom, gutterBottom)

	var b strings.Builder
	fmt.Fprintf(&b, `<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg">
<style>
  text { font-family: 'Courier New', monospace; }
  .title { font-size: 20px; font-weight: bold; fill: %s; font-family: system-ui, -apple-system, sans-serif; }
  .lbl   { font-size: 9px; fill: %s; text-anchor: end; dominant-baseline: middle; }
  .tlbl  { font-size: 7px; fill: %s; text-anchor: middle; dominant-baseline: middle; }
  .clbl  { font-size: 8px; font-weight: 600; fill: %s; text-anchor: middle; dominant-baseline: middle; }
  .io    { font-size: 8px; font-weight: bold; fill: %s; text-anchor: middle; dominant-baseline: middle; }
  .ltxt  { font-size: 8px; fill: %s; dominant-baseline: middle; }
</style>
`, canvasW, canvasH,
		pal["ui.text_head"], pal["ui.text_label"], pal["ui.text_tensor"],
		pal["ui.text_head"], pal["ui.dot"], pal["ui.text_body"])

	emitGradients(&b, "vl_")

	fmt.Fprintf(&b, "<defs>\n")
	fmt.Fprintf(&b, "  <symbol id=\"%s\" overflow=\"visible\">\n%s  </symbol>\n", syms.attnID, syms.attnSVG)
	fmt.Fprintf(&b, "  <symbol id=\"%s\" overflow=\"visible\">\n%s  </symbol>\n", syms.ffnID, syms.ffnSVG)
	fmt.Fprintf(&b, "</defs>\n")

	fmt.Fprintf(&b, "  <rect width=\"%d\" height=\"%d\" fill=\"%s\" rx=\"8\"/>\n\n", canvasW, canvasH, pal["ui.canvas_bg"])

	titleCX := (vlRightOffX + vlRowOutX) / 2
	fmt.Fprintf(&b, "  <text class=\"title\" x=\"%d\" y=\"32\" text-anchor=\"middle\">%s</text>\n", titleCX, title)
	fmt.Fprintf(&b, "  <text x=\"%d\" y=\"47\" text-anchor=\"middle\" font-size=\"11\" font-weight=\"600\" fill=\"%s\">%s</text>\n",
		titleCX, pal["ui.text_hint"], xmlEsc(visionLayersSubtitle(v, n)))

	// --- Patch-embed entry cell at the top of the left column. ---
	leftXOff := 0
	rightXOff := vlRightOffX
	entryCenterX := leftXOff + (vlTrunkInX+vlRowOutX)/2
	emitVisionEntryCell(&b, entryCenterX, entryCY, v)

	// Entry → layer 0: down from the entry cell, left to the column's trunk-in
	// rail, down to layer 0's row, then right into its block-left (mirrors the
	// per-row U-turn so the corner joins are clean).
	turnX := leftXOff + vlTrunkInX
	in0Y := leftCY(0)
	blockLeftX := leftXOff + vlColLeftX
	entryBottomY := entryCY + vlCellH/2
	// Turn at a quarter of the drop (not the midpoint) so the initial descender
	// off Patch Embed is half as long before its first 90° turn.
	yMid := entryBottomY + (in0Y-entryBottomY)/4
	fmt.Fprintf(&b, "  <path d=\"M %d,%d L %d,%d L %d,%d L %d,%d L %d,%d\" fill=\"none\" stroke=\"%s\" stroke-width=\"%s\" stroke-linejoin=\"miter\"/>\n",
		entryCenterX, entryBottomY, entryCenterX, yMid, turnX, yMid, turnX, in0Y, blockLeftX, in0Y, pal["ui.spine"], vlTrunkW)
	emitVisionArrowhead(&b, blockLeftX, in0Y, "right")

	// --- Left column (descending) ---
	for i := range leftCount {
		emitVisionLayerRow(&b, &syms, leftXOff, i, leftCY(i))
		if i < leftCount-1 {
			emitVisionUTurn(&b, leftXOff, leftCY(i), leftCY(i+1))
		}
	}

	// --- Column-bottom hop (last left → first right) ---
	if rightCount > 0 {
		emitVisionBottomHop(&b, leftXOff, rightXOff, bottomCY, rightCY(0))
	}

	// --- Right column (ascending) ---
	for j := range rightCount {
		l := leftCount + j
		emitVisionLayerRow(&b, &syms, rightXOff, l, rightCY(j))
		if j < rightCount-1 {
			emitVisionUTurn(&b, rightXOff, rightCY(j), rightCY(j+1))
		}
	}

	// --- Exit: top of whichever column holds the last layer → "to merge + projector". ---
	exitXOff := rightXOff
	exitCY := rightCY(rightCount - 1)
	if rightCount == 0 {
		exitXOff, exitCY = leftXOff, bottomCY
	}
	emitVisionExit(&b, exitXOff, exitCY, entryBottomY, def)

	// --- Legend ---
	emitVisionLayersLegend(&b, legBoxX, legBoxY, legBoxW, legBoxH, v)

	fmt.Fprintf(&b, "</svg>\n")
	_, err := io.WriteString(w, b.String())
	return err
}

// visionLayersSubtitle notes the example depth + that every layer is identical.
func visionLayersSubtitle(v *arch.VisionDef, n int) string {
	return fmt.Sprintf("uniform encoder · %d layers (example) · %s · bidirectional",
		n, normTypeLabel(v.NormType))
}

// --- symbol construction (data-driven, no arch branch) ---

// buildVisionLayerSymbols renders the attention + FFN cell symbols once for the
// uniform encoder block, reading the actual per-layer tensors from
// [vision.layers.common_weights] and the FFN tensors from [vision.ffn.weights].
func buildVisionLayerSymbols(def *arch.ArchDef) visionLayerSymbols {
	v := def.Vision
	cw := v.Layers.CommonWeights

	blkName := v.Layers.Routing.Uniform
	if blkName == "" {
		blkName = v.Layers.Routing.IfTrue
	}
	blk := v.Blocks[blkName]

	// Size the cell from the attention projection-row count so the stacked
	// rows are always contained. The FFN cell shares this height so the two
	// cells in a row read as a matched pair.
	cellH := visionCellHeight(len(visionAttnProjRows(cw, blk)))

	return visionLayerSymbols{
		attnID:  "vl-attn",
		ffnID:   "vl-ffn",
		cellH:   cellH,
		attnSVG: buildVisionAttnSymbol(cw, blk, cellH),
		ffnSVG:  buildVisionFFNSymbol(cw, v.FFN, cellH),
	}
}

// visionCellHeight returns a cell height that contains nProjRows stacked
// projection sub-rects (each vlSubH high, vlSubGap apart) plus top/bottom
// padding, never smaller than the base vlCellH (so 1-row cells keep the
// standard height).
func visionCellHeight(nProjRows int) int {
	stackH := nProjRows*vlSubH + (nProjRows-1)*vlSubGap
	return max(vlCellH, stackH+2*vlCellPad)
}

// visionAttnProjRows returns the projection-column row labels for the attention
// cell: a single fused "QKV" or separate "Q","K","V", plus optional QK-norm
// rows when those per-layer weights are present. Shared by the height
// computation and the renderer so they never disagree.
func visionAttnProjRows(cw map[string]string, blk arch.BlockDef) []string {
	fused, _ := blk.Config["qkv_fused"].(bool)
	var projRows []string
	if fused || cw["attn_qkv"] != "" {
		projRows = []string{"QKV"}
	} else {
		projRows = []string{"Q", "K", "V"}
	}
	if cw["attn_q_norm"] != "" {
		projRows = append(projRows, "Qn")
	}
	if cw["attn_k_norm"] != "" {
		projRows = append(projRows, "Kn")
	}
	return projRows
}

// hasCW reports whether a common-weights key is populated.
func hasCW(cw map[string]string, key string) bool { return cw[key] != "" }

// buildVisionAttnSymbol renders the attention cell: left pre-norm slot, then the
// projection tensors. Fused QKV (one attn_qkv tensor) and separate Q/K/V both
// render from the actual present keys — no arch branch. Optional QK-norm and
// attn_output appear when their keys are present.
func buildVisionAttnSymbol(cw map[string]string, blk arch.BlockDef, cellH int) string {
	var sb strings.Builder
	pp := "full_attention"
	fmt.Fprintf(&sb, "    <rect x=\"0\" y=\"0\" width=\"%d\" height=\"%d\" fill=\"url(#vl_%s)\" stroke=\"%s\" stroke-width=\"0.8\" rx=\"2\"/>\n",
		vlCellW, cellH, pp, pal[pp+".stroke"])

	normH := cellH - 2*vlCellPad
	x := 2
	drawVisionTensor(&sb, x, vlCellPad, vlNormW, normH, "norm", "ln1")
	x += vlNormW + 1
	emitVisionDivider(&sb, x, cellH)
	x += vlDivW

	// Projection column: fused QKV is one stacked label; separate is Q/K/V rows
	// (plus optional Qn/Kn). The cell height was sized to contain these rows.
	projRows := visionAttnProjRows(cw, blk)
	// Reserve a right-edge post-attention-norm slot if that per-layer weight exists.
	postNorm := hasCW(cw, "attn_post_norm")
	avail := vlCellW - x - 2
	postW := 0
	if postNorm {
		postW = vlNormW + vlDivW
		avail -= postW
	}
	projW := avail * 56 / 100
	emitVisionStackedRows(&sb, x, projW, cellH, pp, projRows)
	x += projW + 1
	emitVisionDivider(&sb, x, cellH)
	x += vlDivW

	// Output projection.
	outW := avail - projW - vlDivW - 1
	drawVisionTensor(&sb, x, vlCellPad, outW, normH, pp, "out")
	x += outW + 1

	if postNorm {
		emitVisionDivider(&sb, x, cellH)
		x += vlDivW
		drawVisionTensor(&sb, x, vlCellPad, vlNormW, normH, "norm", "pn")
	}

	return sb.String()
}

// buildVisionFFNSymbol renders the FFN cell: left pre-norm slot, gate/up (if a
// gate weight is present) or a single up column, then down. Reads the present
// keys from [vision.ffn.weights]; gated vs plain MLP both render from data.
func buildVisionFFNSymbol(cw map[string]string, ffn arch.FFNDef, cellH int) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "    <rect x=\"0\" y=\"0\" width=\"%d\" height=\"%d\" fill=\"url(#vl_ffn)\" stroke=\"%s\" stroke-width=\"0.8\" rx=\"2\"/>\n",
		vlFfnW, cellH, pal["ffn.stroke"])

	normH := cellH - 2*vlCellPad
	x := 2
	drawVisionTensor(&sb, x, vlCellPad, vlNormW, normH, "norm", "ln2")
	x += vlNormW + 1
	emitVisionDivider(&sb, x, cellH)
	x += vlDivW

	var guRows []string
	if ffn.Weights["gate"] != "" {
		guRows = []string{"G", "U"}
	} else {
		guRows = []string{"U"}
	}

	postNorm := hasCW(cw, "ffn_post_norm")
	avail := vlFfnW - x - 2
	if postNorm {
		avail -= vlNormW + vlDivW
	}
	guW := avail * 60 / 100
	emitVisionStackedRows(&sb, x, guW, cellH, "ffn", guRows)
	x += guW + 1
	emitVisionDivider(&sb, x, cellH)
	x += vlDivW

	downW := avail - guW - vlDivW - 1
	drawVisionTensor(&sb, x, vlCellPad, downW, normH, "ffn", "D")
	x += downW + 1

	if postNorm {
		emitVisionDivider(&sb, x, cellH)
		x += vlDivW
		drawVisionTensor(&sb, x, vlCellPad, vlNormW, normH, "norm", "pn")
	}

	return sb.String()
}

// emitVisionStackedRows draws a vertical stack of equal-height labeled sub-rects
// centered in the cell (used for Q/K/V or gate/up).
func emitVisionStackedRows(sb *strings.Builder, x, w, cellH int, pp string, labels []string) {
	n := len(labels)
	totalH := n*vlSubH + (n-1)*vlSubGap
	startY := (cellH - totalH) / 2
	for i, lbl := range labels {
		y := startY + i*(vlSubH+vlSubGap)
		drawVisionTensor(sb, x, y, w, vlSubH, pp, lbl)
	}
}

// drawVisionTensor draws one tensor sub-rect with a centered label. Norm slots
// use the norm palette regardless of the host cell's palette.
func drawVisionTensor(sb *strings.Builder, x, y, w, h int, pp, label string) {
	fill, stroke := pal[pp+".fill"], pal[pp+".stroke"]
	if pp == "norm" || strings.HasPrefix(label, "ln") || label == "Qn" || label == "Kn" {
		fill, stroke = pal["norm.fill"], pal["norm.stroke"]
	}
	fmt.Fprintf(sb, "    <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"%s\" stroke=\"%s\" stroke-width=\"0.5\" rx=\"1\"/>\n",
		x, y, w, h, fill, stroke)
	if label == "" || w < 8 {
		return
	}
	cx, cy := x+w/2, y+h/2
	if h > w*2 {
		fmt.Fprintf(sb, "    <text class=\"tlbl\" x=\"%d\" y=\"%d\" transform=\"rotate(-90,%d,%d)\">%s</text>\n", cx, cy, cx, cy, label)
	} else {
		fmt.Fprintf(sb, "    <text class=\"tlbl\" x=\"%d\" y=\"%d\">%s</text>\n", cx, cy, label)
	}
}

func emitVisionDivider(sb *strings.Builder, x, cellH int) {
	fmt.Fprintf(sb, "    <line x1=\"%d\" y1=\"0\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"0.7\"/>\n",
		x, x, cellH, pal["ui.divider"])
}

// --- per-row layout ---

// emitVisionLayerRow renders one encoder layer: the layer label, the attention
// and FFN cells, the trunk threading through them, and the inline residual
// adders. Mirrors the decoder layers diagram's per-row chrome.
func emitVisionLayerRow(b *strings.Builder, syms *visionLayerSymbols, xOff, l, cy int) {
	rowY := cy - syms.cellH/2
	trunkY := cy

	// Layer label — sits on the upper part of the cell: halfway between the row
	// centerline (where the entry arrowhead lands) and just above the cell top,
	// so it clears the arrowhead without floating in the inter-row gap.
	fmt.Fprintf(b, "  <text class=\"lbl\" x=\"%d\" y=\"%d\">%d</text>\n", xOff+vlColLeftX-3, (rowY-5+trunkY)/2, l)

	attnLeftX := xOff + vlColLeftX
	attnRightX := attnLeftX + vlCellW
	ffnLeftX := xOff + vlFfnLeftX
	ffnRightX := ffnLeftX + vlFfnW
	rowOutX := xOff + vlRowOutX

	fmt.Fprintf(b, "  <use href=\"#%s\" transform=\"translate(%d,%d)\"/>\n", syms.attnID, attnLeftX, rowY)
	fmt.Fprintf(b, "  <use href=\"#%s\" transform=\"translate(%d,%d)\"/>\n", syms.ffnID, ffnLeftX, rowY)

	exitY := rowY + syms.cellH*3/4

	// Attention → FFN trunk + residual ⊕.
	blkAdderX := (attnRightX + ffnLeftX) / 2
	emitVisionTrunk(b, attnRightX, ffnLeftX, trunkY, true)
	emitVisionLBend(b, attnRightX, exitY, blkAdderX, trunkY, pal["full_attention.stroke"])
	emitVisionAdder(b, blkAdderX, trunkY, pal["full_attention.stroke"])

	// FFN → row-out trunk + residual ⊕.
	ffnAdderX := (ffnRightX + rowOutX) / 2
	emitVisionTrunk(b, ffnRightX, rowOutX, trunkY, true)
	emitVisionLBend(b, ffnRightX, exitY, ffnAdderX, trunkY, pal["ffn.stroke"])
	emitVisionAdder(b, ffnAdderX, trunkY, pal["ffn.stroke"])
}

// emitVisionTrunk draws a trunk-thick horizontal line, optionally with a
// rightward arrowhead at x2.
func emitVisionTrunk(b *strings.Builder, x1, x2, y int, arrow bool) {
	fmt.Fprintf(b, "  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"%s\"/>\n",
		x1, y, x2, y, pal["ui.spine"], vlTrunkW)
	if arrow {
		emitVisionArrowhead(b, x2, y, "right")
	}
}

// emitVisionLBend draws the thin residual contribution from a cell's bottom-right
// up into the inline ⊕.
func emitVisionLBend(b *strings.Builder, cellRightX, exitY, adderX, adderY int, color string) {
	fmt.Fprintf(b, "  <path d=\"M %d,%d L %d,%d L %d,%d\" fill=\"none\" stroke=\"%s\" stroke-width=\"1.0\"/>\n",
		cellRightX, exitY, adderX, exitY, adderX, adderY+5, color)
}

// emitVisionAdder draws the residual ⊕ glyph (gray disk, colored ring, plus).
func emitVisionAdder(b *strings.Builder, cx, cy int, color string) {
	fmt.Fprintf(b, "  <circle cx=\"%d\" cy=\"%d\" r=\"5\" fill=\"%s\" stroke=\"none\"/>\n", cx, cy, pal["ui.spine"])
	fmt.Fprintf(b, "  <circle cx=\"%d\" cy=\"%d\" r=\"5\" fill=\"none\" stroke=\"%s\" stroke-width=\"1.2\"/>\n", cx, cy, color)
	fmt.Fprintf(b, "  <text x=\"%d\" y=\"%d\" text-anchor=\"middle\" dominant-baseline=\"central\" font-size=\"9\" font-weight=\"bold\" fill=\"%s\">+</text>\n", cx, cy, color)
}

// emitVisionUTurn draws the U-turn from one row's trunk-out to the next row's
// block-left edge (works ascending or descending).
func emitVisionUTurn(b *strings.Builder, xOff, cyA, cyB int) {
	yMid := (cyA + cyB) / 2
	xRight := xOff + vlRowOutX
	xLeft := xOff + vlTrunkInX
	xBlockLeft := xOff + vlColLeftX
	fmt.Fprintf(b, "  <path d=\"M %d,%d L %d,%d L %d,%d L %d,%d L %d,%d\" fill=\"none\" stroke=\"%s\" stroke-width=\"%s\" stroke-linejoin=\"miter\"/>\n",
		xRight, cyA, xRight, yMid, xLeft, yMid, xLeft, cyB, xBlockLeft, cyB, pal["ui.spine"], vlTrunkW)
	emitVisionArrowhead(b, xBlockLeft, cyB, "right")
}

// emitVisionBottomHop links the last left-column row's trunk-out to the first
// right-column row's block-left (straight across when the rows share a Y).
func emitVisionBottomHop(b *strings.Builder, leftXOff, rightXOff, cyLeft, cyRight int) {
	xLeftOut := leftXOff + vlRowOutX
	xRightIn := rightXOff + vlColLeftX
	if cyLeft == cyRight {
		fmt.Fprintf(b, "  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"%s\" stroke-linecap=\"square\"/>\n",
			xLeftOut, cyLeft, xRightIn, cyRight, pal["ui.spine"], vlTrunkW)
	} else {
		yMid := (cyLeft + cyRight) / 2
		fmt.Fprintf(b, "  <path d=\"M %d,%d L %d,%d L %d,%d L %d,%d\" fill=\"none\" stroke=\"%s\" stroke-width=\"%s\" stroke-linejoin=\"miter\"/>\n",
			xLeftOut, cyLeft, xLeftOut, yMid, xRightIn, yMid, xRightIn, cyRight, pal["ui.spine"], vlTrunkW)
	}
	emitVisionArrowhead(b, xRightIn, cyRight, "right")
}

// emitVisionArrowhead reuses the decoder diagram's triangular arrowhead.
func emitVisionArrowhead(b *strings.Builder, apexX, apexY int, dir string) {
	emitArrowhead(b, apexX, apexY, dir)
}

// --- entry / exit decorations ---

// emitVisionEntryCell draws the patch-embed entry box at the top of the left
// column with an "IMAGE PATCHES" label feeding in from above.
func emitVisionEntryCell(b *strings.Builder, centerX, cy int, v *arch.VisionDef) {
	w := vlCellW
	leftX := centerX - w/2
	topY := cy - vlCellH/2
	// Fill with the diagram's own global gradient (this renderer prefixes its
	// gradient ids with "vl_"); the shared emitGlobalCellRect hardcodes the
	// decoder diagram's "zm_global" id, which is undefined here.
	fmt.Fprintf(b, "  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"url(#vl_global)\" stroke=\"%s\" stroke-width=\"0.8\" rx=\"2\"/>\n",
		leftX, topY, w, vlCellH, pal["global.stroke"])
	fmt.Fprintf(b, "  <text class=\"clbl\" x=\"%d\" y=\"%d\" fill=\"%s\">Patch Embed</text>\n",
		centerX, cy-6, pal["global.text"])
	hint := "conv → patch tokens"
	if len(visionGlobalWeights(v, positionEmbedKeys)) > 0 {
		hint = "conv + position grid"
	}
	fmt.Fprintf(b, "  <text class=\"tlbl\" x=\"%d\" y=\"%d\">%s</text>\n", centerX, cy+7, hint)

	// "IMAGE PATCHES" label + down-arrow into the cell.
	labelY := topY - 16
	fmt.Fprintf(b, "  <text class=\"io\" x=\"%d\" y=\"%d\">IMAGE PATCHES</text>\n", centerX, labelY)
	fmt.Fprintf(b, "  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"%s\"/>\n",
		centerX, labelY+5, centerX, topY, pal["ui.spine"], vlTrunkW)
	emitVisionArrowhead(b, centerX, topY, "down")
}

// vlExitCellH is the height of the post-encoder exit box. Matches the base
// diagram's output-cell proportions while staying compact enough to fit the
// headroom band between the title and the last encoder layer.
const vlExitCellH = 40

// emitVisionExit draws the post-encoder → decoder pipeline as a filled exit box
// (mirroring the base layers diagram's output box) with three sub-cells:
// Spatial Merge | Post Norm | Splice. The trunk rises out of the last encoder
// layer's row-out, routes up-and-left into the box's left edge, and an
// inverted-U "IMAGE TOKENS" exit decoration leaves the box top-right — mirroring
// emitOutputCell + emitOutputTokenExit. Drawn with this renderer's own vl_
// gradients (the shared emitGlobalCellRect/emitGlobalSubRect hardcode the base
// diagram's zm_ ids, which are undefined here).
//
// Data-aware: the post-encoder norm is a global post_ln for Qwen but per-layer/
// projector for Gemma, so the "Post Norm" cell is kept either way (it is the
// post-encoder normalization conceptually) without asserting a specific tensor.
// The projector type (previously surfaced in the bare-text label) is folded in
// as a sub-label under the Splice cell.
func emitVisionExit(b *strings.Builder, xOff, cy, boxBottomY int, def *arch.ArchDef) {
	// Box geometry: spans from the right column's block-left to its row-out, so
	// it sits over the encoder rows below it; ends well clear of the right-gutter
	// legend. Placed in the headroom band above the last encoder layer.
	leftX := xOff + vlColLeftX
	boxW := vlRowOutX - vlColLeftX
	rightX := leftX + boxW
	// Align the box bottom with the Patch-Embed entry cell's bottom so the two
	// end-cap boxes sit on the same top band.
	topY := boxBottomY - vlExitCellH
	midY := topY + vlExitCellH/2

	// Spine: leave the last layer's row-out, rise, make a hard left in the band
	// just below the box, then come up the column's left rail and into the box's
	// left-edge input arrow — wrapping around the box rather than running behind it.
	rowOutX := xOff + vlRowOutX
	leftRailX := xOff + vlTrunkInX
	belowBoxY := boxBottomY + vlUturnGap
	fmt.Fprintf(b, "  <path d=\"M %d,%d L %d,%d L %d,%d L %d,%d L %d,%d\" fill=\"none\" stroke=\"%s\" stroke-width=\"%s\" stroke-linejoin=\"miter\"/>\n",
		rowOutX, cy, rowOutX, belowBoxY, leftRailX, belowBoxY, leftRailX, midY, leftX, midY, pal["ui.spine"], vlTrunkW)
	emitVisionArrowhead(b, leftX, midY, "right")

	// Box rect — vl_ global gradient + global stroke (matches the base output box).
	fmt.Fprintf(b, "  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"url(#vl_global)\" stroke=\"%s\" stroke-width=\"0.8\" rx=\"2\"/>\n",
		leftX, topY, boxW, vlExitCellH, pal["global.stroke"])

	// Three evenly-split sub-cells in pipeline order, in the red/global family.
	subW := (boxW - 4) / 3
	subY := topY + vlCellPad
	subH := vlExitCellH - 2*vlCellPad
	x0 := leftX + 2
	emitVisionExitSubCell(b, x0, subY, subW-1, subH, "Spatial Merge")
	emitVisionExitSubCell(b, x0+subW, subY, subW-1, subH, "Post Norm")
	emitVisionExitSubCell(b, x0+2*subW, subY, subW-1, subH, "Splice")

	// Projector type as a sub-label under the Splice cell.
	if def.Projector != nil && def.Projector.Type != "" {
		spliceCX := x0 + 2*subW + (subW-1)/2
		fmt.Fprintf(b, "  <text x=\"%d\" y=\"%d\" text-anchor=\"middle\" font-size=\"7\" font-style=\"italic\" fill=\"%s\">%s</text>\n",
			spliceCX, topY+vlExitCellH+8, pal["ui.text_sec"], xmlEsc(formatBuilderName(def.Projector.Type)))
	}

	// Inverted-U exit decoration leaving the box top-right with an "IMAGE TOKENS"
	// label — the image-token stream entering the decoder (mirrors "TOKEN OUT").
	stubX := rightX + 8
	clearY := topY - 14
	textRightX := leftX + 2*boxW/3
	fmt.Fprintf(b, "  <path d=\"M %d,%d L %d,%d L %d,%d L %d,%d\" fill=\"none\" stroke=\"%s\" stroke-width=\"%s\" stroke-linejoin=\"miter\"/>\n",
		rightX, midY, stubX, midY, stubX, clearY, textRightX, clearY, pal["ui.spine"], vlTrunkW)
	emitVisionArrowhead(b, textRightX, clearY, "left")
	fmt.Fprintf(b, "  <text class=\"io\" style=\"text-anchor:end\" x=\"%d\" y=\"%d\">IMAGE TOKENS</text>\n",
		textRightX-2, clearY)
}

// emitVisionExitSubCell draws one labeled sub-rect inside the exit box, in the
// red/global family (solid global fill, global stroke) — matching the base
// diagram's output sub-cells. Long labels are centered at the global text color.
func emitVisionExitSubCell(b *strings.Builder, x, y, w, h int, label string) {
	fmt.Fprintf(b, "  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"%s\" stroke=\"%s\" stroke-width=\"0.5\" rx=\"1\"/>\n",
		x, y, w, h, pal["global.fill"], pal["global.stroke"])
	fmt.Fprintf(b, "  <text x=\"%d\" y=\"%d\" text-anchor=\"middle\" dominant-baseline=\"middle\" font-size=\"7\" font-weight=\"600\" fill=\"%s\">%s</text>\n",
		x+w/2, y+h/2, pal["global.text"], label)
}

// --- legend ---

func emitVisionLayersLegend(b *strings.Builder, x, y, w, h int, v *arch.VisionDef) {
	fmt.Fprintf(b, "  <text style=\"font-size:9px;font-weight:bold;fill:%s;font-family:'Courier New',monospace\" x=\"%d\" y=\"%d\">Legend</text>\n",
		pal["ui.text_head"], x+4, y-5)
	fmt.Fprintf(b, "  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"%s\" stroke=\"%s\" stroke-width=\"0.8\" rx=\"2\"/>\n",
		x, y, w, h, pal["ui.box_bg"], pal["ui.box_border"])
	ly := y + 10
	item := func(fill, stroke, label string) {
		fmt.Fprintf(b, "  <rect x=\"%d\" y=\"%d\" width=\"10\" height=\"10\" fill=\"%s\" stroke=\"%s\" stroke-width=\"0.8\" rx=\"1\"/>\n",
			x+4, ly, fill, stroke)
		fmt.Fprintf(b, "  <text class=\"ltxt\" x=\"%d\" y=\"%d\">%s</text>\n", x+4+14, ly+5, label)
		ly += 18
	}
	item("url(#vl_full_attention)", pal["full_attention.stroke"], "attention")
	item("url(#vl_ffn)", pal["ffn.stroke"], "FFN")
	item("url(#vl_global)", pal["global.stroke"], "patch embed")
	item(pal["norm.fill"], pal["norm.stroke"], normTypeLabel(v.NormType))
	// Residual ⊕ marker.
	cy := ly + 5
	fmt.Fprintf(b, "  <circle cx=\"%d\" cy=\"%d\" r=\"5\" fill=\"%s\"/>\n", x+9, cy, pal["ui.spine"])
	fmt.Fprintf(b, "  <text x=\"%d\" y=\"%d\" text-anchor=\"middle\" dominant-baseline=\"central\" font-size=\"9\" font-weight=\"bold\" fill=\"%s\">+</text>\n",
		x+9, cy, pal["ui.text_body"])
	fmt.Fprintf(b, "  <text class=\"ltxt\" x=\"%d\" y=\"%d\">residual add</text>\n", x+4+14, cy)
}
