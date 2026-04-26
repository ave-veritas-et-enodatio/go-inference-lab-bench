package archdiagram

import (
	"fmt"
	"io"
	"slices"
	"sort"
	"strconv"
	"strings"

	"inference-lab-bench/internal/inference/arch"
)

// symbolDef holds a pre-rendered SVG symbol body keyed by its <symbol> id.
type symbolDef struct {
	id  string
	svg string // rendered content for the <symbol> body (without outer <symbol> tags)
}

// layerEntry records which modules (block and/or FFN) were found for a single layer.
type layerEntry struct {
	blockID, ffnID   int
	hasBlock, hasFfn bool
}

// layerCtx bundles parsed module state + symbol tables shared across helpers.
type layerCtx struct {
	def          *arch.ArchDef
	mm           *arch.ModuleMap
	moduleByID   map[int]*arch.Module
	blockSymbols map[string]symbolDef
	ffnSymbols   map[string]symbolDef
}

// --- Layout constants ---
//
// Per-row layout: block cell and FFN cell sit on a horizontal trunk. Block
// and FFN write into the trunk via an inline ⊕ immediately to the right of
// each; contributions L-bend from the cell's bottom-right up into the ⊕.
//
// Macro layout: 2-column "U" snake — left column descends, right column
// ascends, column-bottom hop links the two. Per-row geometry is decoupled
// from row placement so the macro layout can change without touching the
// per-row code.
const (
	cellX1     = 22                            // block cell absolute left edge
	blockCellW = 110
	cellGap    = 34                            // gap between block and FFN cells (also drives ⊕ breathing room)
	cellX2     = cellX1 + blockCellW + cellGap // FFN cell absolute left edge
	ffnCellW   = 98
	cellH      = 40
	cellPad    = 3
	subRowH    = 10
	subRowGap  = 3

	// Block symbol: left norm slot offset (origin = block cell top-left).
	blkNormX = 2

	// FFN symbol: relative slot positions (origin = FFN cell top-left).
	ffnNormW  = 12
	ffnNormXr = 2                        // left norm slot
	ffnNDivr  = ffnNormXr + ffnNormW + 1 // divider after norm
	ffnGUXr   = ffnNDivr + 2             // gate/up column start
	ffnGUW    = 52
	ffnGUDivr = ffnGUXr + ffnGUW + 1 // divider after gate/up
	ffnDownXr = ffnGUDivr + 2        // down projection slot
	ffnDownW  = 22

	// Post-FFN trunk: distance from FFN right edge to the down-bend (start
	// of the U-turn). Sized to match cellGap so block and FFN sides feel
	// symmetric and the L-bend contribution has clear space.
	ffnPostTrunkW = cellGap

	// Trunk geometry. The snake enters each row at rowTrunkInX (absolute),
	// threads horizontally through the block + FFN with two ⊕ markers, then
	// exits at rowTrunkOutX where the per-row U-turn begins.
	rowTrunkInX  = 8
	rowTrunkOutX = cellX2 + ffnCellW + ffnPostTrunkW

	// Block/FFN ⊕ x-positions in absolute coords — midpoints of their trunk
	// segments so the L-bend contribution has equal breathing room each side.
	blkAbsCX = cellX1 + blockCellW + cellGap/2
	ffnAbsCX = cellX2 + ffnCellW + ffnPostTrunkW/2

	// Per-row U-turn vertical extent. Total row pitch = cellH + uturnGap.
	uturnGap = 14
	pitch    = cellH + uturnGap

	// Right-column offset and content geometry for the 2-column U layout.
	colGap         = 32
	rightColOffset = rowTrunkOutX + colGap

	// Composite cell widths reused by multiple emitters.
	outputCellW        = blockCellW + cellGap + ffnCellW         // span of the output (and buffer-when-diffusion) cell
	columnCenterOffset = (rowTrunkInX + rowTrunkOutX) / 2        // x-offset of a column's horizontal center (for centered top-of-column cells)

	// Trunk thickness + ⊕ radius.
	trunkW = "2.5"
	circR  = 5

	// Right-gutter (legend + summary) sizing.
	statsBoxH = 60
	legPad    = 4
)

// tensorLabels maps canonical weight/param names to the short display label.
var tensorLabels = map[string]string{
	arch.WeightAttnNorm:     "norm",
	arch.WeightAttnQ:        "Q",
	arch.WeightAttnK:        "K",
	arch.WeightAttnV:        "V",
	arch.WeightAttnOutput:   "out",
	arch.WeightAttnQNorm:    "Qn",
	arch.WeightAttnKNorm:    "Kn",
	arch.WeightRoPEFreqs:    "rope",
	arch.WeightRoPE:         "rope",
	arch.WeightPostAttnNorm: "norm",
	arch.WeightFFNNorm:      "norm",
	arch.WeightFFNGate:      "G",
	arch.WeightFFNUp:        "U",
	arch.WeightFFNDown:      "D",
	arch.WeightFFNGateExps:  "Gx",
	arch.WeightFFNUpExps:    "Ux",
	arch.WeightFFNDownExps:  "Dx",
	arch.WeightFFNGateShexp: "Gs",
	arch.WeightFFNUpShexp:   "Us",
	arch.WeightFFNDownShexp: "Ds",
	arch.WeightOutputNorm:   "norm",
	arch.WeightSSMNorm:      "norm",
	arch.WeightTokenEmbd:    "embd",
	arch.WeightOutput:       "lm_head",
}

func isNormWeight(name string) bool {
	return name == arch.WeightAttnNorm ||
		name == arch.WeightPostAttnNorm ||
		name == arch.WeightOutputNorm ||
		name == arch.WeightSSMNorm ||
		name == arch.WeightFFNNorm
}

// derivedLabel returns the short display label for a weight name. Falls back
// to the trailing underscore-separated segment, truncated to 5 chars.
func derivedLabel(shortName string) string {
	if lbl, ok := tensorLabels[shortName]; ok {
		return lbl
	}
	parts := strings.Split(shortName, "_")
	last := parts[len(parts)-1]
	if len(last) > 5 {
		last = last[:5]
	}
	return last
}

// ffnSymKey maps an FFN module to its symbol key: FFNSymDense or FFNSymMoE.
func ffnSymKey(m *arch.Module) string {
	if m.FFNExpertRouted {
		return arch.FFNSymMoE
	}
	return arch.FFNSymDense
}

// emitCellDivider draws a thin vertical divider line spanning a full cell
// height, used inside block/FFN symbols to separate logical sub-rect columns.
func emitCellDivider(sb *strings.Builder, x int) {
	fmt.Fprintf(sb, "  <line x1=\"%d\" y1=\"0\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"0.7\"/>\n",
		x, x, cellH, pal["ui.divider"])
}

// emitGlobalCellRect draws the outer rounded rect for a top-of-column cell
// (embd, output, diffusion buffer) — global gradient fill, global stroke.
func emitGlobalCellRect(sb *strings.Builder, x, y, w, h int) {
	fmt.Fprintf(sb, "  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"url(#zm_global)\" stroke=\"%s\" stroke-width=\"0.8\" rx=\"2\"/>\n",
		x, y, w, h, pal["global.stroke"])
}

// emitGlobalSubRect draws a labeled sub-rect inside a global cell (output
// norm, prompt, response). Solid global fill, 0.5-stroke. Label is centered.
func emitGlobalSubRect(sb *strings.Builder, x, y, w, h int, label string) {
	fmt.Fprintf(sb, "  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"%s\" stroke=\"%s\" stroke-width=\"0.5\" rx=\"1\"/>\n",
		x, y, w, h, pal["global.fill"], pal["global.stroke"])
	if label != "" {
		fmt.Fprintf(sb, "  <text class=\"tlbl\" x=\"%d\" y=\"%d\">%s</text>\n",
			x+w/2, y+h/2, label)
	}
}

// drawBaseRect renders one tensor sub-rect + its derived label.
func drawBaseRect(sb *strings.Builder, x, y, w, h int, blockType, shortName string) {
	prefix := palPrefix(blockType)
	if isNormWeight(shortName) {
		prefix = "norm"
	}
	fmt.Fprintf(sb, "  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"%s\" stroke=\"%s\" stroke-width=\"0.5\" rx=\"1\"/>\n",
		x, y, w, h, pal[prefix+".fill"], pal[prefix+".stroke"])
	lbl := derivedLabel(shortName)
	if lbl == "" || w < 8 {
		return
	}
	cx, cy := x+w/2, y+h/2
	if h > int(float64(w)*2.0) {
		fmt.Fprintf(sb, "  <text class=\"tlbl\" x=\"%d\" y=\"%d\" transform=\"rotate(-90,%d,%d)\">%s</text>\n",
			cx, cy, cx, cy, lbl)
	} else {
		fmt.Fprintf(sb, "  <text class=\"tlbl\" x=\"%d\" y=\"%d\">%s</text>\n", cx, cy, lbl)
	}
}

// parseLayers walks mm.Modules, collecting per-layer block/FFN module IDs.
func parseLayers(mm *arch.ModuleMap) (map[int]*layerEntry, map[int]*arch.Module, []int) {
	layerMap := make(map[int]*layerEntry)

	for i := range mm.Modules {
		m := &mm.Modules[i]
		if m.Name == arch.ModuleGlobal {
			continue
		}
		if after, ok := strings.CutPrefix(m.Name, arch.PrefixBlock); ok {
			if l, err := strconv.Atoi(after); err == nil {
				if layerMap[l] == nil {
					layerMap[l] = &layerEntry{}
				}
				layerMap[l].blockID = m.ID
				layerMap[l].hasBlock = true
			}
		} else if after, ok := strings.CutPrefix(m.Name, arch.PrefixFFN); ok {
			if l, err := strconv.Atoi(after); err == nil {
				if layerMap[l] == nil {
					layerMap[l] = &layerEntry{}
				}
				layerMap[l].ffnID = m.ID
				layerMap[l].hasFfn = true
			}
		}
	}

	var layerIndices []int
	for l := range layerMap {
		layerIndices = append(layerIndices, l)
	}
	sort.Ints(layerIndices)

	moduleByID := make(map[int]*arch.Module, len(mm.Modules))
	for i := range mm.Modules {
		moduleByID[mm.Modules[i].ID] = &mm.Modules[i]
	}

	return layerMap, moduleByID, layerIndices
}

// computeStats totals module, tensor, and weight counts across the module map.
func computeStats(mm *arch.ModuleMap) (nTotal, totalTensors, totalWeights int) {
	nTotal = len(mm.Modules)
	for _, m := range mm.Modules {
		totalTensors += len(m.Weights) + len(m.Params)
		totalWeights += len(m.Weights)
	}
	return
}

// buildBlockSymbol renders the SVG body for one block type and collects slot
// geometry. The symbol origin is the cell top-left (0,0). The block symbol
// renders only the cell content; trunk segments and the inline ⊕ are emitted
// separately by the per-row chrome path.
func buildBlockSymbol(blockType string, m *arch.Module) symbolDef {
	id := "blk-" + blockType
	var sb strings.Builder
	pp := palPrefix(blockType)
	normH := cellH - 2*cellPad
	normY := cellPad

	fmt.Fprintf(&sb, "  <rect x=\"0\" y=\"0\" width=\"%d\" height=\"%d\" fill=\"url(#zm_%s)\" stroke=\"%s\" stroke-width=\"0.8\" rx=\"2\"/>\n",
		blockCellW, cellH, pp, pal[pp+".stroke"])

	if arch.IsAttentionModule(m) {
		// Classify weights into data-flow columns:
		//   norm | Q,K,V | Qn,Kn,rope | out | post_norm
		coreSet := map[string]bool{arch.WeightAttnNorm: true, arch.WeightAttnQ: true, arch.WeightAttnK: true, arch.WeightAttnV: true, arch.WeightAttnOutput: true, arch.WeightPostAttnNorm: true}
		hasPostNorm := false
		var extras []string
		for _, w := range m.Weights {
			if w == arch.WeightPostAttnNorm {
				hasPostNorm = true
			} else if !coreSet[w] {
				extras = append(extras, w)
			}
		}
		hasRoPE := slices.Contains(extras, arch.WeightRoPEFreqs)
		if !hasRoPE {
			extras = append(extras, arch.WeightRoPE)
		}
		sort.Strings(extras)

		const divW = 3
		inputNormW := 12
		nDividers := 2
		if len(extras) > 0 {
			nDividers++
		}
		if hasPostNorm {
			nDividers++
		}
		avail := blockCellW - blkNormX - inputNormW - 2 - nDividers*divW
		qkvW := avail * 40 / 100
		outW := avail * 20 / 100
		postNormW := 0
		extraW := 0
		if hasPostNorm {
			postNormW = avail * 12 / 100
		}
		if len(extras) > 0 {
			extraW = avail - qkvW - outW - postNormW
		} else if hasPostNorm {
			postNormW = avail * 15 / 100
			outW = avail - qkvW - postNormW
		} else {
			outW = avail - qkvW
		}

		x := blkNormX
		drawBaseRect(&sb, x, normY, inputNormW, normH, blockType, arch.WeightAttnNorm)
		x += inputNormW + 1
		emitCellDivider(&sb, x)
		x += 2

		total3H := 3*subRowH + 2*subRowGap
		qStartY := (cellH - total3H) / 2
		for qi, qn := range []string{arch.WeightAttnQ, arch.WeightAttnK, arch.WeightAttnV} {
			qy := qStartY + qi*(subRowH+subRowGap)
			drawBaseRect(&sb, x, qy, qkvW, subRowH, blockType, qn)
		}
		x += qkvW + 1
		emitCellDivider(&sb, x)
		x += 2

		if len(extras) > 0 {
			nExtras := len(extras)
			totalExH := nExtras*subRowH + (nExtras-1)*subRowGap
			exStartY := (cellH - totalExH) / 2
			for ei, en := range extras {
				ey := exStartY + ei*(subRowH+subRowGap)
				drawBaseRect(&sb, x, ey, extraW, subRowH, blockType, en)
			}
			x += extraW + 1
			emitCellDivider(&sb, x)
			x += 2
		}

		drawBaseRect(&sb, x, normY, outW, normH, blockType, arch.WeightAttnOutput)
		x += outW + 1

		if hasPostNorm {
			emitCellDivider(&sb, x)
			x += 2
			drawBaseRect(&sb, x, normY, postNormW, normH, blockType, arch.WeightPostAttnNorm)
		}

	} else {
		// Generic: attn_norm as left norm slot, remaining weights as equal sub-rects.
		var normWt string
		remaining := make([]string, 0, len(m.Weights))
		for _, w := range m.Weights {
			if w == arch.WeightAttnNorm {
				normWt = w
			} else {
				remaining = append(remaining, w)
			}
		}
		sort.Strings(remaining)
		startX := 2
		if normWt != "" {
			drawBaseRect(&sb, startX, normY, 12, normH, blockType, normWt)
			divX := startX + 12 + 1
			emitCellDivider(&sb, divX)
			startX = divX + 2
		}
		if n := len(remaining); n > 0 {
			innerW := blockCellW - 2 - startX
			subW := max(innerW/n, 1)
			for wi, wn := range remaining {
				wx := startX + wi*subW
				drawBaseRect(&sb, wx, normY, subW-1, normH, blockType, wn)
			}
		}
	}

	return symbolDef{id: id, svg: sb.String()}
}

// buildFFNSymbol renders the SVG body for one FFN type and collects slot geometry.
// The symbol origin is the FFN cell top-left (0,0). Trunk/⊕ are drawn outside.
func buildFFNSymbol(sk string, m *arch.Module) symbolDef {
	id := "ffn-" + sk
	isMoE := sk == arch.FFNSymMoE
	var sb strings.Builder
	normH := cellH - 2*cellPad
	normY := cellPad

	fmt.Fprintf(&sb, "  <rect x=\"0\" y=\"0\" width=\"%d\" height=\"%d\" fill=\"url(#zm_ffn)\" stroke=\"%s\" stroke-width=\"0.8\" rx=\"2\"/>\n",
		ffnCellW, cellH, pal["ffn.stroke"])

	// Any norm weight present in the FFN module IS the FFN's pre-norm by
	// construction (the module map routes the FFN's pre-norm here regardless
	// of its GGUF tensor name — Llama uses "ffn_norm", Qwen3.5 uses
	// "post_attention_norm" mapped from the ffn_norm logical key, etc.).
	var normWt string
	for _, w := range m.Weights {
		if isNormWeight(w) {
			normWt = w
			break
		}
	}
	if normWt != "" {
		drawBaseRect(&sb, ffnNormXr, normY, ffnNormW, normH, arch.TypeFFN, normWt)
		emitCellDivider(&sb, ffnNDivr)
	}

	if !isMoE {
		total2H := 2*subRowH + 1*subRowGap
		guStartY := (cellH - total2H) / 2
		for gi, gn := range []string{arch.WeightFFNGate, arch.WeightFFNUp} {
			gy := guStartY + gi*(subRowH+subRowGap)
			drawBaseRect(&sb, ffnGUXr, gy, ffnGUW, subRowH, arch.TypeFFN, gn)
		}
		drawBaseRect(&sb, ffnDownXr, normY, ffnDownW, normH, arch.TypeFFN, arch.WeightFFNDown)
		emitCellDivider(&sb, ffnGUDivr)
	} else {
		expertAreaX := ffnGUXr
		expertAreaW := ffnCellW - expertAreaX - 2

		var expertWts, sharedWts []string
		for _, w := range m.Weights {
			switch {
			case isNormWeight(w):
				// already handled by the norm-finder above
			case arch.MoEExpertWeights[w]:
				expertWts = append(expertWts, w)
			default:
				sharedWts = append(sharedWts, w)
			}
		}
		sort.Strings(expertWts)
		sort.Strings(sharedWts)

		if ne := len(expertWts); ne > 0 {
			bs := max(expertAreaW*10/(ne*12), 6)
			ld := max((bs*2+9)/10, 1)
			gap := ld
			totalBoxW := ne*bs + (ne-1)*gap + ld
			startX := max(expertAreaX+(expertAreaW-totalBoxW)/2, expertAreaX)
			bY := normY + (normH-bs-ld)/2 + ld

			for xi, wn := range expertWts {
				bX := startX + xi*(bs+gap)
				fmt.Fprintf(&sb, "  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"%s\" stroke=\"%s\" stroke-width=\"0.5\"/>\n",
					bX, bY, bs, bs, pal["ffn.fill"], pal["ffn.stroke"])
				fmt.Fprintf(&sb, "  <polygon points=\"%d,%d %d,%d %d,%d %d,%d\" fill=\"%s\" stroke=\"%s\" stroke-width=\"0.5\"/>\n",
					bX, bY, bX+ld, bY-ld, bX+bs+ld, bY-ld, bX+bs, bY,
					pal["ffn_moe.lid"], pal["ffn.stroke"])
				lbl := derivedLabel(wn)
				if lbl != "" && bs >= 8 {
					fmt.Fprintf(&sb, "  <text class=\"tlbl\" x=\"%d\" y=\"%d\">%s</text>\n",
						bX+bs/2, bY+bs/2, lbl)
				}
			}
		}
		if ns := len(sharedWts); ns > 0 {
			ne := len(expertWts)
			avail := expertAreaW
			if ne > 0 {
				bs := max(expertAreaW*10/(ne*12), 6)
				ld := max((bs*2+9)/10, 1)
				avail -= ne*bs + (ne-1)*ld + ld + 2
			}
			subW := avail / ns
			if subW >= 6 {
				sharedX := expertAreaX + (expertAreaW - avail)
				for ri, wn := range sharedWts {
					wx := sharedX + ri*subW
					drawBaseRect(&sb, wx, normY, subW-1, normH, arch.TypeFFNMoE, wn)
				}
			}
		}
	}

	return symbolDef{id: id, svg: sb.String()}
}

// buildSymbols populates ctx's block/FFN symbol tables for every unique block
// type and FFN kind encountered in mm.Modules.
func buildSymbols(ctx *layerCtx) (blockTypes, ffnKeys []string) {
	for i := range ctx.mm.Modules {
		m := &ctx.mm.Modules[i]
		if strings.HasPrefix(m.Name, arch.PrefixBlock) && m.BlockName != "" {
			if _, seen := ctx.blockSymbols[m.BlockName]; !seen {
				ctx.blockSymbols[m.BlockName] = buildBlockSymbol(m.BlockName, m)
			}
		} else if strings.HasPrefix(m.Name, arch.PrefixFFN) {
			sk := ffnSymKey(m)
			if _, seen := ctx.ffnSymbols[sk]; !seen {
				ctx.ffnSymbols[sk] = buildFFNSymbol(sk, m)
			}
		}
	}
	for bt := range ctx.blockSymbols {
		blockTypes = append(blockTypes, bt)
	}
	sort.Strings(blockTypes)
	for fk := range ctx.ffnSymbols {
		ffnKeys = append(ffnKeys, fk)
	}
	sort.Strings(ffnKeys)
	return
}

// Arrowhead geometry. All arrowheads in the diagram share these dimensions,
// sized at 125% of the previous largest arrow (which was 6×6). Uniform sizing
// makes flow direction read consistently across the snake, U-turns, IN feed,
// and the diffusion loop-back.
const (
	arrowLen      = 8 // apex-to-base distance along arrow axis
	arrowHalfBase = 4 // perpendicular extent on each side of centerline
)

// emitArrowhead writes a triangular arrowhead with apex at (apexX, apexY).
// dir is "left", "right", "up", or "down" — direction the arrow points.
func emitArrowhead(b *strings.Builder, apexX, apexY int, dir string) {
	var bx1, by1, bx2, by2 int
	switch dir {
	case "right":
		bx1, by1 = apexX-arrowLen, apexY-arrowHalfBase
		bx2, by2 = apexX-arrowLen, apexY+arrowHalfBase
	case "left":
		bx1, by1 = apexX+arrowLen, apexY-arrowHalfBase
		bx2, by2 = apexX+arrowLen, apexY+arrowHalfBase
	case "down":
		bx1, by1 = apexX-arrowHalfBase, apexY-arrowLen
		bx2, by2 = apexX+arrowHalfBase, apexY-arrowLen
	case "up":
		bx1, by1 = apexX-arrowHalfBase, apexY+arrowLen
		bx2, by2 = apexX+arrowHalfBase, apexY+arrowLen
	}
	fmt.Fprintf(b, "  <polygon points=\"%d,%d %d,%d %d,%d\" fill=\"%s\"/>\n",
		bx1, by1, apexX, apexY, bx2, by2, pal["ui.spine"])
}

// emitTrunkSegment writes a trunk-thick horizontal line from (x1,y) to (x2,y)
// with an optional rightward arrowhead. Three modes:
//   - "none"  — line only, square linecap (extends ~half-stroke past x2 to fill
//     the corner with a separate path turning down).
//   - "cell"  — line + arrow apex AT x2 (used when the trunk feeds into a cell;
//     the cell's left edge sits at x2 so the arrow tip touches it).
//   - "turn"  — line + arrow apex 2px PAST x2, plus the line itself extending
//     to that point. This overlaps the right edge of the U-turn's down-stroke
//     (which is centered on x2 with stroke-width 2.5), eliminating the notch
//     that would otherwise appear at the corner.
func emitTrunkSegment(b *strings.Builder, x1, x2, y int, mode string) {
	switch mode {
	case "none":
		fmt.Fprintf(b, "  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"%s\" stroke-linecap=\"square\"/>\n",
			x1, y, x2, y, pal["ui.spine"], trunkW)
	case "turn":
		ax := x2 + 2 // arrow apex past x2 to overlap the U-turn's right edge
		fmt.Fprintf(b, "  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"%s\"/>\n",
			x1, y, ax, y, pal["ui.spine"], trunkW)
		emitArrowhead(b, ax, y, "right")
	default: // "cell"
		fmt.Fprintf(b, "  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"%s\"/>\n",
			x1, y, x2, y, pal["ui.spine"], trunkW)
		emitArrowhead(b, x2, y, "right")
	}
}

// emitInlineAdder writes the ⊕ glyph at (cx, cy) as three layered elements,
// in this order: solid gray disk (covers the trunk underneath), contribution-
// color ring on top, then the + glyph. Caller draws this AFTER the trunk and
// the contribution L-bend so the ⊕ sits cleanly on top of both.
func emitInlineAdder(b *strings.Builder, cx, cy int, strokeColor string) {
	// Gray background disk — opaque, covers the trunk.
	fmt.Fprintf(b, "  <circle cx=\"%d\" cy=\"%d\" r=\"%d\" fill=\"%s\" stroke=\"none\"/>\n",
		cx, cy, circR, pal["ui.spine"])
	// Contribution-color ring.
	fmt.Fprintf(b, "  <circle cx=\"%d\" cy=\"%d\" r=\"%d\" fill=\"none\" stroke=\"%s\" stroke-width=\"1.2\"/>\n",
		cx, cy, circR, strokeColor)
	// + glyph.
	fmt.Fprintf(b, "  <text x=\"%d\" y=\"%d\" text-anchor=\"middle\" dominant-baseline=\"central\" font-size=\"9\" font-weight=\"bold\" fill=\"%s\">+</text>\n",
		cx, cy, strokeColor)
}

// emitContributionLBend draws the thin L-bend from a cell's bottom-right
// corner up into the inline ⊕ above it. cellRightX/cellBottomY are the
// cell's right and bottom-quarter Y; adderX/adderY is the ⊕ center.
func emitContributionLBend(b *strings.Builder, cellRightX, exitY, adderX, adderY int, color string) {
	// Path: right from cell exit, then up to the ⊕'s bottom edge.
	fmt.Fprintf(b, "  <path d=\"M %d,%d L %d,%d L %d,%d\" fill=\"none\" stroke=\"%s\" stroke-width=\"1.0\"/>\n",
		cellRightX, exitY, adderX, exitY, adderX, adderY+circR, color)
}

// layerCellGeom captures the absolute x-coords and per-cell colors that
// emitLayerCells produces, for use by downstream trunk-wiring code (AR or
// diffusion). All x-coords are absolute canvas coords (xOff already added).
type layerCellGeom struct {
	rowY, trunkY            int
	blockLeftX, blockRightX int
	ffnLeftX, ffnRightX     int
	rowOutX                 int
	blockColor, ffnColor    string
	hasBlock, hasFfn        bool
}

// emitLayerCells renders the layer label and the block/FFN cell <use>
// instances for one layer row at (xOff, cy). It does NOT draw the residual
// trunk, ⊕ markers, contribution L-bends, or any inter-row wiring — that's
// the trunk renderer's job, which differs between the AR and diffusion
// generation modes.
//
// Returns geometry needed by the trunk renderer.
func emitLayerCells(ctx *layerCtx, b *strings.Builder, l int, lr *layerEntry, xOff, cy int) layerCellGeom {
	rowY := cy - cellH/2

	// Layer label on the left. text-anchor="end" (from .lbl), so x is the right
	// edge of the label. Place it a small fraction of a glyph width left of the
	// cell. y is the top of the glyph (text-before-edge), nudged down ~half the
	// digit height so the optical center sits below the original 10% line.
	const labelXGap = 1.5  // sub-pixel gap between digit right edge and cell left
	const labelYNudge = 4  // half digit cap height for 9px font
	labelX := float64(xOff+cellX1) - labelXGap
	labelY := rowY + cellH/10 + labelYNudge
	fmt.Fprintf(b, "  <text class=\"lbl\" x=\"%g\" y=\"%d\" dominant-baseline=\"text-before-edge\">%d</text>\n", labelX, labelY, l)

	g := layerCellGeom{
		rowY:        rowY,
		trunkY:      cy,
		blockLeftX:  xOff + cellX1,
		blockRightX: xOff + cellX1 + blockCellW,
		ffnLeftX:    xOff + cellX2,
		ffnRightX:   xOff + cellX2 + ffnCellW,
		rowOutX:     xOff + rowTrunkOutX,
		hasBlock:    lr.hasBlock,
		hasFfn:      lr.hasFfn,
	}

	if lr.hasBlock {
		m := ctx.moduleByID[lr.blockID]
		sym := ctx.blockSymbols[m.BlockName]
		g.blockColor = pal[palPrefix(m.BlockName)+".stroke"]
		fmt.Fprintf(b, "  <use href=\"#%s\" transform=\"translate(%d,%d)\"/>\n", sym.id, g.blockLeftX, rowY)
	} else {
		g.blockColor = pal["ui.spine"]
	}

	if lr.hasFfn {
		m := ctx.moduleByID[lr.ffnID]
		sk := ffnSymKey(m)
		sym := ctx.ffnSymbols[sk]
		g.ffnColor = pal["ffn.stroke"]
		fmt.Fprintf(b, "  <use href=\"#%s\" transform=\"translate(%d,%d)\"/>\n", sym.id, g.ffnLeftX, rowY)
	} else {
		g.ffnColor = pal["ui.spine"]
	}

	return g
}

// emitARLayerTrunkRow draws the autoregressive per-row trunk wiring around
// the cells already rendered by emitLayerCells: block→FFN trunk segment +
// arrowhead, block contribution L-bend, block ⊕, then FFN→row-out trunk +
// arrowhead, FFN contribution L-bend, FFN ⊕.
//
// Draw order per cell: trunk line → arrowhead → contribution L-bend → ⊕
// (gray disk + colored ring + plus). The ⊕ paints last so it sits cleanly
// over the trunk and the L-bend terminus.
//
// Trunk-in (from the IN feed or the previous row's U-turn) is NOT drawn
// here — it's part of the IN feed / U-turn path so the corner where vertical
// meets horizontal is one continuous path (no notch). The U-turn linking
// this row to the next is emitted by emitInterRowTurn.
func emitARLayerTrunkRow(b *strings.Builder, g layerCellGeom) {
	blkAdderX := (g.blockRightX + g.ffnLeftX) / 2
	ffnAdderX := (g.ffnRightX + g.rowOutX) / 2
	exitY := g.rowY + (cellH * 3 / 4)

	emitTrunkSegment(b, g.blockRightX, g.ffnLeftX, g.trunkY, "cell")
	if g.hasBlock {
		emitContributionLBend(b, g.blockRightX, exitY, blkAdderX, g.trunkY, g.blockColor)
	}
	emitInlineAdder(b, blkAdderX, g.trunkY, g.blockColor)

	emitTrunkSegment(b, g.ffnRightX, g.rowOutX, g.trunkY, "turn")
	if g.hasFfn {
		emitContributionLBend(b, g.ffnRightX, exitY, ffnAdderX, g.trunkY, g.ffnColor)
	}
	emitInlineAdder(b, ffnAdderX, g.trunkY, g.ffnColor)
}

// emitARLayerRow combines cell rendering and AR trunk wiring for one layer.
func emitARLayerRow(ctx *layerCtx, b *strings.Builder, l int, lr *layerEntry, xOff, cy int) {
	g := emitLayerCells(ctx, b, l, lr, xOff, cy)
	emitARLayerTrunkRow(b, g)
}

// emitColumnBottomHop draws the bottom-row connector from the last left-column
// layer's trunk-out straight across to the first right-column layer's block-
// left. When both columns' bottom rows sit at the same Y (the typical case),
// this is a single horizontal line — no detour below the rows.
//
// The trunk is unidirectional regardless of attention mask: residual-stream
// flow from layer N to layer N+1 is always forward through layer index. The
// bidirectional-attention property is indicated on the attention cell itself
// (via the K-block glyph), not by trunk arrows.
//
// The bottomY parameter is retained so the function tolerates a small Y offset
// between cyLeft and cyRight (rendered as a Z-shape with a midpoint horizontal),
// but the typical case routes straight across at cyLeft = cyRight = bottomY.
func emitColumnBottomHop(b *strings.Builder, leftXOff, rightXOff, cyLeft, cyRight, bottomY int) {
	xLeftOut := leftXOff + rowTrunkOutX
	xRightIn := rightXOff + cellX1

	if cyLeft == cyRight {
		// Straight horizontal across the column gap.
		fmt.Fprintf(b, "  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"%s\" stroke-linecap=\"square\"/>\n",
			xLeftOut, cyLeft, xRightIn, cyRight, pal["ui.spine"], trunkW)
	} else {
		// Z-shape: DOWN/UP to bottomY, ACROSS, UP/DOWN to right column's row.
		fmt.Fprintf(b, "  <path d=\"M %d,%d L %d,%d L %d,%d L %d,%d\" fill=\"none\" stroke=\"%s\" stroke-width=\"%s\" stroke-linejoin=\"miter\"/>\n",
			xLeftOut, cyLeft, xLeftOut, bottomY, xRightIn, bottomY, xRightIn, cyRight, pal["ui.spine"], trunkW)
	}

	// Forward arrowhead pointing RIGHT at the right-column block-left input.
	emitArrowhead(b, xRightIn, cyRight, "right")
}

// emitInterRowTurn draws the U-turn from one row's trunk-out (after FFN ⊕) to
// the next row's block-left edge. Single path with miter-joined corners so the
// down-across-down-right turns and the trunk-in to the next row share clean
// joins (no notch where a separate horizontal segment would meet a vertical).
//
// Works for both descending (cyB > cyA) and ascending (cyB < cyA) within-column
// turns: yMid is just the midpoint, and the path naturally flows DOWN-LEFT-DOWN
// or UP-LEFT-UP depending on the relative position of cyA and cyB.
func emitInterRowTurn(b *strings.Builder, xOff, cyA, cyB int) {
	yMid := (cyA + cyB) / 2
	xRight := xOff + rowTrunkOutX
	xLeft := xOff + rowTrunkInX
	xBlockLeft := xOff + cellX1

	// Single path, four segments: down → across → down → right-into-block.
	fmt.Fprintf(b, "  <path d=\"M %d,%d L %d,%d L %d,%d L %d,%d L %d,%d\" fill=\"none\" stroke=\"%s\" stroke-width=\"%s\" stroke-linejoin=\"miter\"/>\n",
		xRight, cyA, xRight, yMid, xLeft, yMid, xLeft, cyB, xBlockLeft, cyB, pal["ui.spine"], trunkW)

	// Forward arrowhead pointing right at the block-left input — flow direction
	// into the next layer's attention.
	emitArrowhead(b, xBlockLeft, cyB, "right")

	// Forward arrowhead pointing left, on the cross-segment at its center.
	xMid := (xLeft + xRight) / 2
	emitArrowhead(b, xMid, yMid, "left")
}

// emitInputCell draws just the embedding cell (global-pink outer rect + embd
// sub-rect) at the top of the left column, centered horizontally over the
// column. Returns the cell's geometry — needed by callers drawing the input
// label (AR) or the buffer↔embd connection (diffusion).
func emitInputCell(b *strings.Builder, xOff, cy int) (leftX, topY, cellW int) {
	rowY := cy - cellH/2
	cellW = blockCellW
	leftX = xOff + columnCenterOffset - cellW/2
	topY = rowY

	emitGlobalCellRect(b, leftX, rowY, cellW, cellH)
	drawBaseRect(b, leftX+2, rowY+cellPad, cellW-4, cellH-2*cellPad, "global", arch.WeightTokenEmbd)
	return
}

// emitInputTokenLabel draws the "TOKEN(s) IN" label and L-arrow above the
// embd cell — the AR-mode "first iteration / new prompt" entry decoration.
// In diffusion mode this is replaced by the response/mask buffer connection.
func emitInputTokenLabel(b *strings.Builder, embdLeftX, embdTopY, embdCellW int) {
	const labelText = "TOKEN(s) IN"
	const charW = 5     // approx font-size:8 monospace char advance (px)
	const labelGap = 18 // horizontal segment between label-left and corner
	labelY := embdTopY - 10
	textRightX := embdLeftX + embdCellW
	textLeftX := textRightX - len(labelText)*charW
	cornerX := textLeftX - labelGap

	// style="" overrides the .io class's text-anchor:middle.
	fmt.Fprintf(b, "  <text class=\"io\" style=\"text-anchor:end\" x=\"%d\" y=\"%d\">%s</text>\n",
		textRightX, labelY, labelText)
	fmt.Fprintf(b, "  <path d=\"M %d,%d L %d,%d L %d,%d\" fill=\"none\" stroke=\"%s\" stroke-width=\"%s\" stroke-linejoin=\"miter\"/>\n",
		textLeftX, labelY, cornerX, labelY, cornerX, embdTopY, pal["ui.spine"], trunkW)
	emitArrowhead(b, cornerX, embdTopY, "down")
}

// emitInputRow combines emitInputCell + emitInputTokenLabel for AR mode.
func emitInputRow(b *strings.Builder, xOff, cy int) {
	leftX, topY, cellW := emitInputCell(b, xOff, cy)
	emitInputTokenLabel(b, leftX, topY, cellW)
}

// emitOutputCell draws the output projection chain at the top of the right
// column: norm (output_norm weight) → lm_head (output weight) → sample
// (algorithmic, no weight). Trunk enters from the left at trunkY (drawn by
// this function as the entry segment). Returns geometry needed by callers
// drawing the exit decoration (AR token-OUT label, or diffusion loop-back).
//
// sampleLabel parameterizes the rightmost sub-rect's text — "[sample]" for
// AR, "[sample+remask]" for diffusion (where the sampled tokens get partial-
// unmasked according to a confidence schedule).
//
// tiedEmbed=true adds a small italic "(embd xposed)" annotation under the
// lm_head sub-rect — for models like Llama, Gemma, Qwen where the LM head
// matrix is the same tensor as the input embedding (transposed).
func emitOutputCell(b *strings.Builder, xOff, cy int, tiedEmbed bool, sampleLabel string) (leftX, rightX, topY, sampleCenterX int) {
	rowY := cy - cellH/2
	trunkY := cy
	leftX = xOff + cellX1
	rightX = leftX + outputCellW
	topY = rowY

	// Trunk into the cell from the left.
	emitTrunkSegment(b, xOff+rowTrunkInX, leftX, trunkY, "cell")

	// Cell rect.
	emitGlobalCellRect(b, leftX, rowY, outputCellW, cellH)

	// Three sub-rects in computational order: norm, lm_head, sample.
	subW := (outputCellW - 4) / 3
	subY := rowY + cellPad
	subH := cellH - 2*cellPad
	x0 := leftX + 2
	// output_norm — uses global (red) color since it's a top-level tensor of
	// the model (not a per-layer norm), and labeled in full to disambiguate
	// from per-layer "norm" sub-rects elsewhere.
	emitGlobalSubRect(b, x0, subY, subW-1, subH, "output norm")
	drawBaseRect(b, x0+subW, subY, subW-1, subH, "global", arch.WeightOutput)
	if tiedEmbed {
		lmHeadCX := x0 + subW + (subW-1)/2
		fmt.Fprintf(b, "  <text x=\"%d\" y=\"%d\" text-anchor=\"middle\" font-size=\"7\" font-style=\"italic\" fill=\"%s\">(embd xposed)</text>\n",
			lmHeadCX, rowY+cellH/2+9, pal["ui.text_sec"])
	}
	// sample — no box, parenthesized label so it visually reads as an
	// operation rather than a stored tensor (the other two sub-rects DO
	// represent stored tensors, so they get boxes).
	sx := x0 + 2*subW
	sampleCenterX = sx + (subW-1)/2
	fmt.Fprintf(b, "  <text class=\"tlbl\" x=\"%d\" y=\"%d\">%s</text>\n",
		sampleCenterX, subY+subH/2, sampleLabel)
	return
}

// emitOutputTokenExit draws the inverted-U exit trunk and "TOKEN OUT" label
// — the AR-mode "sampled token leaves the diagram" exit decoration. In
// diffusion mode this is replaced by a loop-back to the response/mask buffer.
func emitOutputTokenExit(b *strings.Builder, cellLeftX, cellRightX, cellRowY, trunkY int) {
	cellW := cellRightX - cellLeftX
	stubX := cellRightX + 8
	clearY := cellRowY - 14
	textRightX := cellLeftX + 2*cellW/3
	fmt.Fprintf(b, "  <path d=\"M %d,%d L %d,%d L %d,%d L %d,%d\" fill=\"none\" stroke=\"%s\" stroke-width=\"%s\" stroke-linejoin=\"miter\"/>\n",
		cellRightX, trunkY, stubX, trunkY, stubX, clearY, textRightX, clearY,
		pal["ui.spine"], trunkW)
	emitArrowhead(b, textRightX, clearY, "left")

	const labelArrowGap = 2 // ~1/4 monospace 8px char width
	fmt.Fprintf(b, "  <text class=\"io\" style=\"text-anchor:end\" x=\"%d\" y=\"%d\">TOKEN OUT</text>\n",
		textRightX-labelArrowGap, clearY)
}

// emitOutputRow combines emitOutputCell + emitOutputTokenExit for AR mode.
func emitOutputRow(b *strings.Builder, xOff, cy int, tiedEmbed bool) {
	leftX, rightX, topY, _ := emitOutputCell(b, xOff, cy, tiedEmbed, "[sample]")
	emitOutputTokenExit(b, leftX, rightX, topY, cy)
}

// emitDiffusionBufferCell draws the prompt/response/mask buffer that sits
// at the top of the diffusion diagram and circulates between iterations.
// Represents the persistent token-id state across forward passes; each
// iteration: buffer → embd → layers → output_norm → lm_head → sample+remask
// → buffer.
//
// Three sub-boxes inside the outer cell:
//   - prompt: solid global-fill, immutable across iterations (set at t=0,
//     never re-sampled)
//   - response: solid global-fill, fixed length, holds the committed sampled
//     tokens — the region that progressively fills in across iterations
//   - mask: dashed outline, positions still holding the MASK token, awaiting
//     sampling
func emitDiffusionBufferCell(b *strings.Builder, x, y, w int) {
	emitGlobalCellRect(b, x, y, w, cellH)

	subW := (w - 4) / 3
	subY := y + cellPad
	subH := cellH - 2*cellPad
	x0 := x + 2
	centerY := subY + subH/2

	// Prompt: immutable across iterations.
	emitGlobalSubRect(b, x0, subY, subW-1, subH, "prompt")

	// Response: committed sampled tokens, fixed length. Two-line label
	// "fixed-length / response" to convey the up-front length commitment that
	// distinguishes diffusion from AR.
	rx := x0 + subW
	rcx := rx + (subW-1)/2
	emitGlobalSubRect(b, rx, subY, subW-1, subH, "")
	fmt.Fprintf(b, "  <text class=\"tlbl\" x=\"%d\" y=\"%d\">fixed-length</text>\n",
		rcx, centerY-5)
	fmt.Fprintf(b, "  <text class=\"tlbl\" x=\"%d\" y=\"%d\">response</text>\n",
		rcx, centerY+4)

	// Mask: still-MASK positions, dashed outline, light fill (visually
	// distinct from the solid prompt/response).
	mx := x0 + 2*subW
	fmt.Fprintf(b, "  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"%s\" stroke=\"%s\" stroke-width=\"0.5\" stroke-dasharray=\"3,2\" rx=\"1\"/>\n",
		mx, subY, subW-1, subH, pal["ui.box_bg"], pal["global.stroke"])
	fmt.Fprintf(b, "  <text class=\"tlbl\" x=\"%d\" y=\"%d\">mask</text>\n",
		mx+(subW-1)/2, centerY)
}

// RenderLayersDiagram renders a per-layer module map as an SVG showing
// per-module tensor detail. The per-layer compute and residual-stream trunk
// are identical for AR and diffusion models — both are strictly feed-forward
// through the layer stack. Bidirectional attention (vs causal) is indicated
// by a per-cell glyph on the K weight; diffusion's iterative outer loop is
// indicated only by the subtitle text (a more honest visualization of the
// sequence/iteration/mask-state axes would need a different diagram).
//
// Layout: 2-column "U". Layers split into a left column (descending, indices
// 0..leftCount-1) and a right column (ascending, indices leftCount..N-1).
// Within each column, per-row U-turns connect adjacent layers; a column-bottom
// hop connects the last left layer to the first right layer; the global module
// sits at the top of the right column. The residual stream is the trunk
// threading horizontally through every block and FFN; each cell writes its
// contribution to the trunk via an inline ⊕ to its right.
func RenderLayersDiagram(def *arch.ArchDef, mm *arch.ModuleMap, w io.Writer, subTitle string) error {
	title := capitalizeASCII(def.Architecture.Name) + subTitle
	nonCausal := def.Architecture.NonCausal

	layerMap, moduleByID, layerIndices := parseLayers(mm)
	nLayers := len(layerIndices)
	nTotal, totalTensors, totalWeights := computeStats(mm)

	ctx := &layerCtx{
		def:          def,
		mm:           mm,
		moduleByID:   moduleByID,
		blockSymbols: make(map[string]symbolDef),
		ffnSymbols:   make(map[string]symbolDef),
	}
	blockTypes, ffnKeys := buildSymbols(ctx)

	// --- Column split ---
	// Left column gets the first half (rounded up); right column gets the rest.
	leftCount := (nLayers + 1) / 2
	rightCount := nLayers - leftCount
	leftLayers := layerIndices[:leftCount]
	rightLayers := layerIndices[leftCount:]

	// --- Vertical layout ---
	// Title band sized to clear the title text + the input/output row labels
	// above their cells. Top-of-column rows (input on left, output on right)
	// sit just below the title band, then the layer rows below those.
	titleBandH := 70
	if nonCausal {
		titleBandH = 86
	}
	// Top-of-column rows (embd on left, norm/lm_head/sample on right). Both
	// at the same Y so they read as a matched pair flanking the title.
	//
	// For diffusion, an additional response/mask buffer row sits ABOVE the
	// input/output rows, representing the persistent token-id state that
	// circulates between iterations. Embd and output rows shift down by one
	// pitch to make room.
	isDiffusion := def.Architecture.Generation == arch.GenerationDiffusion
	bufferRowCY := titleBandH + cellH/2 + 6
	inputRowCY := bufferRowCY
	if isDiffusion {
		inputRowCY = bufferRowCY + pitch
	}
	firstLayerCY := inputRowCY + pitch
	leftCY := func(i int) int { return firstLayerCY + i*pitch }
	bottomCY := leftCY(leftCount - 1) // last left layer (bottom of left column)
	// Right column ascends: rightCY(0) at bottom (= bottomCY), rightCY(j) above.
	rightCY := func(j int) int { return bottomCY - j*pitch }
	var lastRightCY int
	if rightCount > 0 {
		lastRightCY = rightCY(rightCount - 1)
	} else {
		lastRightCY = bottomCY
	}
	// Output row sits at the same Y as the input row (top of right column).
	outputRowCY := inputRowCY
	// Bottom hop Y. With cyLeft == cyRight (typical case) the hop is straight
	// across at bottomCY; bottomHopY is only used as the Z-shape midpoint for
	// the rare offset case.
	bottomHopY := bottomCY + cellH/2 + uturnGap/2

	// --- Canvas dimensions ---
	contentRightEdge := rightColOffset + rowTrunkOutX + 16
	legBoxX := contentRightEdge + 8
	legBoxW := 180
	canvasW := legBoxX + legBoxW + 12

	// Legend dimensions (depend on block types and FFN variants present).
	legItemCount := len(blockTypes) + len(ffnKeys) + 2 // block types + FFN variants + global + norm
	legBoxH := 10 + legItemCount*16 + 10

	// Bottom anchor: legend + summary live in the right gutter, stacked
	// vertically.
	contentBottom := bottomHopY + 24
	legBoxY := titleBandH + 8
	gutterBottom := legBoxY + legBoxH + 16 + statsBoxH + 12
	canvasH := max(contentBottom, gutterBottom)

	// --- SVG header + styles ---
	var b strings.Builder
	fmt.Fprintf(&b, `<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg">
<style>
  text { font-family: 'Courier New', monospace; }
  .title { font-size: 20px; font-weight: bold; fill: %s; font-family: system-ui, -apple-system, sans-serif; }
  .lbl   { font-size: 9px; fill: %s; text-anchor: end; dominant-baseline: middle; }
  .tlbl  { font-size: 7px; fill: %s; text-anchor: middle; dominant-baseline: middle; }
  .io    { font-size: 8px; font-weight: bold; fill: %s; text-anchor: middle; dominant-baseline: middle; }
  .ltxt  { font-size: 8px; fill: %s; dominant-baseline: middle; }
  .shdr  { font-size: 8px; font-weight: bold; fill: %s; dominant-baseline: middle; }
</style>
`, canvasW, canvasH,
		pal["ui.text_head"],
		pal["ui.text_label"],
		pal["ui.text_tensor"],
		pal["ui.dot"],
		pal["ui.text_body"],
		pal["ui.text_head"])

	// Gradients.
	emitGradients(&b, "zm_")

	// Symbol defs.
	emitSymbolDef := func(sym symbolDef) {
		fmt.Fprintf(&b, "  <symbol id=\"%s\" overflow=\"visible\">\n", sym.id)
		fmt.Fprint(&b, sym.svg)
		fmt.Fprintf(&b, "  </symbol>\n")
	}
	fmt.Fprintf(&b, "<defs>\n")
	for _, bt := range blockTypes {
		emitSymbolDef(ctx.blockSymbols[bt])
	}
	for _, fk := range ffnKeys {
		emitSymbolDef(ctx.ffnSymbols[fk])
	}
	fmt.Fprintf(&b, "</defs>\n")

	// Background.
	fmt.Fprintf(&b, "  <rect width=\"%d\" height=\"%d\" fill=\"%s\" rx=\"8\"/>\n\n", canvasW, canvasH, pal["ui.canvas_bg"])

	// Title (centered over the U).
	titleCX := (rightColOffset + rowTrunkOutX) / 2
	fmt.Fprintf(&b, "  <text class=\"title\" x=\"%d\" y=\"32\" text-anchor=\"middle\">%s</text>\n", titleCX, title)
	if nonCausal {
		subtitle := "(bidirectional / non-causal)"
		if def.Architecture.Generation == arch.GenerationDiffusion {
			subtitle = "(diffusion — iterative denoising)"
		}
		fmt.Fprintf(&b, "  <text x=\"%d\" y=\"46\" text-anchor=\"middle\" font-size=\"11\" font-weight=\"600\" fill=\"%s\">%s</text>\n",
			titleCX, pal["ui.text_hint"], subtitle)
	}

	// --- Input row: embedding cell at the top of the left column. ---
	leftXOff := 0
	rightXOff := rightColOffset
	embdLeftX := leftXOff + columnCenterOffset - blockCellW/2
	embdMidY := inputRowCY

	if isDiffusion {
		// Buffer cell at the top, spanning from above embd-left to above
		// output-cell-right (so the buffer visually "covers" the U).
		outputCellRightX := rightXOff + cellX1 + outputCellW
		bufferLeftX := embdLeftX
		bufferTopY := bufferRowCY - cellH/2
		emitDiffusionBufferCell(&b, bufferLeftX, bufferTopY, outputCellRightX-bufferLeftX)

		// Embd cell (no TOKEN(s) IN label — buffer connects to it instead).
		_, embdTopY, _ := emitInputCell(&b, leftXOff, inputRowCY)

		// Buffer→embd: vertical line down from buffer-bottom to embd-top,
		// at the embd's horizontal center. This is where the per-iteration
		// "go again" decision lives: each pass reads the (updated) buffer
		// state and re-runs the stack. The "T iterations" label is anchored
		// here, not on the sample→buffer riser, because the iteration count
		// gates entry into the next pass — not the buffer write itself.
		embdCenterX := leftXOff + columnCenterOffset
		bufferBottomY := bufferRowCY + cellH/2
		fmt.Fprintf(&b, "  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"%s\"/>\n",
			embdCenterX, bufferBottomY, embdCenterX, embdTopY, pal["ui.spine"], trunkW)
		emitArrowhead(&b, embdCenterX, embdTopY, "down")
		midY := (bufferBottomY + embdTopY) / 2
		fmt.Fprintf(&b, "  <text x=\"%d\" y=\"%d\" text-anchor=\"start\" font-size=\"8\" font-style=\"italic\" fill=\"%s\">T iterations</text>\n",
			embdCenterX+8, midY+2, pal["ui.text_body"])
	} else {
		emitInputRow(&b, leftXOff, inputRowCY)
	}

	// --- Trunk from embd cell to layer 0: out the LEFT side, DOWN, RIGHT
	// into layer 0's block-left. Mirrors the inter-row U-turn shape used
	// between layers in the same column. Single path, miter-joined corners.
	// embd is centered over the column, so its left edge is at column center
	// minus half cellW.
	turnX := leftXOff + rowTrunkInX
	inLandY := leftCY(0)
	blockLeftX := leftXOff + cellX1
	fmt.Fprintf(&b, "  <path d=\"M %d,%d L %d,%d L %d,%d L %d,%d\" fill=\"none\" stroke=\"%s\" stroke-width=\"%s\" stroke-linejoin=\"miter\"/>\n",
		embdLeftX, embdMidY, turnX, embdMidY, turnX, inLandY, blockLeftX, inLandY,
		pal["ui.spine"], trunkW)
	emitArrowhead(&b, blockLeftX, inLandY, "right")

	// --- Left column (descending) + per-row U-turns ---
	for i, l := range leftLayers {
		emitARLayerRow(ctx, &b, l, layerMap[l], leftXOff, leftCY(i))
		if i < leftCount-1 {
			emitInterRowTurn(&b, leftXOff, leftCY(i), leftCY(i+1))
		}
	}

	// --- Column-bottom hop (last left → first right) ---
	if rightCount > 0 {
		emitColumnBottomHop(&b, leftXOff, rightXOff, bottomCY, rightCY(0), bottomHopY)
	}

	// --- Right column (ascending — rightLayers[0] at bottom, [last] at top) +
	// per-row U-turns going up. Final layer's exit hops UP into the global row. ---
	for j, l := range rightLayers {
		emitARLayerRow(ctx, &b, l, layerMap[l], rightXOff, rightCY(j))
		if j < rightCount-1 {
			emitInterRowTurn(&b, rightXOff, rightCY(j), rightCY(j+1))
		}
	}
	// Top hop: last right layer → output row.
	if rightCount > 0 {
		emitInterRowTurn(&b, rightXOff, lastRightCY, outputRowCY)
	} else {
		// Degenerate: only a left column. No output row to hop to; just route
		// the trunk to a bottom position so the diagram terminates cleanly.
		emitInterRowTurn(&b, leftXOff, bottomCY, bottomCY+pitch)
	}

	// --- Output row (placed in the right column for typical cases) ---
	outputXOff := rightXOff
	if rightCount == 0 {
		outputXOff = leftXOff
	}
	if isDiffusion {
		// Output cell with sample+remask label; loop-back to the buffer
		// instead of an outgoing TOKEN OUT exit.
		_, _, outputTopY, sampleCenterX := emitOutputCell(&b, outputXOff, outputRowCY,
			def.Architecture.TiedEmbeddings, "[sample+remask]")
		bufferBottomY := bufferRowCY + cellH/2
		fmt.Fprintf(&b, "  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"%s\"/>\n",
			sampleCenterX, outputTopY, sampleCenterX, bufferBottomY, pal["ui.spine"], trunkW)
		emitArrowhead(&b, sampleCenterX, bufferBottomY, "up")
	} else {
		emitOutputRow(&b, outputXOff, outputRowCY, def.Architecture.TiedEmbeddings)
	}

	// --- Legend ---
	{
		fmt.Fprintf(&b, "  <text style=\"font-size:9px;font-weight:bold;fill:%s;font-family:'Courier New',monospace\" x=\"%d\" y=\"%d\">Legend</text>\n",
			pal["ui.text_head"], legBoxX+legPad, legBoxY-5)
		fmt.Fprintf(&b, "  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"%s\" stroke=\"%s\" stroke-width=\"0.8\" rx=\"2\"/>\n",
			legBoxX, legBoxY, legBoxW, legBoxH, pal["ui.box_bg"], pal["ui.box_border"])
		ly := legBoxY + 10
		const legSz = 10
		const legLineH = 16
		legItem := func(fill, stroke, label string) {
			fmt.Fprintf(&b, "  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"%s\" stroke=\"%s\" stroke-width=\"0.8\" rx=\"1\"/>\n",
				legBoxX+legPad, ly, legSz, legSz, fill, stroke)
			fmt.Fprintf(&b, "  <text class=\"ltxt\" x=\"%d\" y=\"%d\">%s</text>\n",
				legBoxX+legPad+legSz+4, ly+legSz/2, label)
			ly += legLineH
		}
		for _, bt := range blockTypes {
			pp := palPrefix(bt)
			label := strings.ReplaceAll(bt, "_", " ")
			legItem("url(#zm_"+pp+")", pal[pp+".stroke"], label)
		}
		for _, fk := range ffnKeys {
			var label string
			switch fk {
			case arch.FFNSymMoE:
				label = "FFN (MoE)"
			default:
				label = "FFN (dense)"
			}
			legItem("url(#zm_ffn)", pal["ffn.stroke"], label)
		}
		legItem("url(#zm_global)", pal["global.stroke"], "global")
		legItem(pal["norm.fill"], pal["norm.stroke"], "RMS norm")
	}

	// --- Summary ---
	{
		summaryY := legBoxY + legBoxH + 16
		fmt.Fprintf(&b, "  <text style=\"font-size:9px;font-weight:bold;fill:%s;font-family:'Courier New',monospace\" x=\"%d\" y=\"%d\">Summary</text>\n",
			pal["ui.text_head"], legBoxX+legPad, summaryY-5)
		fmt.Fprintf(&b, "  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"%s\" stroke=\"%s\" stroke-width=\"0.8\" rx=\"2\"/>\n",
			legBoxX, summaryY, legBoxW, statsBoxH, pal["ui.box_bg"], pal["ui.box_border"])
		sy := summaryY + 11
		sectionHeader := func(text string) {
			fmt.Fprintf(&b, "  <text class=\"shdr\" x=\"%d\" y=\"%d\">%s</text>\n", legBoxX+legPad, sy, text)
			sy += 13
		}
		sectionGap := func() { sy += 5 }

		sectionHeader(fmt.Sprintf("%d modules", nTotal))
		sectionGap()
		sectionHeader(fmt.Sprintf("%d tensors", totalTensors))
		sectionGap()
		sectionHeader(fmt.Sprintf("%d weights", totalWeights))
	}

	fmt.Fprintf(&b, "</svg>\n")

	_, err := io.WriteString(w, b.String())
	return err
}
