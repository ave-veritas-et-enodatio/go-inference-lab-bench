package arch

import (
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

// tensorSlot describes the bounding box of one tensor sub-rect within a symbol.
// Coordinates are relative to the symbol's top-left origin (0, 0 = cell top-left).
type tensorSlot struct {
	x, y, w, h int
	shortName   string
}

// symbolDef holds a pre-rendered SVG symbol body and the tensor slot geometry
// for each tensor sub-rect within the symbol.
type symbolDef struct {
	id    string
	slots []tensorSlot
	svg   string // rendered content for the <symbol> body (without outer <symbol> tags)
}

// RenderModuleMap renders a ModuleMap as an SVG showing per-module tensor detail.
//
// Layout: vertical residual spine on left; Block and FFN cells per layer (left-to-right
// data flow within each cell); legend + summary column on the right.
//
// Block and FFN cell types are defined once as SVG <symbol> elements in <defs> and
// instantiated with <use transform="translate(x,y)"> per layer. This eliminates
// duplicate definitions and makes the output efficient for live HTTP delivery.
//
// Each symbol includes its residual contribution arrow (the dashed path back to the
// residual spine). Symbol coordinates are relative to the cell top-left (0,0), with
// overflow="visible" allowing arrows to extend outside the cell boundary.
//
// Output path: moduleMapPath + ".svg"
func RenderModuleMapDiagram(mm *ModuleMap, moduleMapPath string, title string, dims TensorDimsMap) error {
	svgPath := moduleMapPath
	if !strings.HasSuffix(moduleMapPath, ".svg") {
		svgPath = svgPath + ".svg"
	}
	pal := diagramPalette()

	// --- Parse module structure ---
	type layerEntry struct {
		blockID, ffnID   int
		hasBlock, hasFfn bool
	}
	var globalModule *Module
	layerMap := make(map[int]*layerEntry)

	for i := range mm.Modules {
		m := &mm.Modules[i]
		if m.Name == "global" {
			globalModule = m
			continue
		}
		if after, ok := strings.CutPrefix(m.Name, "block_"); ok {
			if l, err := strconv.Atoi(after); err == nil {
				if layerMap[l] == nil {
					layerMap[l] = &layerEntry{}
				}
				layerMap[l].blockID = m.ID
				layerMap[l].hasBlock = true
			}
		} else if after, ok := strings.CutPrefix(m.Name, "ffn_"); ok {
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
	nLayers := len(layerIndices)

	moduleByID := make(map[int]*Module, len(mm.Modules))
	for i := range mm.Modules {
		moduleByID[mm.Modules[i].ID] = &mm.Modules[i]
	}

	// --- Stats ---
	nTotal := len(mm.Modules)
	totalTensors, totalWeights := 0, 0
	for _, m := range mm.Modules {
		totalTensors += len(m.Weights) + len(m.Params)
		totalWeights += len(m.Weights)
	}

	// VRAM stats
	dimsKeyForModule := func(m *Module) string {
		if m.Name == "global" {
			return "global"
		}
		if m.BlockName != "" {
			return m.BlockName
		}
		for _, w := range m.Weights {
			if strings.Contains(w, "_exps") {
				return "ffn_moe"
			}
		}
		return "ffn"
	}
	moduleNbytes := func(m *Module) int64 {
		dk := dimsKeyForModule(m)
		dm := dims[dk]
		if dm == nil {
			return 0
		}
		var total int64
		for _, w := range m.Weights {
			total += dm[w].Nbytes
		}
		for _, p := range m.Params {
			total += dm[p].Nbytes
		}
		return total
	}
	var fullBytes, totalParams int64
	for _, m := range mm.Modules {
		fullBytes += moduleNbytes(&m)
		dk := dimsKeyForModule(&m)
		if dm := dims[dk]; dm != nil {
			for _, w := range m.Weights {
				if td, ok := dm[w]; ok {
					totalParams += td.Ne0 * td.Ne1
				}
			}
		}
	}
	fullMB := float64(fullBytes) / (1024 * 1024)
	hasVRAM := fullBytes > 0

	// title is used in SVG header rendering below

	// --- Layout constants ---
	//
	// All "blk*" positions are relative to the block cell's top-left (0,0).
	// All "ffn*r" positions are relative to the FFN cell's top-left (0,0).
	// These offsets are embedded in the <symbol> definitions; the absolute
	// cell positions (cellX1, cellX2) are supplied via <use transform="translate">.
	const (
		canvasW    = 460
		spineX     = 26
		cellX1     = 36  // block cell absolute left edge
		blockCellW = 110
		cellGap    = 6
		cellX2     = cellX1 + blockCellW + cellGap // 152 — FFN cell absolute left edge
		ffnCellW   = 98
		returnRailX = cellX2 + ffnCellW + 10       // 260
		legendX    = returnRailX + 22              // 282
		cellH      = 40
		pitch      = 58
		cellPad    = 3
		subRowH    = 10
		subRowGap  = 3

		// Block symbol: relative slot positions (origin = cell top-left)
		blockNormW = 16
		blkNormX   = 2                         // left norm slot
		blkDiv1X   = blkNormX + blockNormW + 2 // 20 — divider after norm
		blkQKVX    = blkDiv1X + 2              // 22 — Q/K/V column start
		blockQKVW  = 54
		blkDiv2X   = blkQKVX + blockQKVW + 1  // 77 — divider after QKV
		blkOutX    = blkDiv2X + 2             // 79 — output projection slot
		blockOutW  = 26

		// Block symbol: residual return arrow geometry (relative to cell top-left)
		blkSpineRel = spineX - cellX1          // -10 — spine x in symbol coords
		blkGapCX   = blockCellW + cellGap/2    // 113 — gap center x in symbol coords
		blkGapCY   = cellH / 2                 // 20 — cell vertical center

		// FFN symbol: relative slot positions (origin = FFN cell top-left)
		ffnNormW = 12
		ffnNormXr = 2                           // left norm slot
		ffnNDivr  = ffnNormXr + ffnNormW + 1   // 15 — divider after norm
		ffnGUXr   = ffnNDivr + 2               // 17 — gate/up column start
		ffnGUW    = 52
		ffnGUDivr = ffnGUXr + ffnGUW + 1      // 70 — divider after gate/up
		ffnDownXr = ffnGUDivr + 2             // 72 — down projection slot
		ffnDownW  = 22

		// FFN symbol: return rail geometry (relative to FFN cell top-left)
		ffnSpineRel      = spineX - cellX2           // -126 — spine x in FFN symbol coords
		ffnReturnRailRel = returnRailX - cellX2       // 108 — return rail x in FFN symbol coords
	)

	// --- Helpers ---

	isFFNNorm := func(name string) bool {
		return name == "ffn_norm"
	}
	isNormWeight := func(name string) bool {
		return name == "attn_norm" || name == "output_norm" || name == "ssm_norm" || isFFNNorm(name)
	}

	tensorLabels := map[string]string{
		"attn_norm":           "norm",
		"attn_q":              "Q",
		"attn_k":              "K",
		"attn_v":              "V",
		"attn_output":         "out",
		"ffn_norm":            "norm",
		"post_attention_norm": "norm",
		"ffn_gate":            "G",
		"ffn_up":              "U",
		"ffn_down":            "D",
		"ffn_gate_exps":       "Gx",
		"ffn_up_exps":         "Ux",
		"ffn_down_exps":       "Dx",
		"ffn_gate_shexp":      "Gs",
		"ffn_up_shexp":        "Us",
		"ffn_down_shexp":      "Ds",
		"output_norm":         "norm",
		"ssm_norm":            "norm",
		"token_embd":          "embd",
		"output":              "out",
	}
	derivedLabel := func(shortName string) string {
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

	// ffnSymKey maps an FFN module to its symbol key: "dense" or "moe".
	ffnSymKey := func(m *Module) string {
		for _, w := range m.Weights {
			if strings.Contains(w, "_exps") {
				return "moe"
			}
		}
		return "dense"
	}
	// --- drawBaseRect: render one tensor sub-rect + label into a builder; return its slot.
	drawBaseRect := func(sb *strings.Builder, x, y, w, h int, blockType, shortName string) tensorSlot {
		var prefix string
		if isNormWeight(shortName) {
			prefix = "norm"
		} else {
			prefix = palPrefix(blockType)
		}
		fill, stroke := pal[prefix+".fill"], pal[prefix+".stroke"]
		fmt.Fprintf(sb, "  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"%s\" stroke=\"%s\" stroke-width=\"0.5\" rx=\"1\"/>\n",
			x, y, w, h, fill, stroke)
		lbl := derivedLabel(shortName)
		if lbl != "" && w >= 8 {
			cx, cy := x+w/2, y+h/2
			if h > int(float64(w)*2.0) {
				fmt.Fprintf(sb, "  <text class=\"tlbl\" x=\"%d\" y=\"%d\" transform=\"rotate(-90,%d,%d)\">%s</text>\n",
					cx, cy, cx, cy, lbl)
			} else {
				fmt.Fprintf(sb, "  <text class=\"tlbl\" x=\"%d\" y=\"%d\">%s</text>\n", cx, cy, lbl)
			}
		}
		return tensorSlot{x, y, w, h, shortName}
	}

	// --- buildBlockSymbol: render the SVG body for one block type and collect slot geometry.
	// The symbol origin is the cell top-left (0,0). Residual arrows use overflow="visible"
	// to extend outside the cell into the gap and toward the spine.
	buildBlockSymbol := func(blockType string, m *Module) symbolDef {
		id := "blk-" + blockType
		var sb strings.Builder
		var slots []tensorSlot
		pp := palPrefix(blockType)
		normH := cellH - 2*cellPad
		normY := cellPad

		// Outer cell rect
		fmt.Fprintf(&sb, "  <rect x=\"0\" y=\"0\" width=\"%d\" height=\"%d\" fill=\"url(#zm_%s)\" stroke=\"%s\" stroke-width=\"0.8\" rx=\"2\"/>\n",
			blockCellW, cellH, pp, pal[pp+".stroke"])

		if isAttentionBlock(m) {
			slots = append(slots, drawBaseRect(&sb, blkNormX, normY, blockNormW, normH, blockType, "attn_norm"))
			fmt.Fprintf(&sb, "  <line x1=\"%d\" y1=\"0\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"0.7\"/>\n",
				blkDiv1X, blkDiv1X, cellH, pal["ui.divider"])
			total3H := 3*subRowH + 2*subRowGap
			qStartY := (cellH - total3H) / 2
			for qi, qn := range []string{"attn_q", "attn_k", "attn_v"} {
				qy := qStartY + qi*(subRowH+subRowGap)
				slots = append(slots, drawBaseRect(&sb, blkQKVX, qy, blockQKVW, subRowH, blockType, qn))
			}
			fmt.Fprintf(&sb, "  <line x1=\"%d\" y1=\"0\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"0.7\"/>\n",
				blkDiv2X, blkDiv2X, cellH, pal["ui.divider"])
			slots = append(slots, drawBaseRect(&sb, blkOutX, normY, blockOutW, normH, blockType, "attn_output"))

			// Visual indicator for sliding-window attention blocks
			if strings.Contains(blockType, "swa") {
				fmt.Fprintf(&sb, "  <text x=\"%d\" y=\"%d\" font-size=\"6\" font-weight=\"700\" fill=\"%s\" text-anchor=\"end\">SW</text>\n",
					blockCellW-2, 8, pal[pp+".stroke"])
			}
		} else {
			// Generic: attn_norm as left norm slot, remaining weights as equal sub-rects.
			var normWt string
			remaining := make([]string, 0, len(m.Weights))
			for _, w := range m.Weights {
				if w == "attn_norm" {
					normWt = w
				} else {
					remaining = append(remaining, w)
				}
			}
			sort.Strings(remaining)
			startX := 2
			if normWt != "" {
				slots = append(slots, drawBaseRect(&sb, startX, normY, 12, normH, blockType, normWt))
				divX := startX + 12 + 1
				fmt.Fprintf(&sb, "  <line x1=\"%d\" y1=\"0\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"0.7\"/>\n",
					divX, divX, cellH, pal["ui.divider"])
				startX = divX + 2
			}
			if n := len(remaining); n > 0 {
				innerW := blockCellW - 2 - startX
				subW := innerW / n
				if subW < 1 {
					subW = 1
				}
				for wi, wn := range remaining {
					wx := startX + wi*subW
					slots = append(slots, drawBaseRect(&sb, wx, normY, subW-1, normH, blockType, wn))
				}
			}
		}

		// Block residual return arrow.
		// Extends from the gap center (blkGapCX, blkGapCY) down, then dashes left to spine.
		// Uses overflow="visible" on the symbol to extend outside the cell bounds.
		retCol := pal[pp+".stroke"]
		retY := cellH + 4
		fmt.Fprintf(&sb, "  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"0.8\"/>\n",
			blkGapCX, blkGapCY, blkGapCX, retY, retCol)
		fmt.Fprintf(&sb, "  <path d=\"M %d,%d L %d,%d\" fill=\"none\" stroke=\"%s\" stroke-width=\"1.0\" stroke-dasharray=\"2,3\"/>\n",
			blkGapCX, retY, blkSpineRel, retY, retCol)
		fmt.Fprintf(&sb, "  <polyline points=\"%d,%d %d,%d %d,%d\" fill=\"none\" stroke=\"%s\" stroke-width=\"1.0\"/>\n",
			blkSpineRel+4, retY-2, blkSpineRel, retY, blkSpineRel+4, retY+2, retCol)

		return symbolDef{id: id, slots: slots, svg: sb.String()}
	}

	// --- buildFFNSymbol: render the SVG body for one FFN type and collect slot geometry.
	// The symbol origin is the FFN cell top-left (0,0). The return rail and dashed path
	// extend to the right and left respectively using overflow="visible".
	buildFFNSymbol := func(sk string, m *Module) symbolDef {
		id := "ffn-" + sk
		isMoE := sk == "moe"
		var sb strings.Builder
		var slots []tensorSlot
		normH := cellH - 2*cellPad
		normY := cellPad

		// Outer cell rect
		fmt.Fprintf(&sb, "  <rect x=\"0\" y=\"0\" width=\"%d\" height=\"%d\" fill=\"url(#zm_ffn)\" stroke=\"%s\" stroke-width=\"0.8\" rx=\"2\"/>\n",
			ffnCellW, cellH, pal["ffn.stroke"])

		// Find pre-FFN norm weight
		var normWt string
		for _, w := range m.Weights {
			if isFFNNorm(w) {
				normWt = w
				break
			}
		}
		if normWt != "" {
			slots = append(slots, drawBaseRect(&sb, ffnNormXr, normY, ffnNormW, normH, "ffn", normWt))
			fmt.Fprintf(&sb, "  <line x1=\"%d\" y1=\"0\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"0.7\"/>\n",
				ffnNDivr, ffnNDivr, cellH, pal["ui.divider"])
		}

		if !isMoE {
			// Dense: gate/up stacked | down
			total2H := 2*subRowH + 1*subRowGap
			guStartY := (cellH - total2H) / 2
			for gi, gn := range []string{"ffn_gate", "ffn_up"} {
				gy := guStartY + gi*(subRowH+subRowGap)
				slots = append(slots, drawBaseRect(&sb, ffnGUXr, gy, ffnGUW, subRowH, "ffn", gn))
			}
			slots = append(slots, drawBaseRect(&sb, ffnDownXr, normY, ffnDownW, normH, "ffn", "ffn_down"))
			fmt.Fprintf(&sb, "  <line x1=\"%d\" y1=\"0\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"0.7\"/>\n",
				ffnGUDivr, ffnGUDivr, cellH, pal["ui.divider"])
		} else {
			// MoE: expert 3-D boxes + shared weights
			expertAreaX := ffnGUXr
			expertAreaW := ffnCellW - expertAreaX - 2

			var expertWts, sharedWts []string
			for _, w := range m.Weights {
				switch {
				case isFFNNorm(w):
					// already handled
				case strings.Contains(w, "_exps"):
					expertWts = append(expertWts, w)
				default:
					sharedWts = append(sharedWts, w)
				}
			}
			sort.Strings(expertWts)
			sort.Strings(sharedWts)

			if ne := len(expertWts); ne > 0 {
				bs := expertAreaW * 10 / (ne * 12)
				if bs < 6 {
					bs = 6
				}
				ld := (bs*2 + 9) / 10
				if ld < 1 {
					ld = 1
				}
				gap := ld
				totalBoxW := ne*bs + (ne-1)*gap + ld
				startX := expertAreaX + (expertAreaW-totalBoxW)/2
				if startX < expertAreaX {
					startX = expertAreaX
				}
				bY := normY + (normH-bs-ld)/2 + ld

				for xi, wn := range expertWts {
					bX := startX + xi*(bs+gap)
					// Square face
					fmt.Fprintf(&sb, "  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"%s\" stroke=\"%s\" stroke-width=\"0.5\"/>\n",
						bX, bY, bs, bs, pal["ffn.fill"], pal["ffn.stroke"])
					// Lid parallelogram
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
					bs := expertAreaW * 10 / (ne * 12)
					if bs < 6 {
						bs = 6
					}
					ld := (bs*2 + 9) / 10
					if ld < 1 {
						ld = 1
					}
					avail -= ne*bs + (ne-1)*ld + ld + 2
				}
				subW := avail / ns
				if subW >= 6 {
					sharedX := expertAreaX + (expertAreaW - avail)
					for ri, wn := range sharedWts {
						wx := sharedX + ri*subW
						slots = append(slots, drawBaseRect(&sb, wx, normY, subW-1, normH, "ffn_moe", wn))
					}
				}
			}
		}

		// FFN return rail: stub from right edge to returnRail, then dashes down + left to spine.
		ffnRetY := cellH + 12
		fmt.Fprintf(&sb, "  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"0.8\"/>\n",
			ffnCellW, cellH/2, ffnReturnRailRel, cellH/2, pal["ui.spine"])
		fmt.Fprintf(&sb, "  <circle cx=\"%d\" cy=\"%d\" r=\"2\" fill=\"%s\"/>\n",
			ffnReturnRailRel, cellH/2, pal["ui.dot"])
		fmt.Fprintf(&sb, "  <path d=\"M %d,%d L %d,%d L %d,%d\" fill=\"none\" stroke=\"%s\" stroke-width=\"1.0\" stroke-dasharray=\"2,3\"/>\n",
			ffnReturnRailRel, cellH/2, ffnReturnRailRel, ffnRetY, ffnSpineRel, ffnRetY, pal["ffn.stroke"])
		fmt.Fprintf(&sb, "  <polyline points=\"%d,%d %d,%d %d,%d\" fill=\"none\" stroke=\"%s\" stroke-width=\"1.0\"/>\n",
			ffnSpineRel+4, ffnRetY-2, ffnSpineRel, ffnRetY, ffnSpineRel+4, ffnRetY+2, pal["ffn.stroke"])

		return symbolDef{id: id, slots: slots, svg: sb.String()}
	}

	// --- Collect unique block/FFN symbols from module list ---
	// Uses the first module of each type to build the symbol; all same-type modules
	// have identical weight structure within a model.
	blockSymbols := make(map[string]symbolDef) // blockType → def
	ffnSymbols := make(map[string]symbolDef)   // "dense"/"moe" → def
	for i := range mm.Modules {
		m := &mm.Modules[i]
		if strings.HasPrefix(m.Name, "block_") && m.BlockName != "" {
			if _, seen := blockSymbols[m.BlockName]; !seen {
				blockSymbols[m.BlockName] = buildBlockSymbol(m.BlockName, m)
			}
		} else if strings.HasPrefix(m.Name, "ffn_") {
			sk := ffnSymKey(m)
			if _, seen := ffnSymbols[sk]; !seen {
				ffnSymbols[sk] = buildFFNSymbol(sk, m)
			}
		}
	}
	// Sorted symbol keys for deterministic output.
	var blockTypes []string
	for bt := range blockSymbols {
		blockTypes = append(blockTypes, bt)
	}
	sort.Strings(blockTypes)
	var ffnKeys []string
	for fk := range ffnSymbols {
		ffnKeys = append(ffnKeys, fk)
	}
	sort.Strings(ffnKeys)

	// --- Pre-compute box dimensions needed by symbol builders ---
	legBoxX := legendX - 4
	legBoxW := canvasW - legBoxX - 4
	legItemCount := len(blockTypes) + 3 // block types + FFN + global + norm
	legBoxH := 10 + legItemCount*16 + 10 // top pad + items + bottom pad
	hasParams := totalParams > 0
	statsBoxH := 75
	if hasParams {
		statsBoxH += 20
	}
	if hasVRAM {
		statsBoxH += 20
	}

	// --- Build row-chrome symbol (spine dot + connectors + arrowhead) ---
	// Origin at (0, 0) = row center-Y. Placed with translate(0, cy).
	var rowChromeSVG strings.Builder
	fmt.Fprintf(&rowChromeSVG, "  <circle cx=\"%d\" cy=\"0\" r=\"2.5\" fill=\"%s\"/>\n", spineX, pal["ui.dot"])
	fmt.Fprintf(&rowChromeSVG, "  <line x1=\"%d\" y1=\"0\" x2=\"%d\" y2=\"0\" stroke=\"%s\" stroke-width=\"0.8\"/>\n",
		spineX, cellX1, pal["ui.spine"])
	fmt.Fprintf(&rowChromeSVG, "  <line x1=\"%d\" y1=\"0\" x2=\"%d\" y2=\"0\" stroke=\"%s\" stroke-width=\"0.8\"/>\n",
		cellX1+blockCellW, cellX2, pal["ui.spine"])
	fmt.Fprintf(&rowChromeSVG, "  <polygon points=\"%d,-2 %d,0 %d,2\" fill=\"%s\"/>\n",
		cellX2-4, cellX2, cellX2-4, pal["ui.spine"])

	// --- Build global-row symbol (spine turn + cell + sub-rects + OUT stub) ---
	// Origin at (0, 0) = globalCY. Placed with translate(0, globalCY).
	var globalRowSVG strings.Builder
	halfH := cellH / 2
	fmt.Fprintf(&globalRowSVG, "  <line x1=\"%d\" y1=\"0\" x2=\"%d\" y2=\"0\" stroke=\"%s\" stroke-width=\"0.8\"/>\n",
		spineX, cellX1, pal["ui.spine"])
	fmt.Fprintf(&globalRowSVG, "  <polyline points=\"%d,-2 %d,0 %d,2\" fill=\"none\" stroke=\"%s\" stroke-width=\"0.8\"/>\n",
		cellX1-4, cellX1, cellX1-4, pal["ui.spine"])
	if globalModule != nil {
		gw := blockCellW + cellGap + ffnCellW
		fmt.Fprintf(&globalRowSVG, "  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"url(#zm_global)\" stroke=\"%s\" stroke-width=\"0.8\" rx=\"2\"/>\n",
			cellX1, -halfH, gw, cellH, pal["global.stroke"])
		gWeights := make([]string, len(globalModule.Weights))
		copy(gWeights, globalModule.Weights)
		sort.Strings(gWeights)
		if n := len(gWeights); n > 0 {
			innerW := gw - 4
			subW := innerW / n
			if subW < 1 {
				subW = 1
			}
			gNormY := -halfH + cellPad
			gNormH := cellH - 2*cellPad
			for wi, wn := range gWeights {
				wx := cellX1 + 2 + wi*subW
				drawBaseRect(&globalRowSVG, wx, gNormY, subW-1, gNormH, "global", wn)
			}
		}
		// OUT stub: arrow going up from top-center of global cell
		cellCX := cellX1 + gw/2
		outStubY := -halfH - 16
		fmt.Fprintf(&globalRowSVG, "  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"0.8\"/>\n",
			cellCX, -halfH, cellCX, outStubY, pal["ui.spine"])
		fmt.Fprintf(&globalRowSVG, "  <polyline points=\"%d,%d %d,%d %d,%d\" fill=\"none\" stroke=\"%s\" stroke-width=\"1\"/>\n",
			cellCX-3, outStubY+6, cellCX, outStubY, cellCX+3, outStubY+6, pal["ui.spine"])
		fmt.Fprintf(&globalRowSVG, "  <text class=\"io\" x=\"%d\" y=\"%d\">OUT</text>\n", cellCX, outStubY-6)
	}

	// --- Build legend symbol ---
	// Origin at (0, 0) = box top-left. Placed with translate(legBoxX, legBoxY).
	const legPad = 4 // legendX - legBoxX
	var legendSVG strings.Builder
	fmt.Fprintf(&legendSVG, "  <text style=\"font-size:9px;font-weight:bold;fill:%s;font-family:'Courier New',monospace\" x=\"%d\" y=\"-5\">Legend</text>\n", pal["ui.text_head"], legPad)
	fmt.Fprintf(&legendSVG, "  <rect width=\"%d\" height=\"%d\" fill=\"%s\" stroke=\"%s\" stroke-width=\"0.8\" rx=\"2\"/>\n",
		legBoxW, legBoxH, pal["ui.box_bg"], pal["ui.box_border"])
	{
		ly := 10
		const legSz = 10
		const legLineH = 16
		legItem := func(fill, stroke, label string) {
			fmt.Fprintf(&legendSVG, "  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"%s\" stroke=\"%s\" stroke-width=\"0.8\" rx=\"1\"/>\n",
				legPad, ly, legSz, legSz, fill, stroke)
			fmt.Fprintf(&legendSVG, "  <text class=\"ltxt\" x=\"%d\" y=\"%d\">%s</text>\n", legPad+legSz+4, ly+legSz/2, label)
			ly += legLineH
		}
		// Show legend entries for block types actually present in this model,
		// using the block type name as the label (underscores → spaces).
		for _, bt := range blockTypes {
			pp := palPrefix(bt)
			label := strings.ReplaceAll(bt, "_", " ")
			legItem("url(#zm_"+pp+")", pal[pp+".stroke"], label)
		}
		legItem("url(#zm_ffn)", pal["ffn.stroke"], "FFN (dense)")
		legItem("url(#zm_global)", pal["global.stroke"], "global")
		legItem(pal["norm.fill"], pal["norm.stroke"], "RMS norm")
	}

	// --- Build summary symbol ---
	// Origin at (0, 0) = box top-left. Placed with translate(legBoxX, statsBoxY).
	var summarySVG strings.Builder
	fmt.Fprintf(&summarySVG, "  <text style=\"font-size:9px;font-weight:bold;fill:%s;font-family:'Courier New',monospace\" x=\"%d\" y=\"-5\">Summary</text>\n", pal["ui.text_head"], legPad)
	fmt.Fprintf(&summarySVG, "  <rect width=\"%d\" height=\"%d\" fill=\"%s\" stroke=\"%s\" stroke-width=\"0.8\" rx=\"2\"/>\n",
		legBoxW, statsBoxH, pal["ui.box_bg"], pal["ui.box_border"])
	{
		sy := 11
		sectionHeader := func(text string) {
			fmt.Fprintf(&summarySVG, "  <text class=\"shdr\" x=\"%d\" y=\"%d\">%s</text>\n", legPad, sy, text)
			sy += 13
		}
		sectionGap := func() { sy += 5 }

		sectionHeader(fmt.Sprintf("%d modules", nTotal))
		sectionGap()
		sectionHeader(fmt.Sprintf("%d tensors", totalTensors))
		sectionGap()
		sectionHeader(fmt.Sprintf("%d weights", totalWeights))

		if totalParams > 0 {
			sectionGap()
			fmtParams := func(n int64) string {
				switch {
				case n >= 1_000_000_000:
					return fmt.Sprintf("%.1fB", float64(n)/1e9)
				case n >= 1_000_000:
					return fmt.Sprintf("%.0fM", float64(n)/1e6)
				default:
					return fmt.Sprintf("%dK", n/1000)
				}
			}
			sectionHeader(fmt.Sprintf("%s parameters", fmtParams(totalParams)))
		}

		if hasVRAM {
			sectionGap()
			sectionHeader(fmt.Sprintf("%.1f MB vram", fullMB))
		}
	}

	// --- SVG rendering ---
	var b strings.Builder

	// --- U-layout: split layers into left (descending) and right (ascending) columns ---
	// N = nLayers + 1 (global counts as one element).
	// Left column gets int((N+1)/2) layers, right gets remaining layers + global at top.
	totalElements := nLayers + 1 // layers + global
	leftCount := (totalElements + 1) / 2
	if leftCount > nLayers {
		leftCount = nLayers
	}
	rightCount := nLayers - leftCount // layers on the right (global is separate)
	leftLayers := layerIndices[:leftCount]
	rightLayers := layerIndices[leftCount:]

	const rightColOffset = 290 // horizontal offset for right column

	// Vertical layout
	spineY1 := 30
	firstRowCY := spineY1 + 10
	colRows := leftCount // left column always has >= right column rows
	bottomCY := firstRowCY + (colRows-1)*pitch
	turnY := bottomCY + pitch/2 + 10 // U-turn Y below the bottom row

	// Right column: layers ascend from bottom to top
	// rightLayers[0] at bottomCY, rightLayers[last] near the top
	// Global module sits above the top right layer
	rightTopCY := bottomCY - (rightCount-1)*pitch
	globalCY := rightTopCY - pitch

	// Canvas dimensions
	canvasW2 := rightColOffset + returnRailX + 30
	canvasH := turnY + 30

	// Legend + summary below the U (side by side, top-aligned)
	legBoxY := turnY + 30
	tallBox := legBoxH
	if statsBoxH > tallBox {
		tallBox = statsBoxH
	}
	minCanvasH := legBoxY + tallBox + 12
	if canvasH < minCanvasH {
		canvasH = minCanvasH
	}

	fmt.Fprintf(&b, `<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg">
<style>
  text { font-family: 'Courier New', monospace; }
  .title { font-size: 11px; font-weight: bold; fill: %s; }
  .hdr   { font-size: 9px; font-weight: bold; fill: %s; text-anchor: middle; dominant-baseline: middle; }
  .lbl   { font-size: 9px; fill: %s; text-anchor: end; dominant-baseline: middle; }
  .tlbl  { font-size: 7px; fill: %s; text-anchor: middle; dominant-baseline: middle; }
  .io    { font-size: 8px; font-weight: bold; fill: %s; text-anchor: middle; dominant-baseline: middle; }
  .gtxt  { font-size: 9px; text-anchor: middle; dominant-baseline: middle; }
  .ltxt  { font-size: 8px; fill: %s; dominant-baseline: middle; }
  .shdr  { font-size: 8px; font-weight: bold; fill: %s; dominant-baseline: middle; }
</style>
<defs>
`, canvasW2, canvasH,
		pal["ui.text_title"],
		pal["ui.text_head"],
		pal["ui.text_label"],
		pal["ui.text_tensor"],
		pal["ui.dot"],
		pal["ui.text_body"],
		pal["ui.text_head"])

	// Gradients
	for _, name := range []string{"full_attention", "swa", "recurrent", "ffn", "global"} {
		fmt.Fprintf(&b, "  <linearGradient id=\"zm_%s\" x1=\"0\" y1=\"0\" x2=\"0\" y2=\"1\">\n", name)
		fmt.Fprintf(&b, "    <stop offset=\"0%%%%\" stop-color=\"%s\"/><stop offset=\"100%%%%\" stop-color=\"%s\"/>\n",
			pal[name+".grad_top"], pal[name+".grad_bottom"])
		fmt.Fprintf(&b, "  </linearGradient>\n")
	}

	// Block symbols
	for _, bt := range blockTypes {
		sym := blockSymbols[bt]
		fmt.Fprintf(&b, "  <symbol id=\"%s\" overflow=\"visible\">\n", sym.id)
		fmt.Fprint(&b, sym.svg)
		fmt.Fprintf(&b, "  </symbol>\n")
	}

	// FFN symbols
	for _, fk := range ffnKeys {
		sym := ffnSymbols[fk]
		fmt.Fprintf(&b, "  <symbol id=\"%s\" overflow=\"visible\">\n", sym.id)
		fmt.Fprint(&b, sym.svg)
		fmt.Fprintf(&b, "  </symbol>\n")
	}

	// Row chrome symbol
	fmt.Fprintf(&b, "  <symbol id=\"row-chrome\" overflow=\"visible\">\n")
	fmt.Fprint(&b, rowChromeSVG.String())
	fmt.Fprintf(&b, "  </symbol>\n")

	// Global row symbol
	fmt.Fprintf(&b, "  <symbol id=\"global-row\" overflow=\"visible\">\n")
	fmt.Fprint(&b, globalRowSVG.String())
	fmt.Fprintf(&b, "  </symbol>\n")

	// Legend symbol
	fmt.Fprintf(&b, "  <symbol id=\"legend\" overflow=\"visible\">\n")
	fmt.Fprint(&b, legendSVG.String())
	fmt.Fprintf(&b, "  </symbol>\n")

	// Summary symbol
	fmt.Fprintf(&b, "  <symbol id=\"summary\" overflow=\"visible\">\n")
	fmt.Fprint(&b, summarySVG.String())
	fmt.Fprintf(&b, "  </symbol>\n")

	fmt.Fprintf(&b, "</defs>\n")

  // Background
	fmt.Fprintf(&b, "  <rect width=\"%d\" height=\"%d\" fill=\"%s\" rx=\"8\"/>\n\n", canvasW2, canvasH, pal["ui.canvas_bg"])

	// Title (centered over the U)
	titleCX := (rightColOffset + returnRailX) / 2
	fmt.Fprintf(&b, "  <text class=\"title\" x=\"%d\" y=\"14\" text-anchor=\"middle\">%s</text>\n", titleCX, title)

	// U-shaped spine
	rightSpineX := rightColOffset + spineX
	leftSpineBottom := bottomCY + pitch/4
	rightSpineBottom := leftSpineBottom
	rightSpineTop := globalCY

	// Left spine (descending)
	fmt.Fprintf(&b, "  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"1.5\"/>\n",
		spineX, spineY1, spineX, leftSpineBottom, pal["ui.spine"])
	fmt.Fprintf(&b, "  <text class=\"io\" x=\"%d\" y=\"%d\">IN</text>\n", spineX, spineY1-6)

	// U-turn at bottom (left spine → right spine)
	fmt.Fprintf(&b, "  <path d=\"M %d,%d L %d,%d L %d,%d\" fill=\"none\" stroke=\"%s\" stroke-width=\"1.5\"/>\n",
		spineX, leftSpineBottom, spineX, turnY, rightSpineX, turnY, pal["ui.spine"])
	fmt.Fprintf(&b, "  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"1.5\"/>\n",
		rightSpineX, turnY, rightSpineX, rightSpineBottom, pal["ui.spine"])

	// Right spine (ascending)
	fmt.Fprintf(&b, "  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"1.5\"/>\n",
		rightSpineX, rightSpineBottom, rightSpineX, rightSpineTop, pal["ui.spine"])

	// --- Left column layers (descending) ---
	emitLayerRow := func(l int, lr *layerEntry, cy, xOff int) {
		rowY := cy - cellH/2
		fmt.Fprintf(&b, "  <use href=\"#row-chrome\" transform=\"translate(%d,%d)\"/>\n", xOff, cy)
		fmt.Fprintf(&b, "  <text class=\"lbl\" x=\"%d\" y=\"%d\">%d</text>\n", xOff+spineX-5, cy, l)

		if lr.hasBlock {
			m := moduleByID[lr.blockID]
			sym := blockSymbols[m.BlockName]
			fmt.Fprintf(&b, "  <use href=\"#%s\" transform=\"translate(%d,%d)\"/>\n", sym.id, xOff+cellX1, rowY)
		}

		if lr.hasFfn {
			m := moduleByID[lr.ffnID]
			sk := ffnSymKey(m)
			sym := ffnSymbols[sk]
			fmt.Fprintf(&b, "  <use href=\"#%s\" transform=\"translate(%d,%d)\"/>\n", sym.id, xOff+cellX2, rowY)
		}
	}

	for i, l := range leftLayers {
		cy := firstRowCY + i*pitch
		emitLayerRow(l, layerMap[l], cy, 0)
	}

	// --- Right column layers (ascending — rightLayers[0] at bottom, last at top) ---
	for j, l := range rightLayers {
		cy := bottomCY - j*pitch
		emitLayerRow(l, layerMap[l], cy, rightColOffset)
	}

	// Global row at top of right column
	fmt.Fprintf(&b, "  <use href=\"#global-row\" transform=\"translate(%d,%d)\"/>\n", rightColOffset, globalCY)

	// Legend (bottom-left) + Summary (bottom-right, top-aligned with legend)
	fmt.Fprintf(&b, "  <use href=\"#legend\" transform=\"translate(%d,%d)\"/>\n", spineX-4, legBoxY)
	fmt.Fprintf(&b, "  <use href=\"#summary\" transform=\"translate(%d,%d)\"/>\n", rightColOffset+spineX-4, legBoxY)

	fmt.Fprintf(&b, "</svg>\n")

	return os.WriteFile(svgPath, []byte(b.String()), 0644)
}

// isAttentionBlock returns true if the module has the standard attention weight pattern
// (attn_q, attn_k, attn_v, attn_output). Used to determine rendering layout in the
// module map — attention blocks get the structured Q/K/V layout regardless of palette prefix.
func isAttentionBlock(m *Module) bool {
	has := map[string]bool{}
	for _, w := range m.Weights {
		has[w] = true
	}
	return has["attn_q"] && has["attn_k"] && has["attn_v"] && has["attn_output"]
}
