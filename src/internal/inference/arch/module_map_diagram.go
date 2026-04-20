package arch

import (
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
)

// tensorSlot describes the bounding box of one tensor sub-rect within a symbol.
// Coordinates are relative to the symbol's top-left origin (0, 0 = cell top-left).
// Used to position per-layer trim overlays on top of placed <use> instances.
type tensorSlot struct {
	x, y, w, h int
	shortName   string
}

// symbolDef holds a pre-rendered SVG symbol body and the tensor slot geometry
// needed to overlay per-layer trim annotations on placed <use> elements.
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
// instantiated with <use transform="translate(x,y)"> per layer. Per-layer trim overlays
// are drawn as separate <rect> elements on top of each <use>, using the slot geometry
// stored in each symbolDef. This eliminates duplicate definitions and makes the output
// efficient for live HTTP delivery.
//
// Each symbol includes its residual contribution arrow (the dashed path back to the
// residual spine). Symbol coordinates are relative to the cell top-left (0,0), with
// overflow="visible" allowing arrows to extend outside the cell boundary.
//
// Output path: moduleMapPath + ".svg"
// engageOpacity maps engagement (1 - cosSim) to overlay opacity using sqrt scale.
// sqrt gives good discrimination at both the low end (where near-zero engagement
// maps to near-zero opacity) and the high end (where 0.7 vs 0.95 are visually
// distinguishable). Returns 0 for sub-threshold values, up to 0.8 for full engagement.
func engageOpacity(engagement float64) float64 {
	const minThresh = 0.001
	const maxOpacity = 0.8
	if engagement < minThresh {
		return 0
	}
	return math.Sqrt(engagement) * maxOpacity
}

func RenderModuleMapDiagram(arch *ArchDef, mm *ModuleMap, moduleMapPath string, subTitle string, dims TensorDimsMap, engagement *EngagementData) error {
  title := strings.Title(arch.Architecture.Name) + subTitle
  nonCausal := arch.Architecture.NonCausal
  generation := arch.Architecture.Generation

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
		if m.Name == ModuleGlobal {
			globalModule = m
			continue
		}
		if after, ok := strings.CutPrefix(m.Name, PrefixBlock); ok {
			if l, err := strconv.Atoi(after); err == nil {
				if layerMap[l] == nil {
					layerMap[l] = &layerEntry{}
				}
				layerMap[l].blockID = m.ID
				layerMap[l].hasBlock = true
			}
		} else if after, ok := strings.CutPrefix(m.Name, PrefixFFN); ok {
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
	totalTensors := 0
	totalWeights := 0
	for _, m := range mm.Modules {
		n := len(m.Weights) + len(m.Params)
		nw := len(m.Weights)
		totalTensors += n
		totalWeights += nw
	}
	nCulled, culledTensors, elidedWeights, culledBytes, culledParams := 0, 0, 0, int64(0), int64(0)
	for _, cs := range mm.CulledByType {
		nCulled += cs.Count
		culledTensors += cs.Tensors
		elidedWeights += cs.Weights
		culledParams += cs.Parameters
		culledBytes += cs.Bytes
	}
	culledMB := float64(culledBytes) / (1024 * 1024)
	pct := func(n, total int) int {
		if total == 0 {
			return 0
		}
		return n * 100 / total
	}

	// VRAM stats
	dimsKeyForModule := func(m *Module) string {
		if m.Name == ModuleGlobal {
			return ModuleGlobal
		}
		if m.BlockName != "" {
			return m.BlockName
		}
		if m.FFNExpertRouted {
			return TypeFFNMoE
		}
		return TypeFFN
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

	inactiveParams := culledParams

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
		cellGap    = 14
		cellX2     = cellX1 + blockCellW + cellGap // 160 — FFN cell absolute left edge
		ffnCellW   = 98
		returnRailX = cellX2 + ffnCellW + 10       // 268
		legendX    = returnRailX + 22              // 290
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
		blkGapCX   = blockCellW + cellGap/2    // 117 — gap center x in symbol coords
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
		ffnSpineRel      = spineX - cellX2           // -134 — spine x in FFN symbol coords
		ffnReturnRailRel = returnRailX - cellX2       // 108 — return rail x in FFN symbol coords
	)

	// --- Helpers ---

	isFFNNorm := func(name string) bool {
		return name == WeightFFNNorm
	}
	isNormWeight := func(name string) bool {
		return name == WeightAttnNorm || name == WeightPostAttnNorm || name == WeightOutputNorm || name == WeightSSMNorm || isFFNNorm(name)
	}

	tensorLabels := map[string]string{
		WeightAttnNorm:     "norm",
		WeightAttnQ:        "Q",
		WeightAttnK:        "K",
		WeightAttnV:        "V",
		WeightAttnOutput:   "out",
		WeightAttnQNorm:    "Qn",
		WeightAttnKNorm:    "Kn",
		WeightRoPEFreqs:    "rope",
		WeightRoPE:         "rope",
		WeightPostAttnNorm: "norm",
		WeightFFNNorm:      "norm",
		WeightFFNGate:      "G",
		WeightFFNUp:        "U",
		WeightFFNDown:      "D",
		WeightFFNGateExps:  "Gx",
		WeightFFNUpExps:    "Ux",
		WeightFFNDownExps:  "Dx",
		WeightFFNGateShexp: "Gs",
		WeightFFNUpShexp:   "Us",
		WeightFFNDownShexp: "Ds",
		WeightOutputNorm:   "norm",
		WeightSSMNorm:      "norm",
		WeightTokenEmbd:    "embd",
		WeightOutput:       "out",
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

	// ffnSymKey maps an FFN module to its symbol key: FFNSymDense or FFNSymMoE.
	ffnSymKey := func(m *Module) string {
		if m.FFNExpertRouted {
			return FFNSymMoE
		}
		return FFNSymDense
	}
	// --- drawBaseRect: render one tensor sub-rect + label into a builder; return its slot.
	// Draws the active (full-color) version only — trim overlays are applied separately.
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
	// The symbol origin is the cell top-left (0,0). Residual return features are in
	// separate -res-desc / -res-asc symbols (see buildBlockResidual).
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
			// Classify weights into data-flow columns:
			//   norm | Q,K,V | Qn,Kn,rope (pre-attn extras) | out | post_norm
			// This matches the actual computation order in prepareQKV → attention → output → post-norm.
			coreSet := map[string]bool{WeightAttnNorm: true, WeightAttnQ: true, WeightAttnK: true, WeightAttnV: true, WeightAttnOutput: true, WeightPostAttnNorm: true}
			hasPostNorm := false
			var extras []string
			for _, w := range m.Weights {
				if w == WeightPostAttnNorm {
					hasPostNorm = true
				} else if !coreSet[w] {
					extras = append(extras, w)
				}
			}
			// RoPE is always part of the attention compute graph. If rope_freqs
			// is not a stored weight (most models compute frequencies from params),
			// inject a synthetic entry so the diagram represents the full compute path.
			hasRoPE := false
			for _, e := range extras {
				if e == WeightRoPEFreqs {
					hasRoPE = true
					break
				}
			}
			if !hasRoPE {
				extras = append(extras, WeightRoPE)
			}
			sort.Strings(extras)

			// Compute column widths dynamically.
			// Columns: input_norm(fixed 12) | QKV | extras(if any) | out | post_norm(if any)
			const divW = 3   // divider (1px line + 2px gap)
			inputNormW := 12 // narrower than default blockNormW to save space
			nDividers := 2   // after input_norm, after QKV, and between out sections
			if len(extras) > 0 {
				nDividers++
			}
			if hasPostNorm {
				nDividers++
			}
			avail := blockCellW - blkNormX - inputNormW - 2 - nDividers*divW
			// Distribute remaining space proportionally.
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
			slots = append(slots, drawBaseRect(&sb, x, normY, inputNormW, normH, blockType, WeightAttnNorm))
			x += inputNormW + 1
			fmt.Fprintf(&sb, "  <line x1=\"%d\" y1=\"0\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"0.7\"/>\n",
				x, x, cellH, pal["ui.divider"])
			x += 2

			// QKV column
			total3H := 3*subRowH + 2*subRowGap
			qStartY := (cellH - total3H) / 2
			for qi, qn := range []string{WeightAttnQ, WeightAttnK, WeightAttnV} {
				qy := qStartY + qi*(subRowH+subRowGap)
				slots = append(slots, drawBaseRect(&sb, x, qy, qkvW, subRowH, blockType, qn))
			}
			x += qkvW + 1
			fmt.Fprintf(&sb, "  <line x1=\"%d\" y1=\"0\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"0.7\"/>\n",
				x, x, cellH, pal["ui.divider"])
			x += 2

			// Pre-attention extras: Qn, Kn, rope_freqs (applied before attention)
			if len(extras) > 0 {
				nExtras := len(extras)
				totalExH := nExtras*subRowH + (nExtras-1)*subRowGap
				exStartY := (cellH - totalExH) / 2
				for ei, en := range extras {
					ey := exStartY + ei*(subRowH+subRowGap)
					slots = append(slots, drawBaseRect(&sb, x, ey, extraW, subRowH, blockType, en))
				}
				x += extraW + 1
				fmt.Fprintf(&sb, "  <line x1=\"%d\" y1=\"0\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"0.7\"/>\n",
					x, x, cellH, pal["ui.divider"])
				x += 2
			}

			// Output projection
			slots = append(slots, drawBaseRect(&sb, x, normY, outW, normH, blockType, WeightAttnOutput))
			x += outW + 1

			// Post-attention norm (applied after output projection, in graph.go)
			if hasPostNorm {
				fmt.Fprintf(&sb, "  <line x1=\"%d\" y1=\"0\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"0.7\"/>\n",
					x, x, cellH, pal["ui.divider"])
				x += 2
				slots = append(slots, drawBaseRect(&sb, x, normY, postNormW, normH, blockType, WeightPostAttnNorm))
				x += postNormW + 1
			}

		} else {
			// Generic: attn_norm as left norm slot, remaining weights as equal sub-rects.
			var normWt string
			remaining := make([]string, 0, len(m.Weights))
			for _, w := range m.Weights {
				if w == WeightAttnNorm {
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

		return symbolDef{id: id, slots: slots, svg: sb.String()}
	}

	// --- buildBlockCompound: compound symbol = <use> of base block + residual return features.
	// Descending (-desc): gap center → down → dashes left to spine + addition indicator.
	// Ascending (-asc): gap center → up → dashes left to spine (over the top).
	buildBlockCompound := func(blockType string, ascending bool) symbolDef {
		suffix := "-desc"
		if ascending {
			suffix = "-asc"
		}
		baseID := "blk-" + blockType
		id := baseID + suffix
		pp := palPrefix(blockType)
		retCol := pal[pp+".stroke"]
		retY := cellH + 4
		if ascending {
			retY = -4
		}
		var sb strings.Builder
		fmt.Fprintf(&sb, "  <use href=\"#%s\"/>\n", baseID)
		fmt.Fprintf(&sb, "  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"0.8\"/>\n",
			blkGapCX, blkGapCY, blkGapCX, retY, retCol)
		fmt.Fprintf(&sb, "  <path d=\"M %d,%d L %d,%d\" fill=\"none\" stroke=\"%s\" stroke-width=\"1.0\" stroke-dasharray=\"2,3\"/>\n",
			blkGapCX, retY, blkSpineRel, retY, retCol)
		fmt.Fprintf(&sb, "  <circle cx=\"%d\" cy=\"%d\" r=\"4\" fill=\"%s\"/>\n",
			blkSpineRel, retY, pal["ui.spine"])
		fmt.Fprintf(&sb, "  <text x=\"%d\" y=\"%d\" text-anchor=\"middle\" dominant-baseline=\"central\" font-size=\"8\" font-weight=\"bold\" fill=\"%s\">+</text>\n",
			blkSpineRel, retY, retCol)
		return symbolDef{id: id, svg: sb.String()}
	}

	// --- buildFFNSymbol: render the SVG body for one FFN type and collect slot geometry.
	// The symbol origin is the FFN cell top-left (0,0). The return rail extends to the
	// right using overflow="visible". Residual dashes are in separate -res-desc/-res-asc symbols.
	buildFFNSymbol := func(sk string, m *Module) symbolDef {
		id := "ffn-" + sk
		isMoE := sk == FFNSymMoE
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
			slots = append(slots, drawBaseRect(&sb, ffnNormXr, normY, ffnNormW, normH, TypeFFN, normWt))
			fmt.Fprintf(&sb, "  <line x1=\"%d\" y1=\"0\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"0.7\"/>\n",
				ffnNDivr, ffnNDivr, cellH, pal["ui.divider"])
		}

		if !isMoE {
			// Dense: gate/up stacked | down
			total2H := 2*subRowH + 1*subRowGap
			guStartY := (cellH - total2H) / 2
			for gi, gn := range []string{WeightFFNGate, WeightFFNUp} {
				gy := guStartY + gi*(subRowH+subRowGap)
				slots = append(slots, drawBaseRect(&sb, ffnGUXr, gy, ffnGUW, subRowH, TypeFFN, gn))
			}
			slots = append(slots, drawBaseRect(&sb, ffnDownXr, normY, ffnDownW, normH, TypeFFN, WeightFFNDown))
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
				case MoEExpertWeights[w]:
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
						slots = append(slots, drawBaseRect(&sb, wx, normY, subW-1, normH, TypeFFNMoE, wn))
					}
				}
			}
		}

		// FFN return rail: stub from right edge to returnRail with arrowhead.
		fmt.Fprintf(&sb, "  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"0.8\"/>\n",
			ffnCellW, cellH/2, ffnReturnRailRel, cellH/2, pal["ui.spine"])
		// Arrowhead at return rail pointing right (output direction)
		fmt.Fprintf(&sb, "  <polygon points=\"%d,%d %d,%d %d,%d\" fill=\"%s\"/>\n",
			ffnReturnRailRel-4, cellH/2-2, ffnReturnRailRel, cellH/2, ffnReturnRailRel-4, cellH/2+2, pal["ui.spine"])

		return symbolDef{id: id, slots: slots, svg: sb.String()}
	}

	// --- buildFFNCompound: compound symbol = <use> of base FFN + residual return features.
	// Solid vertical from return rail to retY, then dashed horizontal to spine + indicator.
	// The vertical segment is solid (not dashed) so the horizontal dashes align with
	// block residual dashes — both start from fresh horizontal paths.
	buildFFNCompound := func(sk string, ascending bool) symbolDef {
		suffix := "-desc"
		if ascending {
			suffix = "-asc"
		}
		baseID := "ffn-" + sk
		id := baseID + suffix
		ffnRetY := cellH + 12
		if ascending {
			ffnRetY = -12
		}
		var sb strings.Builder
		fmt.Fprintf(&sb, "  <use href=\"#%s\"/>\n", baseID)
		// Solid vertical from return rail to residual Y
		fmt.Fprintf(&sb, "  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"1.0\"/>\n",
			ffnReturnRailRel, cellH/2, ffnReturnRailRel, ffnRetY, pal["ffn.stroke"])
		// Dashed horizontal from return rail to spine
		fmt.Fprintf(&sb, "  <path d=\"M %d,%d L %d,%d\" fill=\"none\" stroke=\"%s\" stroke-width=\"1.0\" stroke-dasharray=\"2,3\"/>\n",
			ffnReturnRailRel, ffnRetY, ffnSpineRel, ffnRetY, pal["ffn.stroke"])
		// Residual addition indicator: gray circle with colored +
		fmt.Fprintf(&sb, "  <circle cx=\"%d\" cy=\"%d\" r=\"4\" fill=\"%s\"/>\n",
			ffnSpineRel, ffnRetY, pal["ui.spine"])
		fmt.Fprintf(&sb, "  <text x=\"%d\" y=\"%d\" text-anchor=\"middle\" dominant-baseline=\"central\" font-size=\"8\" font-weight=\"bold\" fill=\"%s\">+</text>\n",
			ffnSpineRel, ffnRetY, pal["ffn.stroke"])
		return symbolDef{id: id, svg: sb.String()}
	}

	// --- Collect unique block/FFN symbols from module list ---
	// Uses the first module of each type to build the symbol; all same-type modules
	// have identical weight structure within a model.
	// Base symbols contain cell content only. Compound -desc/-asc symbols compose
	// <use> of the base + direction-specific residual return features.
	blockSymbols := make(map[string]symbolDef)  // blockType → base
	blockDesc := make(map[string]symbolDef)     // blockType → compound descending
	blockAsc := make(map[string]symbolDef)      // blockType → compound ascending
	ffnSymbols := make(map[string]symbolDef)    // "dense"/"moe" → base
	ffnDesc := make(map[string]symbolDef)       // "dense"/"moe" → compound descending
	ffnAsc := make(map[string]symbolDef)        // "dense"/"moe" → compound ascending
	for i := range mm.Modules {
		m := &mm.Modules[i]
		if strings.HasPrefix(m.Name, PrefixBlock) && m.BlockName != "" {
			if _, seen := blockSymbols[m.BlockName]; !seen {
				blockSymbols[m.BlockName] = buildBlockSymbol(m.BlockName, m)
				blockDesc[m.BlockName] = buildBlockCompound(m.BlockName, false)
				blockAsc[m.BlockName] = buildBlockCompound(m.BlockName, true)
			}
		} else if strings.HasPrefix(m.Name, PrefixFFN) {
			sk := ffnSymKey(m)
			if _, seen := ffnSymbols[sk]; !seen {
				ffnSymbols[sk] = buildFFNSymbol(sk, m)
				ffnDesc[sk] = buildFFNCompound(sk, false)
				ffnAsc[sk] = buildFFNCompound(sk, true)
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
	hasEngagement := engagement != nil && len(engagement.BlockCosSim) > 0
	engageLegH := 0
	if hasEngagement {
		engageLegH = 50 // gap + header + gradient bar + labels
	}
	legBoxH := 10 + legItemCount*16 + engageLegH + 10 // top pad + items + engagement + bottom pad
	hasParams := totalParams > 0
	statsBoxH := 96
	if hasParams {
		statsBoxH += 30
	}
	if hasVRAM {
		statsBoxH += 38
	}

	// --- Build row-chrome symbol (spine dot + connectors + arrowhead) ---
	// Origin at (0, 0) = row center-Y. Placed with translate(0, cy).
	var rowChromeSVG strings.Builder
	fmt.Fprintf(&rowChromeSVG, "  <circle cx=\"%d\" cy=\"0\" r=\"2.5\" fill=\"%s\"/>\n", spineX, pal["ui.dot"])
	fmt.Fprintf(&rowChromeSVG, "  <line x1=\"%d\" y1=\"0\" x2=\"%d\" y2=\"0\" stroke=\"%s\" stroke-width=\"0.8\"/>\n",
		spineX, cellX1, pal["ui.spine"])
	// Input arrowhead at block cell left edge
	fmt.Fprintf(&rowChromeSVG, "  <polygon points=\"%d,-2 %d,0 %d,2\" fill=\"%s\"/>\n",
		cellX1-4, cellX1, cellX1-4, pal["ui.spine"])
	fmt.Fprintf(&rowChromeSVG, "  <line x1=\"%d\" y1=\"0\" x2=\"%d\" y2=\"0\" stroke=\"%s\" stroke-width=\"0.8\"/>\n",
		cellX1+blockCellW, cellX2, pal["ui.spine"])
	// Input arrowhead at FFN cell left edge
	fmt.Fprintf(&rowChromeSVG, "  <polygon points=\"%d,-2 %d,0 %d,2\" fill=\"%s\"/>\n",
		cellX2-4, cellX2, cellX2-4, pal["ui.spine"])
	if nonCausal {
		// Reverse arrowhead at block cell left edge (bidirectional flow)
		fmt.Fprintf(&rowChromeSVG, "  <polygon points=\"%d,-2 %d,0 %d,2\" fill=\"%s\"/>\n",
			cellX1+4, cellX1, cellX1+4, pal["ui.spine"])
	}

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

		// Engagement legend (only when engagement data is provided)
		if hasEngagement {
			ly += legSz + 10
			fmt.Fprintf(&legendSVG, "  <text class=\"shdr\" x=\"%d\" y=\"%d\">engagement (1-cos):</text>\n", legPad, ly)
			ly += 14
			gradW := legBoxW - 2*legPad
			fmt.Fprintf(&legendSVG, "  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"url(#engage-grad)\" stroke=\"%s\" stroke-width=\"0.5\" rx=\"1\"/>\n",
				legPad, ly, gradW, legSz, pal["ui.box_border"])
			fmt.Fprintf(&legendSVG, "  <text class=\"ltxt\" x=\"%d\" y=\"%d\">0</text>\n", legPad+2, ly+legSz+9)
			fmt.Fprintf(&legendSVG, "  <text class=\"ltxt\" x=\"%d\" y=\"%d\" text-anchor=\"end\">1</text>\n", legPad+gradW-2, ly+legSz+9)
		}
	}

	// --- Build summary symbol ---
	// Origin at (0, 0) = box top-left. Placed with translate(legBoxX, statsBoxY).
	var summarySVG strings.Builder
	fmt.Fprintf(&summarySVG, "  <text style=\"font-size:9px;font-weight:bold;fill:%s;font-family:'Courier New',monospace\" x=\"%d\" y=\"-5\">Summary</text>\n", pal["ui.text_head"], legPad)
	fmt.Fprintf(&summarySVG, "  <rect width=\"%d\" height=\"%d\" fill=\"%s\" stroke=\"%s\" stroke-width=\"0.8\" rx=\"2\"/>\n",
		legBoxW, statsBoxH, pal["ui.box_bg"], pal["ui.box_border"])
	{
		sy := 11
		subLine := func(text string) {
			fmt.Fprintf(&summarySVG, "  <text class=\"ltxt\" x=\"%d\" y=\"%d\">  %s</text>\n", legPad, sy, text)
			sy += 12
		}
		sectionHeader := func(text string) {
			fmt.Fprintf(&summarySVG, "  <text class=\"shdr\" x=\"%d\" y=\"%d\">%s</text>\n", legPad, sy, text)
			sy += 13
		}
		sectionGap := func() { sy += 5 }

		sectionHeader(fmt.Sprintf("%d modules", nTotal))
		subLine(fmt.Sprintf("%d culled (%d%%)", nCulled, pct(nCulled, nTotal)))
		sectionGap()
		sectionHeader(fmt.Sprintf("%d tensors", totalTensors))
		subLine(fmt.Sprintf("%d culled (%d%%)", culledTensors, pct(culledTensors, totalTensors)))
		sectionGap()
		sectionHeader(fmt.Sprintf("%d weights", totalWeights))
		subLine(fmt.Sprintf("%d elided (%d%%)", elidedWeights, pct(elidedWeights, totalWeights)))

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
			subLine(fmt.Sprintf("%s inactive (%d%%)", fmtParams(inactiveParams), pct(int(inactiveParams), int(totalParams))))
		}

		if hasVRAM {
			usedMB := fullMB - culledMB
			sectionGap()
			sectionHeader(fmt.Sprintf("%.1f MB vram", fullMB))
			subLine(fmt.Sprintf("%.1f MB used (%d%%)", usedMB, pct(int(usedMB*10), int(fullMB*10))))
			subLine(fmt.Sprintf("%.1f MB culled (%d%%)", culledMB, pct(int(culledMB*10), int(fullMB*10))))
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
	isDiffusion := generation == GenerationDiffusion
	spineY1 := 54
	if nonCausal {
		spineY1 = 72 // extra room for subtitle
	}
	if isDiffusion {
		spineY1 = 90 // extra room for loop-back arrow above title
	}
	firstRowCY := spineY1 + 10
	colRows := leftCount // left column always has >= right column rows
	bottomCY := firstRowCY + (colRows-1)*pitch
	turnY := bottomCY + pitch/2 + 10 // U-turn Y below the bottom row
	if nonCausal {
		turnY = bottomCY + 60 // extra room for bidirectional arrows below last layer
	}

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
  .title { font-size: 20px; font-weight: bold; fill: %s; font-family: system-ui, -apple-system, sans-serif; }
  .hdr   { font-size: 9px; font-weight: bold; fill: %s; text-anchor: middle; dominant-baseline: middle; }
  .lbl   { font-size: 9px; fill: %s; text-anchor: end; dominant-baseline: middle; }
  .tlbl  { font-size: 7px; fill: %s; text-anchor: middle; dominant-baseline: middle; }
  .io    { font-size: 8px; font-weight: bold; fill: %s; text-anchor: middle; dominant-baseline: middle; }
  .gtxt  { font-size: 9px; text-anchor: middle; dominant-baseline: middle; }
  .ltxt  { font-size: 8px; fill: %s; dominant-baseline: middle; }
  .shdr  { font-size: 8px; font-weight: bold; fill: %s; dominant-baseline: middle; }
  .etxt  { font-size: 6px; fill: %s; text-anchor: middle; dominant-baseline: hanging; }
</style>
<defs>
`, canvasW2, canvasH,
		pal["ui.text_head"],
		pal["ui.text_head"],
		pal["ui.text_label"],
		pal["ui.text_tensor"],
		pal["ui.dot"],
		pal["ui.text_body"],
		pal["ui.text_head"],
		pal["ui.text_hint"])

	// Gradients
	for _, name := range []string{TypeFullAttention, TypeSWA, TypeRecurrent, TypeFFN, ModuleGlobal} {
		fmt.Fprintf(&b, "  <linearGradient id=\"zm_%s\" x1=\"0\" y1=\"0\" x2=\"0\" y2=\"1\">\n", name)
		fmt.Fprintf(&b, "    <stop offset=\"0%%%%\" stop-color=\"%s\"/><stop offset=\"100%%%%\" stop-color=\"%s\"/>\n",
			pal[name+".grad_top"], pal[name+".grad_bottom"])
		fmt.Fprintf(&b, "  </linearGradient>\n")
	}
	if hasEngagement {
		fmt.Fprintf(&b, "  <linearGradient id=\"engage-grad\" x1=\"0\" y1=\"0\" x2=\"1\" y2=\"0\">\n")
		fmt.Fprintf(&b, "    <stop offset=\"0\" stop-color=\"%s\" stop-opacity=\"0\"/><stop offset=\"1\" stop-color=\"%s\" stop-opacity=\"0.7\"/>\n",
			pal["engage.hot"], pal["engage.hot"])
		fmt.Fprintf(&b, "  </linearGradient>\n")
	}

	// Block symbols: base, then compound -desc/-asc (which reference the base via <use>)
	for _, bt := range blockTypes {
		for _, sym := range []symbolDef{blockSymbols[bt], blockDesc[bt], blockAsc[bt]} {
			fmt.Fprintf(&b, "  <symbol id=\"%s\" overflow=\"visible\">\n", sym.id)
			fmt.Fprint(&b, sym.svg)
			fmt.Fprintf(&b, "  </symbol>\n")
		}
	}

	// FFN symbols: base, then compound -desc/-asc (which reference the base via <use>)
	for _, fk := range ffnKeys {
		for _, sym := range []symbolDef{ffnSymbols[fk], ffnDesc[fk], ffnAsc[fk]} {
			fmt.Fprintf(&b, "  <symbol id=\"%s\" overflow=\"visible\">\n", sym.id)
			fmt.Fprint(&b, sym.svg)
			fmt.Fprintf(&b, "  </symbol>\n")
		}
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
	fmt.Fprintf(&b, "  <text class=\"title\" x=\"%d\" y=\"32\" text-anchor=\"middle\">%s</text>\n", titleCX, title)
	if nonCausal {
		subtitle := "(bidirectional / non-causal)"
		if generation == GenerationDiffusion {
			subtitle = "(diffusion — iterative denoising, bidirectional)"
		}
		fmt.Fprintf(&b, "  <text x=\"%d\" y=\"46\" text-anchor=\"middle\" font-size=\"11\" font-weight=\"600\" fill=\"%s\">%s</text>\n",
			titleCX, pal["ui.text_hint"], subtitle)
	}

	// U-shaped spine
	rightSpineX := rightColOffset + spineX
	leftSpineBottom := bottomCY + pitch/4
	if nonCausal {
		leftSpineBottom = bottomCY + 48 // extend past last layer's down arrowhead
	}
	rightSpineBottom := leftSpineBottom
	rightSpineTop := globalCY

	// Left spine (descending)
	fmt.Fprintf(&b, "  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"1.5\"/>\n",
		spineX, spineY1, spineX, leftSpineBottom, pal["ui.spine"])
	// IN label + downward arrowhead at spine start
	inLabelY := spineY1 - 12
	if isDiffusion {
		inLabelY = spineY1 - 4 // below the loop-back arrow landing
	}
	fmt.Fprintf(&b, "  <text class=\"io\" x=\"%d\" y=\"%d\">IN</text>\n", spineX, inLabelY)
	if !isDiffusion {
		// Spine start arrowhead — omitted for diffusion where the loop arrow already points down.
		fmt.Fprintf(&b, "  <polygon points=\"%d,%d %d,%d %d,%d\" fill=\"%s\"/>\n",
			spineX-3, spineY1, spineX, spineY1+6, spineX+3, spineY1, pal["ui.spine"])
	}
	if nonCausal && !isDiffusion {
		// Upward arrowhead on left spine (reverse flow) — omitted for diffusion
		// where the loop-back arrow already conveys return flow.
		fmt.Fprintf(&b, "  <polygon points=\"%d,%d %d,%d %d,%d\" fill=\"%s\"/>\n",
			spineX-3, spineY1+18, spineX, spineY1+12, spineX+3, spineY1+18, pal["ui.spine"])
	}

	// U-turn at bottom (left spine → right spine)
	fmt.Fprintf(&b, "  <path d=\"M %d,%d L %d,%d L %d,%d\" fill=\"none\" stroke=\"%s\" stroke-width=\"1.5\"/>\n",
		spineX, leftSpineBottom, spineX, turnY, rightSpineX, turnY, pal["ui.spine"])
	if nonCausal {
		// Bidirectional U-turn: right arrowhead at 1/3, left arrowhead at 2/3
		turnSpan := rightSpineX - spineX
		turnRightX := spineX + turnSpan/3
		turnLeftX := spineX + 2*turnSpan/3
		fmt.Fprintf(&b, "  <polygon points=\"%d,%d %d,%d %d,%d\" fill=\"%s\"/>\n",
			turnRightX-3, turnY-3, turnRightX+3, turnY, turnRightX-3, turnY+3, pal["ui.spine"])
		fmt.Fprintf(&b, "  <polygon points=\"%d,%d %d,%d %d,%d\" fill=\"%s\"/>\n",
			turnLeftX+3, turnY-3, turnLeftX-3, turnY, turnLeftX+3, turnY+3, pal["ui.spine"])
	} else {
		turnMidX := (spineX + rightSpineX) / 2
		fmt.Fprintf(&b, "  <polygon points=\"%d,%d %d,%d %d,%d\" fill=\"%s\"/>\n",
			turnMidX-3, turnY-3, turnMidX+3, turnY, turnMidX-3, turnY+3, pal["ui.spine"])
	}
	fmt.Fprintf(&b, "  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"1.5\"/>\n",
		rightSpineX, turnY, rightSpineX, rightSpineBottom, pal["ui.spine"])

	// Right spine (ascending)
	fmt.Fprintf(&b, "  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"1.5\"/>\n",
		rightSpineX, rightSpineBottom, rightSpineX, rightSpineTop, pal["ui.spine"])
	if nonCausal {
		// Downward arrowhead on right spine (reverse flow)
		fmt.Fprintf(&b, "  <polygon points=\"%d,%d %d,%d %d,%d\" fill=\"%s\"/>\n",
			rightSpineX-3, rightSpineTop+6, rightSpineX, rightSpineTop+12, rightSpineX+3, rightSpineTop+6, pal["ui.spine"])
	}

	// --- Left column layers (descending) ---
	// engageLabel formats an engagement value for the SVG annotation.
	engageLabel := func(cosSim float32) string {
		e := 1 - cosSim
		if e >= 0.01 {
			return fmt.Sprintf("%.2f", e)
		}
		return fmt.Sprintf("%.4f", e)
	}

	emitLayerRow := func(l int, lr *layerEntry, cy, xOff int, ascending, skipSpineArrow bool) {
		rowY := cy - cellH/2
		fmt.Fprintf(&b, "  <use href=\"#row-chrome\" transform=\"translate(%d,%d)\"/>\n", xOff, cy)
		fmt.Fprintf(&b, "  <text class=\"lbl\" x=\"%d\" y=\"%d\">%d</text>\n", xOff+spineX-5, cy, l)

		if lr.hasBlock {
			m := moduleByID[lr.blockID]
			sym := blockSymbols[m.BlockName]
			comp := blockDesc[m.BlockName]
			if ascending {
				comp = blockAsc[m.BlockName]
			}
			fmt.Fprintf(&b, "  <use href=\"#%s\" transform=\"translate(%d,%d)\"/>\n", comp.id, xOff+cellX1, rowY)
			if hasEngagement && l < len(engagement.BlockCosSim) {
				e := float64(1 - engagement.BlockCosSim[l])
				if !math.IsNaN(e) {
					if op := engageOpacity(e); op > 0 {
						for _, slot := range sym.slots {
							fmt.Fprintf(&b, "  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"%s\" fill-opacity=\"%.3f\" rx=\"1\" pointer-events=\"none\"/>\n",
								xOff+cellX1+slot.x, rowY+slot.y, slot.w, slot.h, pal["engage.hot"], op)
						}
					}
					// Engagement value annotation below cell
					fmt.Fprintf(&b, "  <text class=\"etxt\" x=\"%d\" y=\"%d\">%s</text>\n",
						xOff+cellX1+blockCellW/2, rowY+cellH+8, engageLabel(engagement.BlockCosSim[l]))
				}
			}
		}

		if lr.hasFfn {
			m := moduleByID[lr.ffnID]
			sk := ffnSymKey(m)
			sym := ffnSymbols[sk]
			ffnComp := ffnDesc[sk]
			if ascending {
				ffnComp = ffnAsc[sk]
			}
			fmt.Fprintf(&b, "  <use href=\"#%s\" transform=\"translate(%d,%d)\"/>\n", ffnComp.id, xOff+cellX2, rowY)
			if hasEngagement && l < len(engagement.FFNCosSim) {
				e := float64(1 - engagement.FFNCosSim[l])
				if !math.IsNaN(e) {
					if op := engageOpacity(e); op > 0 {
						for _, slot := range sym.slots {
							fmt.Fprintf(&b, "  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"%s\" fill-opacity=\"%.3f\" rx=\"1\" pointer-events=\"none\"/>\n",
								xOff+cellX2+slot.x, rowY+slot.y, slot.w, slot.h, pal["engage.hot"], op)
						}
					}
					// Engagement value annotation below cell
					fmt.Fprintf(&b, "  <text class=\"etxt\" x=\"%d\" y=\"%d\">%s</text>\n",
						xOff+cellX2+ffnCellW/2, rowY+cellH+8, engageLabel(engagement.FFNCosSim[l]))
				}
			}
		}

		// Spine directional arrow between layers.
		// Arrow body centered midway between FFN residual return and next layer's tap.
		if !skipSpineArrow {
			sx := xOff + spineX
			if nonCausal {
				// Bidirectional: arrows hugging the circle-+ indicators.
				// Up arrow above the top circle-+, down arrow below the bottom circle-+.
				const circR = 4 // circle-+ radius
				const gap = 2  // gap between circle edge and arrow base
				if ascending {
					// Top circle-+ is FFN at cy - 32, bottom is block at cy - 24
					upTip := cy - 32 - circR - gap - 6 // tip above FFN circle
					dnTip := cy - 24 + circR + gap + 6 // tip below block circle
					fmt.Fprintf(&b, "  <polygon points=\"%d,%d %d,%d %d,%d\" fill=\"%s\"/>\n",
						sx-3, upTip+6, sx, upTip, sx+3, upTip+6, pal["ui.spine"])
					fmt.Fprintf(&b, "  <polygon points=\"%d,%d %d,%d %d,%d\" fill=\"%s\"/>\n",
						sx-3, dnTip-6, sx, dnTip, sx+3, dnTip-6, pal["ui.spine"])
				} else {
					// Top circle-+ is block at cy + 24, bottom is FFN at cy + 32
					upTip := cy + 24 - circR - gap - 6 // tip above block circle
					dnTip := cy + 32 + circR + gap + 6 // tip below FFN circle
					fmt.Fprintf(&b, "  <polygon points=\"%d,%d %d,%d %d,%d\" fill=\"%s\"/>\n",
						sx-3, upTip+6, sx, upTip, sx+3, upTip+6, pal["ui.spine"])
					fmt.Fprintf(&b, "  <polygon points=\"%d,%d %d,%d %d,%d\" fill=\"%s\"/>\n",
						sx-3, dnTip-6, sx, dnTip, sx+3, dnTip-6, pal["ui.spine"])
				}
			} else {
				// Autoregressive: single arrow midway between FFN return and next tap.
				ffnReturnOfs := cellH/2 + 12
				arrowCenter := (ffnReturnOfs + pitch) / 2
				const arrowHalf = 3
				if ascending {
					arrowY := cy - arrowCenter - arrowHalf
					fmt.Fprintf(&b, "  <polygon points=\"%d,%d %d,%d %d,%d\" fill=\"%s\"/>\n",
						sx-3, arrowY+6, sx, arrowY, sx+3, arrowY+6, pal["ui.spine"])
				} else {
					arrowY := cy + arrowCenter + arrowHalf
					fmt.Fprintf(&b, "  <polygon points=\"%d,%d %d,%d %d,%d\" fill=\"%s\"/>\n",
						sx-3, arrowY-6, sx, arrowY, sx+3, arrowY-6, pal["ui.spine"])
				}
			}
		}
	}

	for i, l := range leftLayers {
		cy := firstRowCY + i*pitch
		emitLayerRow(l, layerMap[l], cy, 0, false, i == len(leftLayers)-1 && !nonCausal)
	}

	// --- Right column layers (ascending — rightLayers[0] at bottom, last at top) ---
	for j, l := range rightLayers {
		cy := bottomCY - j*pitch
		emitLayerRow(l, layerMap[l], cy, rightColOffset, true, false)
	}

	// Global row at top of right column
	fmt.Fprintf(&b, "  <use href=\"#global-row\" transform=\"translate(%d,%d)\"/>\n", rightColOffset, globalCY)

	// Diffusion iteration loop: curved return arrow from OUT back to IN
	if isDiffusion && globalModule != nil {
		gw := blockCellW + cellGap + ffnCellW
		outAbsX := rightColOffset + cellX1 + gw/2
		outAbsY := globalCY - cellH/2 - 28 // above OUT label to avoid transecting it
		inAbsY := spineY1 - 12             // just above IN label
		loopY := 54                         // horizontal segment Y (below subtitle at y=46)
		// Squared path from OUT up, across, and down to IN.
		fmt.Fprintf(&b, "  <path d=\"M %d,%d L %d,%d L %d,%d L %d,%d\" fill=\"none\" stroke=\"%s\" stroke-width=\"1.2\" stroke-dasharray=\"4,3\"/>\n",
			outAbsX, outAbsY, outAbsX, loopY, spineX, loopY, spineX, inAbsY, pal["ui.spine"])
		// Arrowhead pointing down at IN
		fmt.Fprintf(&b, "  <polygon points=\"%d,%d %d,%d %d,%d\" fill=\"%s\"/>\n",
			spineX-3, inAbsY-6, spineX, inAbsY, spineX+3, inAbsY-6, pal["ui.spine"])
		// Label below the horizontal segment, right-justified near the output vertical
		fmt.Fprintf(&b, "  <text x=\"%d\" y=\"%d\" text-anchor=\"end\" font-size=\"8\" font-style=\"italic\" fill=\"%s\">unmask &amp; re-run (T steps)</text>\n",
			outAbsX-4, loopY+10, pal["ui.text_hint"])
	}

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
	return has[WeightAttnQ] && has[WeightAttnK] && has[WeightAttnV] && has[WeightAttnOutput]
}
