package archdiagram

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"strings"

	"inference-lab-bench/internal/inference/arch"
)

// vision diagram layout constants (kept local; the arch diagram's are in its
// own function and not exported).
const (
	visWidth    = 960
	visMargin   = 30
	visArrowLen = 20
	visStageW   = 460 // single-stage box width
	visNormH    = 34
	visTitleH   = 22 // vertical band for the box title
	visLineH    = 14 // per content line (weight / hint)
	visBoxPad   = 10 // bottom padding inside a box
)

// RenderVisionDiagram writes an SVG for a multimodal model's vision tower
// (image → soft tokens spliced into the decoder embedding stream). Every
// per-arch difference — norm type, patch-conv count, projector layer count,
// decoder bidirectional span — is read from def.Vision / def.Projector, not
// branched on architecture name. Mirrors RenderArchDiagram's vertical-cursor,
// buffer-then-size structure and shares its palette and helpers.
func RenderVisionDiagram(def *arch.ArchDef, w io.Writer) error {
	if def.Vision == nil {
		return fmt.Errorf("RenderVisionDiagram: def has no [vision] section")
	}
	v := def.Vision

	bw := bufio.NewWriter(w)
	var bodyBuf bytes.Buffer
	body := bufio.NewWriter(&bodyBuf)

	emitGradientDefs(body)
	emitImageGradient(body)

	// Title
	cx := visWidth / 2
	title := capitalizeASCII(def.Architecture.Name) + " Vision Tower"
	fmt.Fprintf(body, "  <text x=\"%d\" y=\"32\" text-anchor=\"middle\" font-size=\"20\" font-weight=\"bold\" fill=\"%s\">%s</text>\n",
		cx, pal["ui.text_head"], title)

	cursor := 60

	// 1. Image input — the raw RGB source. It's unconstrained: patch_size /
	// n_merge are downstream (Patch Align) properties, not properties of the
	// source image, so they don't belong here. Only surface a fixed input size
	// if the arch declares one (towers here smart-resize, so it's arbitrary).
	imgInfo := "raw RGB · arbitrary size"
	if v.InputSize > 0 {
		imgInfo = fmt.Sprintf("raw RGB · fixed %d px", v.InputSize)
	}
	cursor = emitVisionStage(body, cx, cursor, "image", "Image", nil, lines(imgInfo))
	cursor = emitVisionArrow(body, cx, cursor)

	// 2. Preprocess — describe the actual operations (resize/pad/normalize the
	// raw image into a model-ready, patch-aligned tensor) rather than the opaque
	// "Preprocess"; surface a named strategy when the arch declares one.
	preHints := []string{"aspect-resize to patch×merge multiple + pad"}
	if v.Preprocessing != "" {
		preHints = append(preHints, "strategy: "+formatBuilderName(v.Preprocessing))
	}
	preInfo := joinNonEmpty(" · ",
		labelIfPos("patch", v.PatchSize),
		labelIfPos("merge", v.NMerge),
		softTokenBounds(v))
	if preInfo != "" {
		preHints = append(preHints, preInfo)
	}
	cursor = emitVisionStage(body, cx, cursor, "norm", "Patch Align", nil, preHints)
	cursor = emitVisionArrow(body, cx, cursor)

	// 3. Patch embed — concise stage (conv count + position-embed presence).
	// The full tensor list lives in the layers diagram; here we only summarize.
	cursor = emitVisionStage(body, cx, cursor, "global", "Patch Embed",
		nil, patchEmbedHints(v))
	cursor = emitVisionArrow(body, cx, cursor)

	// 4. Encoder layer group (dashed box, "× block_count").
	cursor = emitEncoderGroup(body, def, cx, cursor)
	cursor = emitVisionArrow(body, cx, cursor)

	// 5. Post-LN. Gemma carries its post-encoder norms per-layer (no global
	// post_ln), so this stage is data-driven absent for it and present for Qwen.
	postWeights := visionGlobalWeights(v, postLNKeys)
	if len(postWeights) > 0 {
		cursor = emitVisionStage(body, cx, cursor, "norm",
			"Post-Encoder "+normTypeLabel(v.NormType), nil, nil)
		cursor = emitVisionArrow(body, cx, cursor)
	}

	// 6. Spatial merge. The verb is data-driven from the projector type: an mlp
	// projector concat+MLP-merges patches ("merge"); a pool-style projector
	// (linear_post_norm) averages them ("pool"). Never branch on arch name.
	mergeInfo := "(no spatial merge)"
	if v.NMerge > 1 {
		mergeInfo = fmt.Sprintf("%s n_merge=%d (×%d tokens merged)",
			spatialMergeVerb(def.Projector), v.NMerge, v.NMerge*v.NMerge)
	}
	cursor = emitVisionStage(body, cx, cursor, "swa", "Spatial Merge", nil, lines(mergeInfo))
	cursor = emitVisionArrow(body, cx, cursor)

	// 7. Projector — concise type + shape hint (full mm.* tensors are in the
	// layers diagram). E.g. "linear + post-norm" / "2-layer MLP".
	if def.Projector != nil {
		cursor = emitVisionStage(body, cx, cursor, "ffn",
			"Projector: "+formatBuilderName(def.Projector.Type), nil, projectorHints(def.Projector))
		cursor = emitVisionArrow(body, cx, cursor)
	}

	// 8. Soft tokens → splice into decoder.
	spliceLines := []string{softTokenLabel(v)}
	if v.ImageToken != "" {
		spliceLines = append(spliceLines, "placeholder: "+v.ImageToken)
	}
	spliceLines = append(spliceLines, decoderSpanLabel(v))
	cursor = emitVisionStage(body, cx, cursor, "global", "Splice into Decoder", nil, spliceLines)

	cursor += 20

	// Emit <svg> header + background, then body.
	finalHeight := cursor
	fmt.Fprintf(bw, `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" font-family="system-ui, -apple-system, sans-serif" font-size="13">`, visWidth, finalHeight)
	bw.WriteString("\n")
	fmt.Fprintf(bw, "  <rect width=\"%d\" height=\"%d\" fill=\"%s\" rx=\"8\"/>\n\n", visWidth, finalHeight, pal["ui.canvas_bg"])
	body.Flush()
	bw.Write(bodyBuf.Bytes())
	fmt.Fprintf(bw, "</svg>\n")
	return bw.Flush()
}

// emitEncoderGroup draws the dashed "encoder layer × block_count" box with its
// inner Norm → Attention → Norm → FFN stack, all sourced from the [vision]
// blocks/ffn/layers data.
func emitEncoderGroup(body *bufio.Writer, def *arch.ArchDef, cx, cursor int) int {
	v := def.Vision
	stroke := pal["full_attention.stroke"]

	// Resolve the encoder block + FFN from the (uniform) vision routing.
	blkName := v.Layers.Routing.Uniform
	if blkName == "" {
		blkName = v.Layers.Routing.IfTrue // tolerate rule/pattern routing if a tower ever uses it
	}
	blk := v.Blocks[blkName]

	normKind := normTypeLabel(v.NormType)

	// Overview: concise stage labels only (no per-tensor enumeration — that
	// lives in the layers diagram). Attention summary reads its rope / qk-norm
	// / fused-vs-separate QKV from config; FFN summary reads type + activation.
	attnHint := lines(attnSummary(blk))
	ffnHint := lines(ffnSummary(v.FFN))

	attnH := visionBoxHeight(nil, attnHint)
	ffnH := visionBoxHeight(nil, ffnHint)

	// Compute group height from its contents, then emit the border first.
	innerGap := 12
	groupTop := cursor
	groupContentH := 28 + // group label band
		visNormH + visArrowLen +
		attnH + visArrowLen +
		visNormH + visArrowLen +
		ffnH + innerGap
	groupW := visWidth - 2*visMargin

	fmt.Fprintf(body, "  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" rx=\"8\" fill=\"none\" stroke=\"%s\" stroke-width=\"1.5\" stroke-dasharray=\"6,3\" opacity=\"0.6\"/>\n",
		visMargin, groupTop, groupW, groupContentH, stroke)
	fmt.Fprintf(body, "  <text x=\"%d\" y=\"%d\" font-size=\"11\" font-weight=\"600\" fill=\"%s\">Encoder layer × %s</text>\n",
		visMargin+12, groupTop+20, stroke, blockCountLabel(v))

	c := groupTop + 28

	// Pre-attention norm (label only — the tensor names are in the layers diagram).
	emitVisionNormBox(body, centerX(cx, visStageW), c, visStageW, visNormH, normKind, "pre-attention")
	c += visNormH
	c = emitVisionArrow(body, cx, c)

	// Attention.
	emitVisionStageBox(body, centerX(cx, visStageW), c, visStageW, attnH, "full_attention",
		"Attention: "+formatBuilderName(blk.Builder), nil, attnHint)
	c += attnH
	c = emitVisionArrow(body, cx, c)

	// Pre-FFN norm.
	emitVisionNormBox(body, centerX(cx, visStageW), c, visStageW, visNormH, normKind, "pre-FFN")
	c += visNormH
	c = emitVisionArrow(body, cx, c)

	// FFN.
	emitVisionStageBox(body, centerX(cx, visStageW), c, visStageW, ffnH, "ffn",
		"FFN: "+formatBuilderName(v.FFN.Builder), nil, ffnHint)

	return groupTop + groupContentH
}

// --- stage box helpers ---

// emitVisionStage draws a centered titled box at the cursor and returns the
// cursor advanced past it. mono lines render monospace (tensor names), hint
// lines render small grey (config/notes). The box grows to fit its content,
// so long weight lists never overflow.
func emitVisionStage(body *bufio.Writer, cx, cursor int, palKey, title string, mono, hint []string) int {
	h := visionBoxHeight(mono, hint)
	emitVisionStageBox(body, centerX(cx, visStageW), cursor, visStageW, h, palKey, title, mono, hint)
	return cursor + h
}

// visionBoxHeight computes a box height that fits the title plus all content
// lines. Kept in one place so the cursor advance and the drawn rect agree.
func visionBoxHeight(mono, hint []string) int {
	return visTitleH + visLineH*(len(mono)+len(hint)) + visBoxPad
}

// emitVisionStageBox draws a titled box with monospace weight lines and grey
// hint lines at an explicit position. Shared by top-level stages and the
// encoder group's inner attention/FFN boxes.
func emitVisionStageBox(body *bufio.Writer, x, y, width, height int, palKey, title string, mono, hint []string) {
	gradID := visionGradID(palKey)
	fmt.Fprintf(body, "  <g transform=\"translate(%d, %d)\">\n", x, y)
	fmt.Fprintf(body, "    <rect width=\"%d\" height=\"%d\" rx=\"6\" fill=\"url(#%s)\" stroke=\"%s\" stroke-width=\"1.2\" filter=\"url(#shadow)\"/>\n",
		width, height, gradID, pal[palKey+".stroke"])
	fmt.Fprintf(body, "    <text x=\"%d\" y=\"16\" text-anchor=\"middle\" font-weight=\"600\" fill=\"%s\">%s</text>\n",
		width/2, pal["ui.text_head"], xmlEsc(title))
	ty := visTitleH + visLineH - 4
	for _, ln := range mono {
		fmt.Fprintf(body, "    <text x=\"%d\" y=\"%d\" text-anchor=\"middle\" font-size=\"10\" font-family=\"monospace\" fill=\"%s\">%s</text>\n",
			width/2, ty, pal["ui.text_sub"], xmlEsc(ln))
		ty += visLineH
	}
	for _, ln := range hint {
		fmt.Fprintf(body, "    <text x=\"%d\" y=\"%d\" text-anchor=\"middle\" font-size=\"9\" fill=\"%s\">%s</text>\n",
			width/2, ty, pal["ui.text_hint"], xmlEsc(ln))
		ty += visLineH
	}
	body.WriteString("  </g>\n")
}

// emitVisionNormBox draws a norm box labeled per NormType (RMSNorm vs LayerNorm),
// the data-driven analogue of the RMS-only emitRMSNormBox.
func emitVisionNormBox(body *bufio.Writer, x, y, width, height int, kind, paramName string) {
	fmt.Fprintf(body, "  <g transform=\"translate(%d, %d)\">\n", x, y)
	fmt.Fprintf(body, "    <rect width=\"%d\" height=\"%d\" rx=\"5\" fill=\"url(#norm)\" stroke=\"%s\" stroke-width=\"1\" filter=\"url(#shadow)\"/>\n",
		width, height, pal["norm.stroke"])
	fmt.Fprintf(body, "    <text x=\"%d\" y=\"14\" text-anchor=\"middle\" font-weight=\"600\" fill=\"%s\">%s</text>\n",
		width/2, pal["norm.text"], xmlEsc(kind))
	fmt.Fprintf(body, "    <text x=\"%d\" y=\"27\" text-anchor=\"middle\" font-size=\"10\" font-family=\"monospace\" fill=\"%s\">%s</text>\n",
		width/2, pal["ui.text_sub"], xmlEsc(paramName))
	body.WriteString("  </g>\n")
}

func emitVisionArrow(body *bufio.Writer, cx, cursor int) int {
	emitArrow(body, cx, cursor, visArrowLen)
	return cursor + visArrowLen
}

// visionGradID maps a palette family to the gradient id emitted by
// emitGradientDefs (no id prefix). All vision palette families have a gradient.
func visionGradID(palKey string) string { return palKey }

// emitImageGradient writes the "image" gradient used by the Image input box: a
// portable approximation of a polychromatic RGB image. A true 4-corner mesh
// gradient (r/g/b/white at the corners) needs SVG2 mesh gradients, which
// rsvg-convert and most renderers do not support — so this is a diagonal
// linearGradient red→green→blue→white. The four color stops live in pal as the
// image.grad_0..image.grad_3 ramp (like every other gradient's colors); only the
// offset and opacity are geometry, kept inline here. The id is "image" to match
// visionGradID("image").
func emitImageGradient(w *bufio.Writer) {
	w.WriteString("  <defs>\n")
	w.WriteString("    <linearGradient id=\"image\" x1=\"0\" y1=\"0\" x2=\"1\" y2=\"1\">\n")
	fmt.Fprintf(w, "      <stop offset=\"0%%\" stop-color=\"%s\" stop-opacity=\"0.55\"/>", pal["image.grad_0"])
	fmt.Fprintf(w, "<stop offset=\"40%%\" stop-color=\"%s\" stop-opacity=\"0.55\"/>", pal["image.grad_1"])
	fmt.Fprintf(w, "<stop offset=\"75%%\" stop-color=\"%s\" stop-opacity=\"0.55\"/>", pal["image.grad_2"])
	fmt.Fprintf(w, "<stop offset=\"100%%\" stop-color=\"%s\" stop-opacity=\"0.55\"/>\n", pal["image.grad_3"])
	w.WriteString("    </linearGradient>\n")
	w.WriteString("  </defs>\n")
}

// --- data extraction (no architecture branches) ---

// Logical key sets the vision diagram looks for. These are the conventional
// [vision.weights.global] / [vision.layers.common_weights] keys; whichever are
// present render, the rest are silently absent — so a single-conv tower (Gemma)
// and a dual-conv tower (Qwen) both render correctly from their own data.
var (
	patchEmbedConvKeys = []string{"patch_embd", "patch_embd_1"}
	positionEmbedKeys  = []string{"position_embd"}
	postLNKeys         = []string{"post_ln", "post_ln_bias"}
)

// visionGlobalWeights returns the GGUF tensor names for the present keys in
// def.Vision.Weights.Global, in the given key order.
func visionGlobalWeights(v *arch.VisionDef, keys []string) []string {
	var out []string
	for _, k := range keys {
		if name := v.Weights.Global[k]; name != "" {
			out = append(out, name)
		}
	}
	return out
}

// attnSummary renders a concise one-line attention descriptor for the overview
// from the encoder block's config (rope kind, qk-norm, fused-vs-separate QKV) —
// no per-tensor enumeration. All values come from data, none branch on arch.
func attnSummary(blk arch.BlockDef) string {
	var parts []string
	if rope, ok := blk.Config["rope"].(string); ok && rope != "" {
		parts = append(parts, formatBuilderName(rope)+" RoPE")
	}
	if qk, ok := blk.Config["qk_norm"].(string); ok && qk != "" {
		parts = append(parts, "QK-norm")
	}
	if fused, _ := blk.Config["qkv_fused"].(bool); fused {
		parts = append(parts, "fused QKV")
	} else {
		parts = append(parts, "separate Q/K/V")
	}
	return strings.Join(parts, " · ")
}

// ffnSummary renders a concise one-line FFN descriptor for the overview. The
// builder name is already in the box title, so the hint only adds the explicit
// activation when one is configured (e.g. the mlp builder's "gelu"); for
// builders whose activation is implied by their name (geglu_quick) it returns
// "" so no redundant line is drawn.
func ffnSummary(ffn arch.FFNDef) string {
	if act, ok := ffn.Config["activation"].(string); ok && act != "" {
		return "activation: " + act
	}
	return ""
}

// patchEmbedHints summarizes the patch-embed stage: how many conv kernels and
// whether a learned position grid is present — derived from which weight keys
// are populated, no arch branch.
func patchEmbedHints(v *arch.VisionDef) []string {
	nConv := len(visionGlobalWeights(v, patchEmbedConvKeys))
	hint := "conv patch embed"
	if nConv > 1 {
		hint = fmt.Sprintf("%d-conv patch embed", nConv)
	}
	out := []string{hint}
	if len(visionGlobalWeights(v, positionEmbedKeys)) > 0 {
		out = append(out, "+ learned position grid")
	}
	return out
}

// projectorHints summarizes the projector shape (layer count + whether it
// carries a post-projection norm) from its weight map — no arch branch.
func projectorHints(p *arch.ProjectorDef) []string {
	nProj := 0
	for k := range p.Weights {
		if strings.HasSuffix(k, "_bias") {
			continue
		}
		if strings.Contains(k, "norm") {
			continue
		}
		nProj++
	}
	switch {
	case nProj <= 1 && strings.Contains(p.Type, "norm"):
		return []string{"linear + post-norm"}
	case nProj <= 1:
		return []string{"single linear"}
	default:
		return []string{fmt.Sprintf("%d-layer MLP", nProj)}
	}
}

// spatialMergeVerb returns the verb describing how the tower reduces patch
// tokens, derived from the projector type (data, not arch name): an mlp
// projector concat+MLP-merges ("merge"); a pool-style projector averages
// ("pool"). Defaults to the neutral "merge" when no projector is declared.
func spatialMergeVerb(p *arch.ProjectorDef) string {
	if p != nil && p.Type == arch.ProjectorMLP {
		return "merge"
	}
	if p != nil && p.Type == arch.ProjectorLinearPostNorm {
		return "pool"
	}
	return "merge"
}

// normTypeLabel renders the norm box title for the tower's norm_type.
func normTypeLabel(normType string) string {
	if normType == "layernorm" {
		return "LayerNorm"
	}
	return "RMSNorm" // default (Gemma 4)
}

// blockCountLabel renders the encoder repeat count. There is no GGUF here, so
// the count is symbolic — the param the layer count resolves from.
func blockCountLabel(v *arch.VisionDef) string {
	if ref := v.Params.Keys[v.Layers.Count]; ref != "" {
		return fmt.Sprintf("%s (%s)", v.Layers.Count, ref)
	}
	if v.Layers.Count != "" {
		return v.Layers.Count
	}
	return "block_count"
}

// softTokenBounds renders the smart-resize soft-token bounds, if present.
func softTokenBounds(v *arch.VisionDef) string {
	if v.ImageMinTokens > 0 || v.ImageMaxTokens > 0 {
		return fmt.Sprintf("soft tokens %d..%d", v.ImageMinTokens, v.ImageMaxTokens)
	}
	return ""
}

// softTokenLabel describes the soft-token count produced per image.
func softTokenLabel(v *arch.VisionDef) string {
	if v.NImageTokens > 0 {
		return fmt.Sprintf("%d soft tokens / image", v.NImageTokens)
	}
	return "variable soft tokens / image (smart-resize)"
}

// decoderSpanLabel notes whether the spliced image span is bidirectional.
func decoderSpanLabel(v *arch.VisionDef) string {
	if v.DecoderNonCausal {
		return "decoder_non_causal: bidirectional image span"
	}
	return "causal image span (default)"
}

// --- small string helpers ---

func labelIfPos(name string, v int) string {
	if v > 0 {
		return fmt.Sprintf("%s=%d", name, v)
	}
	return ""
}

func joinNonEmpty(sep string, parts ...string) string {
	var kept []string
	for _, p := range parts {
		if p != "" {
			kept = append(kept, p)
		}
	}
	return strings.Join(kept, sep)
}

// lines wraps a non-empty string into a one-element slice for the extra-lines
// parameter; returns nil for empty so no blank line is drawn.
func lines(s string) []string {
	if s == "" {
		return nil
	}
	return []string{s}
}
