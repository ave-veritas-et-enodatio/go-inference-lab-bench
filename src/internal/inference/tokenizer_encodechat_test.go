package inference

import (
	"strings"
	"testing"

	"github.com/nikolalohinski/gonja/v2"
	"github.com/nikolalohinski/gonja/v2/exec"
)

// synthChatTemplate mirrors the structural shape of the real Gemma-4
// multimodal chat template — loop over messages, wrap each turn in
// role-tagged markers, map role 'assistant' to 'model', iterate typed parts
// emitting an image placeholder for {type:'image'} and trimmed text for
// {type:'text'}, and append a trailing generation prompt — without depending
// on any multi-GB GGUF. It is deliberately minimal: just enough to assert
// turn order, role substitution, and image-placeholder placement
// deterministically.
//
// Markers are chosen so they round-trip as dedicated special tokens through
// encodeWithSpecials (see newSynthTokenizer).
const synthChatTemplate = `{%- for message in messages -%}` +
	`<|turn|>{% if message['role'] == 'assistant' %}model{% else %}{{ message['role'] }}{% endif %}` + "\n" +
	`{% if message['content'] is string -%}` +
	`{{- message['content'] | trim -}}` +
	`{%- else -%}` +
	`{%- for part in message['content'] -%}` +
	`{%- if part['type'] == 'image' -%}<|image|>{%- else -%}{{- part['text'] | trim -}}{%- endif -%}` +
	`{%- endfor -%}` +
	`{%- endif -%}` +
	`<|endturn|>` +
	`{%- endfor -%}` +
	`{%- if add_generation_prompt -%}<|turn|>model` + "\n" + `{% endif %}`

// newSynthTokenizer builds an in-package Tokenizer backed by the synthetic
// chat template above and a tiny vocab. The vocab registers the turn/image
// markers as special tokens (so encodeWithSpecials maps each to a known ID)
// plus a handful of byte-level word tokens used by the test prompts. Anything
// not in the vocab simply contributes no token — the special-token IDs and
// their ordering are what the token-level assertions check.
func newSynthTokenizer(t *testing.T) *Tokenizer {
	t.Helper()
	tpl, err := gonja.FromString(synthChatTemplate)
	if err != nil {
		t.Fatalf("synthetic template compile: %v", err)
	}

	// Special markers — must be registered both as specials and in the ID map.
	specials := []string{"<|turn|>", "<|endturn|>", "<|image|>"}
	tok := &Tokenizer{
		tokenIDs:      map[string]int32{},
		mergeRank:     map[[2]string]int{},
		specialTokens: append([]string(nil), specials...),
		bos:           -1,
		eos:           -1,
		maskTokenID:   -1,
		chatTpl:       tpl,
		useByteLevel:  true,
	}
	// encodeWithSpecials matches longest-first; keep specials sorted that way
	// to mirror the production invariant.
	for i, s := range tok.specialTokens {
		tok.tokenIDs[s] = int32(100 + i) // turn=100, endturn=101, image=102
	}
	return tok
}

const (
	tokTurn    = int32(100)
	tokEndturn = int32(101)
	tokImage   = int32(102)
)

func countTok(ids []int32, want int32) int {
	n := 0
	for _, id := range ids {
		if id == want {
			n++
		}
	}
	return n
}

// renderTemplateString renders just the template (no encode) for string-level
// assertions. It duplicates the reserved-var setup EncodeChat performs so the
// rendered text is exactly what EncodeChat would tokenize.
func renderTemplateString(t *testing.T, tok *Tokenizer, msgs []ChatMessage) string {
	t.Helper()
	wire := make([]map[string]any, len(msgs))
	for i, m := range msgs {
		var content any
		if m.Parts != nil {
			parts := make([]any, len(m.Parts))
			for j, p := range m.Parts {
				parts[j] = map[string]any{"type": p.Type, "text": p.Text}
			}
			content = parts
		} else {
			content = m.Content
		}
		wire[i] = map[string]any{"role": m.Role, "content": content}
	}
	out, err := tok.chatTpl.ExecuteToString(exec.NewContext(map[string]any{
		"messages":              wire,
		"add_generation_prompt": true,
	}))
	if err != nil {
		t.Fatalf("template render: %v", err)
	}
	return out
}

// TestEncodeChat_MultiTurnRendering covers a user→assistant→user conversation:
// all three turns must render in order, the assistant role must map to 'model',
// each turn must be bounded by the open/close markers, and the trailing
// generation prompt (<|turn|>model) must be appended. A dropped turn or a
// reordered loop changes the marker count or the role sequence and fails here.
func TestEncodeChat_MultiTurnRendering(t *testing.T) {
	tok := newSynthTokenizer(t)
	msgs := []ChatMessage{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi there"},
		{Role: "user", Content: "bye"},
	}

	out := renderTemplateString(t, tok, msgs)

	// Turn count: 3 conversation turns + 1 generation-prompt turn.
	if got := strings.Count(out, "<|turn|>"); got != 4 {
		t.Errorf("expected 4 <|turn|> markers (3 turns + gen prompt), got %d\n---\n%s", got, out)
	}
	if got := strings.Count(out, "<|endturn|>"); got != 3 {
		t.Errorf("expected 3 <|endturn|> markers, got %d\n---\n%s", got, out)
	}

	// Role sequence: the role labels must appear in conversation order, with
	// assistant rewritten to model, followed by the trailing model prompt.
	wantRoles := []string{"user", "model", "user", "model"}
	idx := 0
	for _, r := range wantRoles {
		marker := "<|turn|>" + r + "\n"
		found := strings.Index(out[idx:], marker)
		if found < 0 {
			t.Fatalf("role marker %q not found in order at/after offset %d\n---\n%s", marker, idx, out)
		}
		idx += found + len(marker)
	}
	// 'assistant' must never appear verbatim — it must be rewritten to 'model'.
	if strings.Contains(out, "assistant") {
		t.Errorf("role 'assistant' leaked into output; should map to 'model'\n---\n%s", out)
	}

	// Content must render in order.
	hPos := strings.Index(out, "hello")
	aPos := strings.Index(out, "hi there")
	bPos := strings.Index(out, "bye")
	if !(hPos >= 0 && aPos > hPos && bPos > aPos) {
		t.Errorf("turn content out of order: hello@%d hi-there@%d bye@%d\n---\n%s", hPos, aPos, bPos, out)
	}

	// EncodeChat itself must succeed end-to-end and emit the markers as their
	// dedicated special-token IDs in the same 4/3 counts.
	ids, err := tok.EncodeChat(msgs, nil)
	if err != nil {
		t.Fatalf("EncodeChat: %v", err)
	}
	if n := countTok(ids, tokTurn); n != 4 {
		t.Errorf("tokenized <|turn|> count = %d, want 4", n)
	}
	if n := countTok(ids, tokEndturn); n != 3 {
		t.Errorf("tokenized <|endturn|> count = %d, want 3", n)
	}
}

// TestEncodeChat_MixedTextImageParts covers the assistant/model role branch
// together with interleaved [text, image, text] parts: the parts must emit in
// order, producing exactly one image placeholder between the two text spans.
// A reordered inner loop or a collapsed parts list would change placeholder
// placement or count.
func TestEncodeChat_MixedTextImageParts(t *testing.T) {
	tok := newSynthTokenizer(t)
	msgs := []ChatMessage{
		{Role: "user", Parts: []ChatContentPart{
			{Type: "text", Text: "before"},
			{Type: "image"},
			{Type: "text", Text: "after"},
		}},
		{Role: "assistant", Content: "ok"},
	}

	out := renderTemplateString(t, tok, msgs)

	// Exactly one image placeholder, sitting between the two text spans.
	if got := strings.Count(out, "<|image|>"); got != 1 {
		t.Fatalf("expected exactly 1 <|image|>, got %d\n---\n%s", got, out)
	}
	bPos := strings.Index(out, "before")
	iPos := strings.Index(out, "<|image|>")
	aPos := strings.Index(out, "after")
	if !(bPos >= 0 && iPos > bPos && aPos > iPos) {
		t.Errorf("parts out of order: before@%d image@%d after@%d\n---\n%s", bPos, iPos, aPos, out)
	}
	// The assistant turn after the multi-part user turn must still map to model.
	if !strings.Contains(out, "<|turn|>model\n") {
		t.Errorf("assistant role did not map to model after a multi-part turn\n---\n%s", out)
	}

	// Token level: one image-placeholder ID, two text turns + gen prompt → 3
	// turn markers, 2 endturn markers.
	ids, err := tok.EncodeChat(msgs, nil)
	if err != nil {
		t.Fatalf("EncodeChat: %v", err)
	}
	if n := countTok(ids, tokImage); n != 1 {
		t.Errorf("tokenized <|image|> count = %d, want 1", n)
	}
	if n := countTok(ids, tokTurn); n != 3 {
		t.Errorf("tokenized <|turn|> count = %d, want 3 (2 turns + gen prompt)", n)
	}
}

// TestEncodeChat_CrossTurnImageOrder asserts the placeholder-emission order
// across messages each carrying images. Combined with the apiserver
// flattenImages test, this pins both ends of the cross-turn image contract:
// the template emits placeholders in (message, part) order, and the flattened
// image slice is built in that same order — so the Nth image lines up with the
// Nth placeholder. Three images across two messages must produce three
// placeholders, and the interleaved text anchors must bracket them in order.
func TestEncodeChat_CrossTurnImageOrder(t *testing.T) {
	tok := newSynthTokenizer(t)
	msgs := []ChatMessage{
		{Role: "user", Parts: []ChatContentPart{
			{Type: "text", Text: "alpha"},
			{Type: "image"}, // image #0
			{Type: "image"}, // image #1
		}},
		{Role: "user", Parts: []ChatContentPart{
			{Type: "text", Text: "bravo"},
			{Type: "image"}, // image #2
		}},
	}

	ids, err := tok.EncodeChat(msgs, nil)
	if err != nil {
		t.Fatalf("EncodeChat: %v", err)
	}
	if n := countTok(ids, tokImage); n != 3 {
		t.Errorf("expected 3 image placeholders across turns, got %d", n)
	}

	out := renderTemplateString(t, tok, msgs)
	// Order anchors: alpha precedes its two images, which precede bravo, which
	// precedes the third image. A reversed message loop would put bravo first.
	alpha := strings.Index(out, "alpha")
	bravo := strings.Index(out, "bravo")
	if !(alpha >= 0 && bravo > alpha) {
		t.Fatalf("message order wrong: alpha@%d bravo@%d\n---\n%s", alpha, bravo, out)
	}
	// Two placeholders before bravo, one after.
	before := strings.Count(out[:bravo], "<|image|>")
	after := strings.Count(out[bravo:], "<|image|>")
	if before != 2 || after != 1 {
		t.Errorf("placeholder distribution wrong: %d before bravo, %d after; want 2/1\n---\n%s", before, after, out)
	}
}

// TestEncodeChat_NoTemplate confirms the contract failure mode: no template
// loaded returns an error rather than panicking.
func TestEncodeChat_NoTemplate(t *testing.T) {
	tok := &Tokenizer{bos: -1, eos: -1, maskTokenID: -1}
	if _, err := tok.EncodeChat([]ChatMessage{{Role: "user", Content: "x"}}, nil); err == nil {
		t.Error("expected error when no chat template is loaded, got nil")
	}
}
