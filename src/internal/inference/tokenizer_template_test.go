package inference

import (
	"strings"
	"testing"

	"github.com/nikolalohinski/gonja/v2"
	"github.com/nikolalohinski/gonja/v2/exec"
)

// TestRewriteJinjaSetBlocks_Simple covers the most basic case: a set-block
// that captures literal content.
func TestRewriteJinjaSetBlocks_Simple(t *testing.T) {
	in := `{%- set foo -%}hello{%- endset -%}{{ foo }}`
	got := rewriteJinjaSetBlocks(in)
	if strings.Contains(got, "endset") {
		t.Errorf("output still contains endset: %s", got)
	}
	if !strings.Contains(got, "macro __setblock_foo_0") {
		t.Errorf("output missing expected macro definition: %s", got)
	}
	if !strings.Contains(got, "set foo = __setblock_foo_0()") {
		t.Errorf("output missing expected set-with-call: %s", got)
	}
	// Sanity: gonja should now accept it.
	if _, err := gonja.FromString(got); err != nil {
		t.Errorf("gonja rejects rewritten template: %v\n---\n%s", err, got)
	}
}

// TestRewriteJinjaSetBlocks_WithEmbeddedControl covers a set-block whose body
// contains other Jinja constructs (if/for/output) — the actual Gemma 4 case.
func TestRewriteJinjaSetBlocks_WithEmbeddedControl(t *testing.T) {
	in := `
{%- for item in items -%}
{%- set captured_content -%}
{%- if item.kind == 'text' -%}
{{- item.value | trim -}}
{%- elif item.kind == 'image' -%}
<|image|>
{%- endif -%}
{%- endset -%}
{{ captured_content }}
{%- endfor -%}
`
	got := rewriteJinjaSetBlocks(in)
	if strings.Contains(got, "endset") {
		t.Errorf("output still contains endset: %s", got)
	}
	// The inner if/elif/endif must be preserved inside the macro body.
	if !strings.Contains(got, "elif item.kind == 'image'") {
		t.Errorf("output dropped inner control structure: %s", got)
	}
	if _, err := gonja.FromString(got); err != nil {
		t.Errorf("gonja rejects rewritten template with embedded controls: %v\n---\n%s", err, got)
	}
}

// TestRewriteJinjaSetBlocks_MultipleBlocks confirms each block gets a unique
// macro name (indexed counter).
func TestRewriteJinjaSetBlocks_MultipleBlocks(t *testing.T) {
	in := `{%- set a -%}A{%- endset -%}{%- set b -%}B{%- endset -%}{{ a }}{{ b }}`
	got := rewriteJinjaSetBlocks(in)
	if !strings.Contains(got, "__setblock_a_0") {
		t.Errorf("first block missing index 0: %s", got)
	}
	if !strings.Contains(got, "__setblock_b_1") {
		t.Errorf("second block missing index 1: %s", got)
	}
	if _, err := gonja.FromString(got); err != nil {
		t.Errorf("gonja rejects multi-block rewrite: %v\n---\n%s", err, got)
	}
}

// TestRewriteJinjaSetBlocks_SetExpressionUnaffected confirms the rewrite
// doesn't touch normal `{% set var = expr %}` syntax (which gonja already
// handles correctly).
func TestRewriteJinjaSetBlocks_SetExpressionUnaffected(t *testing.T) {
	in := `{%- set x = 42 -%}{%- set y = x + 1 -%}{{ y }}`
	got := rewriteJinjaSetBlocks(in)
	if got != in {
		t.Errorf("set-expression form was unexpectedly modified:\nin:  %s\nout: %s", in, got)
	}
}

// TestRewriteJinjaSetBlocks_RenderEquivalence renders both forms with gonja
// and confirms the output text is identical — the rewrite must be
// semantically transparent.
func TestRewriteJinjaSetBlocks_RenderEquivalence(t *testing.T) {
	// Reference: equivalent template using only set-with-call (already gonja-
	// compatible). The rewrite should produce the same rendered output.
	ref := `{%- macro greeting() -%}hi {{ name }}{%- endmacro -%}{%- set msg = greeting() -%}{{ msg }}!`
	// Rewritten input: the same logic via set-block.
	in := `{%- set msg -%}hi {{ name }}{%- endset -%}{{ msg }}!`
	rewritten := rewriteJinjaSetBlocks(in)

	refTpl, err := gonja.FromString(ref)
	if err != nil {
		t.Fatalf("ref template compile: %v", err)
	}
	gotTpl, err := gonja.FromString(rewritten)
	if err != nil {
		t.Fatalf("rewritten template compile: %v\n---\n%s", err, rewritten)
	}
	refOut, err := refTpl.ExecuteToString(exec.NewContext(map[string]any{"name": "world"}))
	if err != nil {
		t.Fatalf("ref render: %v", err)
	}
	gotOut, err := gotTpl.ExecuteToString(exec.NewContext(map[string]any{"name": "world"}))
	if err != nil {
		t.Fatalf("rewritten render: %v", err)
	}
	if refOut != gotOut {
		t.Errorf("render output mismatch:\nref: %q\ngot: %q", refOut, gotOut)
	}
}

// TestRewriteJinjaDictGet_Basic covers the simple single-arg rewrite.
func TestRewriteJinjaDictGet_Basic(t *testing.T) {
	cases := []struct{ in, want string }{
		{`{{ message.get('reasoning') }}`, `{{ message['reasoning'] }}`},
		{`{{ message.get("content") }}`, `{{ message["content"] }}`},
		{`{%- if part.get('type') == 'text' -%}`, `{%- if part['type'] == 'text' -%}`},
		{`{{ tc.get('id') == follow.get('tool_call_id') }}`, `{{ tc['id'] == follow['tool_call_id'] }}`},
		// Whitespace inside parens should be allowed.
		{`{{ x.get(  'k'  ) }}`, `{{ x['k'] }}`},
	}
	for _, tc := range cases {
		got := rewriteJinjaDictGet(tc.in)
		if got != tc.want {
			t.Errorf("rewriteJinjaDictGet(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// TestRewriteJinjaDictGet_DoesNotMatchTwoArg confirms the rewrite leaves
// two-arg dict.get(key, default) alone (we'd need a different replacement
// shape for those; deferring until a model actually uses them).
func TestRewriteJinjaDictGet_DoesNotMatchTwoArg(t *testing.T) {
	in := `{{ message.get('x', 'fallback') }}`
	got := rewriteJinjaDictGet(in)
	if got != in {
		t.Errorf("two-arg .get was unexpectedly rewritten:\nin:  %s\nout: %s", in, got)
	}
}

// TestRewriteJinjaDictGet_GonjaAcceptsOutput is a smoke test: the rewritten
// form must compile under gonja against the receiver patterns we actually
// use in Gemma 4's template.
func TestRewriteJinjaDictGet_GonjaAcceptsOutput(t *testing.T) {
	in := `
{%- set thinking_text = message.get('reasoning') or message.get('reasoning_content') -%}
{%- if message.get('tool_calls') -%}
  {{ tc.get('id') }}
{%- endif -%}
`
	got := rewriteJinjaDictGet(in)
	if _, err := gonja.FromString(got); err != nil {
		t.Fatalf("gonja rejects rewritten template: %v\n---\n%s", err, got)
	}
}

// TestRewriteJinjaSetBlocks_Gemma4Shape exercises a structure matching the
// actual Gemma 4 multimodal chat template — set-block inside a for-loop with
// nested if/elif chain producing mixed content.
func TestRewriteJinjaSetBlocks_Gemma4Shape(t *testing.T) {
	tpl := `
{%- for message in messages -%}
{%- set captured_content -%}
{%- if message['content'] is string -%}
{{- message['content'] | trim -}}
{%- elif message['content'] is sequence -%}
{%- for item in message['content'] -%}
{%- if item['type'] == 'text' -%}
{{- item['text'] | trim -}}
{%- elif item['type'] == 'image' -%}
<|image|>
{%- endif -%}
{%- endfor -%}
{%- endif -%}
{%- endset -%}
{{ captured_content }}
{%- endfor -%}
`
	got := rewriteJinjaSetBlocks(tpl)
	compiled, err := gonja.FromString(got)
	if err != nil {
		t.Fatalf("gonja rejects Gemma-4-shape rewrite: %v\n---\n%s", err, got)
	}
	// Smoke-render to make sure the macro path actually executes.
	out, err := compiled.ExecuteToString(exec.NewContext(map[string]any{
		"messages": []any{
			map[string]any{"content": "plain text message"},
			map[string]any{"content": []any{
				map[string]any{"type": "text", "text": "  with leading/trailing  "},
				map[string]any{"type": "image"},
			}},
		},
	}))
	if err != nil {
		t.Fatalf("render error: %v", err)
	}
	if !strings.Contains(out, "plain text message") {
		t.Errorf("missing string-content render: %q", out)
	}
	if !strings.Contains(out, "<|image|>") {
		t.Errorf("missing image placeholder render: %q", out)
	}
}
