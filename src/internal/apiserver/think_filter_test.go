package apiserver

import (
	"fmt"
	"testing"
)

func TestThinkFilter_BasicStrip(t *testing.T) {
	f := &thinkFilter{open: "<think>", close: "</think>"}
	out := f.feed("<think>reasoning here</think>The answer is 4.")
	out += f.flush()
	if out != "The answer is 4." {
		t.Errorf("got %q, want %q", out, "The answer is 4.")
	}
}

func TestThinkFilter_EmptyThink(t *testing.T) {
	f := &thinkFilter{open: "<think>", close: "</think>"}
	out := f.feed("<think></think>The answer.")
	out += f.flush()
	if out != "The answer." {
		t.Errorf("got %q, want %q", out, "The answer.")
	}
}

func TestThinkFilter_ThinkWithNewlines(t *testing.T) {
	f := &thinkFilter{open: "<think>", close: "</think>"}
	out := f.feed("<think>\n\n</think>\n\n**Grant** is buried there.")
	out += f.flush()
	want := "\n\n**Grant** is buried there."
	if out != want {
		t.Errorf("got %q, want %q", out, want)
	}
}

func TestThinkFilter_SplitAcrossTokens(t *testing.T) {
	f := &thinkFilter{open: "<think>", close: "</think>"}
	var out string
	// <think> split across two tokens
	out += f.feed("<thi")
	out += f.feed("nk>some thought</think>answer")
	out += f.flush()
	if out != "answer" {
		t.Errorf("got %q, want %q", out, "answer")
	}
}

func TestThinkFilter_EndTagSplit(t *testing.T) {
	f := &thinkFilter{open: "<think>", close: "</think>"}
	var out string
	out += f.feed("<think>thought</th")
	out += f.feed("ink>the answer")
	out += f.flush()
	if out != "the answer" {
		t.Errorf("got %q, want %q", out, "the answer")
	}
}

func TestThinkFilter_TokenByToken(t *testing.T) {
	// Simulate token-by-token streaming like the engine produces
	tokens := []string{"<think>", "\n", "reasoning", "\n", "</think>", "\n", "Hello", " world"}
	f := &thinkFilter{open: "<think>", close: "</think>"}
	var out string
	for _, tok := range tokens {
		out += f.feed(tok)
	}
	out += f.flush()
	if out != "\nHello world" {
		t.Errorf("got %q, want %q", out, "\nHello world")
	}
}

func TestThinkFilter_NoThinkTag(t *testing.T) {
	f := &thinkFilter{open: "<think>", close: "</think>"}
	out := f.feed("Just a normal response with no thinking.")
	out += f.flush()
	want := "Just a normal response with no thinking."
	if out != want {
		t.Errorf("got %q, want %q", out, want)
	}
}

func TestThinkFilter_ThinkContentLogged(t *testing.T) {
	f := &thinkFilter{open: "<think>", close: "</think>"}
	f.feed("<think>secret reasoning</think>visible")
	f.flush()
	got := f.think.String()
	if got != "secret reasoning" {
		t.Errorf("think content = %q, want %q", got, "secret reasoning")
	}
}

func TestThinkFilter_ShortAnswer(t *testing.T) {
	// Answer shorter than the holdback window (6 chars)
	f := &thinkFilter{open: "<think>", close: "</think>"}
	out := f.feed("<think>thought</think>yes")
	out += f.flush()
	if out != "yes" {
		t.Errorf("got %q, want %q", out, "yes")
	}
}

func TestThinkFilter_SingleCharAnswer(t *testing.T) {
	f := &thinkFilter{open: "<think>", close: "</think>"}
	out := f.feed("<think>thought</think>4")
	out += f.flush()
	if out != "4" {
		t.Errorf("got %q, want %q", out, "4")
	}
}

func TestThinkFilter_StreamingNoThinkTag(t *testing.T) {
	tokens := []string{"Just", " a", " normal", " response."}
	f := &thinkFilter{open: "<think>", close: "</think>"}
	var out string
	for _, tok := range tokens {
		out += f.feed(tok)
	}
	out += f.flush()
	want := "Just a normal response."
	if out != want {
		t.Errorf("got %q, want %q", out, want)
	}
}

func TestThinkFilter_StreamingEmptyThink(t *testing.T) {
	// This is the exact pattern that caused the original bug: model produces
	// <think>\n\n</think> then the answer, token by token
	tokens := []string{"<think>", "\n\n", "</think>", "\n\n", "**Grant**", " is", " buried", " there."}
	f := &thinkFilter{open: "<think>", close: "</think>"}
	var out string
	for _, tok := range tokens {
		out += f.feed(tok)
	}
	out += f.flush()
	want := "\n\n**Grant** is buried there."
	if out != want {
		t.Errorf("got %q, want %q", out, want)
	}
}

func TestThinkFilter_StreamingStrip(t *testing.T) {
	// Simulate the streaming path: each token fed individually, collect all feed() returns
	tokens := []string{"<think>", "\n", "Let me", " think", " about", " this", "\n", "</think>", "\n", "The", " answer", " is", " 42", "."}
	f := &thinkFilter{open: "<think>", close: "</think>"}
	var out string
	for _, tok := range tokens {
		out += f.feed(tok)
	}
	out += f.flush()
	want := "\nThe answer is 42."
	if out != want {
		t.Errorf("got %q, want %q", out, want)
	}
}

func TestThinkFilter_StreamingSingleCharTokens(t *testing.T) {
	// Worst case: every character is its own token
	full := "<think>x</think>Hi"
	f := &thinkFilter{open: "<think>", close: "</think>"}
	var out string
	for _, ch := range full {
		out += f.feed(string(ch))
	}
	out += f.flush()
	if out != "Hi" {
		t.Errorf("got %q, want %q", out, "Hi")
	}
}

func TestThinkFilter_StreamingLongAnswer(t *testing.T) {
	// Verify no content loss over many tokens after </think>
	tokens := []string{"<think>", "thought", "</think>"}
	for i := range 20 {
		tokens = append(tokens, fmt.Sprintf("word%d ", i))
	}
	f := &thinkFilter{open: "<think>", close: "</think>"}
	var out string
	for _, tok := range tokens {
		out += f.feed(tok)
	}
	out += f.flush()
	// Reconstruct expected
	want := ""
	for i := range 20 {
		want += fmt.Sprintf("word%d ", i)
	}
	if out != want {
		t.Errorf("got %q, want %q", out, want)
	}
}
