package inference

import "testing"

func TestPromptOpensThink(t *testing.T) {
	const (
		openTok  int32 = 100
		closeTok int32 = 101
	)
	cases := []struct {
		name    string
		ids     []int32
		closeID int32
		want    bool
	}{
		{"closed block (open then close)", []int32{1, openTok, 2, closeTok, 3}, closeTok, false},
		{"open with no close (we-open prime)", []int32{1, openTok, 2}, closeTok, true},
		{"reopened after close (open is last)", []int32{openTok, closeTok, openTok}, closeTok, true},
		{"no open token at all (model-opens)", []int32{1, 2, closeTok, 3}, closeTok, false},
		{"open present, close not a vocab token", []int32{1, openTok}, -1, true},
		{"empty sequence", []int32{}, closeTok, false},
		{"qwen3 thinking-on tail: …<think>", []int32{5, 6, openTok}, closeTok, true},
		{"qwen3 thinking-off tail: …<think></think>", []int32{5, openTok, closeTok}, closeTok, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := promptOpensThink(c.ids, openTok, c.closeID); got != c.want {
				t.Errorf("promptOpensThink(%v, open=%d, close=%d) = %v, want %v",
					c.ids, openTok, c.closeID, got, c.want)
			}
		})
	}
}
