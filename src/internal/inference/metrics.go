package inference

import (
	"encoding/json"
	"time"
)

// FinishReason is the canonical type for generation stop reason strings.
// Values are OpenAI-compatible.
type FinishReason = string

// Finish-reason constants — canonical values for InferenceMetrics.FinishReason
// and the finishReason helper in completions.go.
const (
	FinishReasonStop   FinishReason = "stop"
	FinishReasonLength FinishReason = "length"
	FinishReasonError  FinishReason = "error"
)

// TokenLogProb holds log-probability data for a single generated token.
type TokenLogProb struct {
	ID       int32        // token ID in vocabulary
	Token    string       // decoded token string
	LogProb  float64      // log-probability of the chosen token
	Bytes    ByteArray    // UTF-8 bytes of the token string
	TopProbs []TopLogProb // top-N alternatives (includes the chosen token)
}

// ByteArray is a []byte that JSON-marshals as an integer array [51, 52, ...]
// instead of Go's default base64 encoding for []byte.
type ByteArray []byte

func (b ByteArray) MarshalJSON() ([]byte, error) {
	ints := make([]int, len(b))
	for i, v := range b {
		ints[i] = int(v)
	}
	return json.Marshal(ints)
}

// TopLogProb holds one entry in the top-N log-probability list.
type TopLogProb struct {
	ID      int32     `json:"id"`
	Token   string    `json:"token"`
	LogProb float64   `json:"logprob"`
	Bytes   ByteArray `json:"bytes"`
}

// InferenceMetrics captures timing and throughput data for a single generation request.
type InferenceMetrics struct {
	PromptTokens     int            // input token count
	CompletionTokens int            // output tokens generated
	FinishReason     string         // "stop" or "length"
	PrefillDuration  time.Duration  // wall-clock for prefill (all prompt tokens)
	DecodeDuration   time.Duration  // wall-clock for all decode steps
	TotalDuration    time.Duration  // wall-clock for entire generation
	TokenLogProbs    []TokenLogProb // per-token log-probabilities (nil if not requested)
}

// TokensPerSec returns the decode throughput (output tokens per second).
// Returns 0 if no decode was performed.
func (m *InferenceMetrics) TokensPerSec() float64 {
	if m.DecodeDuration <= 0 || m.CompletionTokens <= 0 {
		return 0
	}
	return float64(m.CompletionTokens) / m.DecodeDuration.Seconds()
}

// PrefillTokensPerSec returns the prefill throughput (prompt tokens per second).
func (m *InferenceMetrics) PrefillTokensPerSec() float64 {
	if m.PrefillDuration <= 0 || m.PromptTokens <= 0 {
		return 0
	}
	return float64(m.PromptTokens) / m.PrefillDuration.Seconds()
}

// TotalTokensPerSec returns total throughput (prompt + completion tokens) over total duration.
// Returns 0 if TotalDuration is zero or negative.
func (m *InferenceMetrics) TotalTokensPerSec() float64 {
	if m.TotalDuration <= 0 {
		return 0
	}
	return float64(m.PromptTokens+m.CompletionTokens) / m.TotalDuration.Seconds()
}
