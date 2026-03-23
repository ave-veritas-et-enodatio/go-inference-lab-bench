package inference

import "time"

// InferenceMetrics captures timing and throughput data for a single generation request.
type InferenceMetrics struct {
	PromptTokens     int           // input token count
	CompletionTokens int           // output tokens generated
	PrefillDuration  time.Duration // wall-clock for prefill (all prompt tokens)
	DecodeDuration   time.Duration // wall-clock for all decode steps
	TotalDuration    time.Duration // wall-clock for entire generation
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

