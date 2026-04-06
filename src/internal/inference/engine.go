package inference

import (
	"fmt"
	"time"

	ggufparser "github.com/gpustack/gguf-parser-go"

	log "inference-lab-bench/internal/log"
	"inference-lab-bench/internal/inference/arch"
	"inference-lab-bench/internal/model"
)

// Engine runs inference for a single loaded model.
type Engine struct {
	model     *arch.GenericModel
	tokenizer *Tokenizer
	nLayers   int
	maxSeqLen int
}

// NewEngine creates an inference engine for the given model.
// archDir is the directory containing architecture definition TOML files.
// maxSeqLen is the KV cache size in tokens (0 = default 8192).
func NewEngine(info *model.ModelInfo, archDir string, maxSeqLen int) (*Engine, error) {
	f, err := ggufparser.ParseGGUFFile(info.Path)
	if err != nil {
		return nil, fmt.Errorf("parsing GGUF for tokenizer: %w", err)
	}

	tok, err := NewTokenizerFromGGUF(f, info.Path)
	if err != nil {
		return nil, fmt.Errorf("building tokenizer: %w", err)
	}

	// DSL-driven model loading
	m, err := arch.NewGenericModel(info.Metadata.Architecture, info.Path, archDir)
	if err != nil {
		return nil, fmt.Errorf("creating model: %w", err)
	}

	if maxSeqLen <= 0 {
		maxSeqLen = defaultMaxSeqLen
	}

	validateArchTokens(m.Def.Tokens, tok)

	return &Engine{
		model:     m,
		tokenizer: tok,
		nLayers:   info.Metadata.NLayers,
		maxSeqLen: maxSeqLen,
	}, nil
}

// validateArchTokens checks each non-empty token string from the arch TOML
// against the loaded vocabulary. Warns if a string is not a direct vocab entry
// or encodes to zero tokens.
func validateArchTokens(tokens arch.TokensDef, tok *Tokenizer) {
	check := func(name, value string) {
		if value == "" {
			return
		}
		ids := tok.Encode(value, false)
		if len(ids) == 0 {
			log.Warn("arch token %s=%q encodes to zero tokens — not in vocabulary", name, value)
		} else if !tok.VocabContains(value) {
			log.Warn("arch token %s=%q is not a single vocabulary entry (encodes to %d BPE sub-tokens)", name, value, len(ids))
		}
	}
	check("think_open", tokens.ThinkOpen)
	check("think_close", tokens.ThinkClose)
}

// ThinkOpen returns the TOML-defined think opening tag, or "".
func (e *Engine) ThinkOpen() string { return e.model.Def.Tokens.ThinkOpen }

// ThinkClose returns the TOML-defined think closing tag, or "".
func (e *Engine) ThinkClose() string { return e.model.Def.Tokens.ThinkClose }

// Close frees all GPU resources.
func (e *Engine) Close() {
	if e.model != nil {
		e.model.Close()
	}
}

// GenerateParams controls the generation loop.
type GenerateParams struct {
	MaxTokens       int
	Temperature     float32
	TopP            float32
	Stateless       bool // bypass KV cache (for testing/comparison)
	ThinkingEnabled bool // passed to template as enable_thinking variable
	LogProbs        bool // include log-probabilities in response
	TopLogProbs     int  // number of top log-probabilities per token
}

// DefaultParams returns sensible defaults.
func DefaultParams() GenerateParams {
	return GenerateParams{
		MaxTokens:       512,
		Temperature:     0.7,
		TopP:            0.9,
		ThinkingEnabled: true,
	}
}

const defaultMaxSeqLen = 8192

// Generate runs the full chat generation loop.
// Uses KV/SSM cache by default; set params.Stateless for full-sequence recomputation.
// Returns metrics for the generation request.
func (e *Engine) Generate(
	messages []ChatMessage,
	params GenerateParams,
	onToken func(token string) bool,
) (*InferenceMetrics, error) {
	promptIDs, err := e.tokenizer.EncodeChat(messages, params.ThinkingEnabled)
	if err != nil {
		return nil, fmt.Errorf("encoding chat: %w", err)
	}
	if len(promptIDs) == 0 {
		return nil, fmt.Errorf("empty prompt after encoding")
	}

	// The template handles enable_thinking natively — when disabled, it emits an
	// empty think block (e.g. "<think>\n\n</think>\n\n") which is the correct
	// prompt. No post-render stripping needed; this matches llama.cpp behavior.

	maxTokens := params.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 512
	}

	stopID := e.tokenizer.StopID()

	metrics := &InferenceMetrics{
		PromptTokens: len(promptIDs),
	}

	start := time.Now()

	var genErr error
	if params.Stateless {
		genErr = e.generateStateless(promptIDs, maxTokens, stopID, params, onToken, metrics)
	} else {
		genErr = e.generateCached(promptIDs, maxTokens, stopID, params, onToken, metrics)
	}

	metrics.TotalDuration = time.Since(start)

	return metrics, genErr
}

func (e *Engine) generateCached(
	promptIDs []int32, maxTokens int, stopID int32,
	params GenerateParams,
	onToken func(string) bool, metrics *InferenceMetrics,
) error {
	cache, err := e.model.NewCache(e.maxSeqLen)
	if err != nil {
		return fmt.Errorf("creating cache: %w", err)
	}
	defer cache.Free()

	prefillStart := time.Now()
	logits, err := e.model.ForwardCached(cache, promptIDs)
	if err != nil {
		return fmt.Errorf("prefill forward: %w", err)
	}
	metrics.PrefillDuration = time.Since(prefillStart)

	decodeStart := time.Now()
	for range maxTokens {
		nextID := e.sample(logits, params)
		if nextID == stopID {
			break
		}
		if params.LogProbs {
			metrics.TokenLogProbs = append(metrics.TokenLogProbs,
				ComputeTopLogProbs(logits, nextID, params.TopLogProbs, e.tokenizer.TokenString))
		}
		if !onToken(e.tokenizer.TokenString(nextID)) {
			break
		}
		metrics.CompletionTokens++
		logits, err = e.model.ForwardCached(cache, []int32{nextID})
		if err != nil {
			return fmt.Errorf("decode forward: %w", err)
		}
	}
	metrics.DecodeDuration = time.Since(decodeStart)
	return nil
}

func (e *Engine) generateStateless(
	promptIDs []int32, maxTokens int, stopID int32,
	params GenerateParams,
	onToken func(string) bool, metrics *InferenceMetrics,
) error {
	tokenIDs := make([]int32, len(promptIDs))
	copy(tokenIDs, promptIDs)

	for range maxTokens {
		stepStart := time.Now()
		logits, err := e.model.ForwardStateless(tokenIDs)
		if err != nil {
			return fmt.Errorf("forward pass: %w", err)
		}
		if metrics.CompletionTokens == 0 {
			metrics.PrefillDuration = time.Since(stepStart)
		} else {
			metrics.DecodeDuration += time.Since(stepStart)
		}
		nextID := e.sample(logits, params)
		if nextID == stopID {
			break
		}
		if params.LogProbs {
			metrics.TokenLogProbs = append(metrics.TokenLogProbs,
				ComputeTopLogProbs(logits, nextID, params.TopLogProbs, e.tokenizer.TokenString))
		}
		if !onToken(e.tokenizer.TokenString(nextID)) {
			break
		}
		metrics.CompletionTokens++
		tokenIDs = append(tokenIDs, nextID)
	}
	return nil
}

func (e *Engine) sample(logits []float32, params GenerateParams) int32 {
	if params.Temperature <= 0 {
		return Greedy(logits)
	}
	return TopP(logits, params.Temperature, params.TopP)
}
