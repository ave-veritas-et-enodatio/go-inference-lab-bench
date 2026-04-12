package apiserver

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"inference-lab-bench/internal/inference"
	log "inference-lab-bench/internal/log"
	"inference-lab-bench/internal/util"
)

// thinkFilter strips <think>...</think> sections from a token stream.
// Handles tags split across token boundaries. Accumulates think content for logging.
type thinkFilter struct {
	open    string // e.g. "<think>" — from arch TOML tokens.think_open
	close   string // e.g. "</think>" — from arch TOML tokens.think_close
	buf     string
	inThink bool // true while inside think block; set initially if model starts in think mode
	think   strings.Builder
}

func (f *thinkFilter) feed(token string) string {
	f.buf += token
	var out strings.Builder
	holdClose := len(f.close) + 1
	holdOpen := len(f.open) + 1
	for {
		if f.inThink {
			if i := strings.Index(f.buf, f.close); i >= 0 {
				f.think.WriteString(f.buf[:i])
				f.buf = f.buf[i+len(f.close):]
				f.inThink = false
			} else {
				// hold tail in case close tag is split across tokens
				if len(f.buf) > holdClose {
					f.think.WriteString(f.buf[:len(f.buf)-holdClose])
					f.buf = f.buf[len(f.buf)-holdClose:]
				}
				break
			}
		} else {
			if f.open != "" {
				if i := strings.Index(f.buf, f.open); i >= 0 {
					out.WriteString(f.buf[:i])
					f.buf = f.buf[i+len(f.open):]
					f.inThink = true
					continue
				}
			}
			// hold tail in case open tag is split across tokens
			safe := len(f.buf) - holdOpen
			if safe <= 0 {
				break
			}
			out.WriteString(f.buf[:safe])
			f.buf = f.buf[safe:]
			break
		}
	}
	return out.String()
}

func (f *thinkFilter) flush() string {
	if f.inThink {
		f.think.WriteString(f.buf) // incomplete think block — treat remainder as think content
		f.buf = ""
	}
	out := f.buf
	f.buf = ""
	f.inThink = false
	return out
}

const maxThinkLogLen = 500 // cap think content in logs

const maxLogValLen = 64

func truncLogVal(s string) string {
	if len(s) > maxLogValLen {
		return s[:maxLogValLen] + "…"
	}
	return s
}

func (f *thinkFilter) logThinking(model string) {
	s := f.think.String()
	if s == "" {
		return
	}
	if len(s) > maxThinkLogLen {
		s = s[:maxThinkLogLen] + "...[truncated]"
	}
	log.Info("[think] model=%s %s", truncLogVal(model), s)
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatCompletionRequest struct {
	Model             string        `json:"model"`
	Messages          []chatMessage `json:"messages"`
	Stream            bool          `json:"stream"`
	MaxTokens         int           `json:"max_tokens"`
	Temperature       float64       `json:"temperature"`
	TopP              float64       `json:"top_p"`
	LogProbs          bool          `json:"logprobs,omitempty"`           // include log-probabilities
	TopLogProbs       int           `json:"top_logprobs,omitempty"`       // number of top alternatives (default 1)
	Stateless         bool          `json:"stateless,omitempty"`          // bypass KV cache (for testing)
	CullMethod        *string       `json:"cull_method,omitempty"`        // null = use server config; "none" = off; "random" = random
	EnableThinking    *bool         `json:"enable_thinking,omitempty"`    // true/false/null; null = use server default
	ElideThinking     *bool         `json:"elide_thinking,omitempty"`     // true/false/null; null = use server default
	FlashAttention    *bool         `json:"flash_attention,omitempty"`    // true/false/null; null = use server default
	Diffusion *diffusionRequestParams `json:"diffusion,omitempty"` // nil = no diffusion params; ignored for non-diffusion models
}

type diffusionRequestParams struct {
	Steps       int    `json:"steps,omitempty"`
	BlockLength int    `json:"block_length,omitempty"`
	Algorithm   string `json:"algorithm,omitempty"`
}

// --- Non-streaming response types ---

type choiceMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type choiceLogProbs struct {
	Content []choiceLogProbEntry `json:"content"`
}

type choiceLogProbEntry struct {
	TopLogProbs []inference.TopLogProb `json:"top_logprobs"`
}

type choice struct {
	Index        int             `json:"index"`
	Message      choiceMessage   `json:"message"`
	FinishReason string          `json:"finish_reason"`
	LogProbs     *choiceLogProbs `json:"logprobs,omitempty"`
}

type usage struct {
	PromptTokens           int     `json:"prompt_tokens"`
	CompletionTokens       int     `json:"completion_tokens"`
	TotalTokens            int     `json:"total_tokens"`
	PromptTokensPerSec     float64 `json:"prompt_tokens_per_sec,omitempty"`
	CompletionTokensPerSec float64 `json:"completion_tokens_per_sec,omitempty"`
	TotalTokensPerSec      float64 `json:"total_tokens_per_sec,omitempty"`
	PrefillSeconds         float64 `json:"prefill_seconds,omitempty"`
	DecodeSeconds          float64 `json:"decode_seconds,omitempty"`
	TotalSeconds           float64 `json:"total_seconds,omitempty"`
}

type chatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []choice `json:"choices"`
	Usage   usage    `json:"usage"`
}

// --- Streaming delta types ---

type deltaContent struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type streamChoice struct {
	Index        int          `json:"index"`
	Delta        deltaContent `json:"delta"`
	FinishReason *string      `json:"finish_reason"`
}

type streamChunk struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []streamChoice `json:"choices"`
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	s.pending.Add(1)
	defer s.pending.Done()
	var req chatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	// Resolve default model
	if req.Model == "" || req.Model == "default" {
		req.Model = s.resolveDefaultModel()
	}
	if req.Model == "" {
		log.Error("no models available")
		writeError(w, http.StatusNotFound, "no models available")
		return
	}
	if s.manager.Get(req.Model) == nil {
		log.Error("model not found: %s", req.Model)
		writeError(w, http.StatusNotFound, "model not found")
		return
	}

	eng, err := s.Engine(req.Model)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	params := inference.DefaultParams()

	// Resolve thinking mode: request field overrides server default
	if req.EnableThinking != nil {
		params.ThinkingEnabled = *req.EnableThinking
	} else {
		params.ThinkingEnabled = s.cfg.Inference.EnableThinkingDefault
	}

	if req.MaxTokens > 0 {
		params.MaxTokens = req.MaxTokens
	}
	if req.Temperature >= 0 {
		params.Temperature = float32(req.Temperature)
	}
	if req.TopP > 0 {
		params.TopP = float32(req.TopP)
	}
	params.Stateless = req.Stateless
	params.LogProbs = req.LogProbs
	params.TopLogProbs = req.TopLogProbs
	if params.TopLogProbs < 1 && params.LogProbs {
		params.TopLogProbs = 1
	}

	// Thinking mode is controlled via the enable_thinking template variable
	// passed to gonja in EncodeChat — no prompt injection needed.

	// Guardrails: enforce context length limits to prevent system overload
	if s.cfg.Inference.MaxRequestSeqLen > 0 && req.MaxTokens > 0 {
		// Estimate prompt token count via tokenizer
		// This will be re-encoded inside Generate(), but we need the estimate for guardrails
		promptText := ""
		for _, m := range req.Messages {
			if m.Role == "user" {
				promptText += m.Content + "\n"
			}
		}
		// Rough estimate: assume ~1 token per 4 chars (English text)
		estimatedPromptTokens := len(promptText) / 4
		if estimatedPromptTokens < 10 {
			estimatedPromptTokens = 10 // minimum reasonable prompt
		}
		totalEstimate := estimatedPromptTokens + req.MaxTokens
		if totalEstimate > s.cfg.Inference.MaxRequestSeqLen {
			if s.cfg.Inference.StrictMode {
				log.Error("request exceeds guardrail limit: estimated_prompt=%d + max_tokens=%d = %d > max_request_seq_len=%d",
					estimatedPromptTokens, req.MaxTokens, totalEstimate, s.cfg.Inference.MaxRequestSeqLen)
				writeError(w, http.StatusBadRequest, fmt.Sprintf("request exceeds context limit: %d tokens (prompt=%d + max_tokens=%d); max_request_seq_len=%d",
					totalEstimate, estimatedPromptTokens, req.MaxTokens, s.cfg.Inference.MaxRequestSeqLen))
				return
			} else {
				log.Warn("request exceeds guardrail limit: estimated_prompt=%d + max_tokens=%d = %d > max_request_seq_len=%d (strict_mode=false, allowing)",
					estimatedPromptTokens, req.MaxTokens, totalEstimate, s.cfg.Inference.MaxRequestSeqLen)
			}
		}
	}

	msgs := make([]inference.ChatMessage, len(req.Messages))
	for i, m := range req.Messages {
		msgs[i] = inference.ChatMessage{Role: m.Role, Content: m.Content}
	}
	// Thinking mode is controlled via the enable_thinking template variable
	// passed to gonja in EncodeChat — no prompt injection needed.

	// Resolve cull method: request overrides server config; null = use config; "none" = off.
	if req.CullMethod != nil {
		params.CullMethod = *req.CullMethod
	} else {
		params.CullMethod = s.cfg.Inference.CullMethodDefault
	}
	// Resolve flash attention: request overrides server config; null = use config.
	if req.FlashAttention != nil {
		params.FlashAttention = req.FlashAttention
	}
	if req.Diffusion != nil {
		if !eng.IsDiffusion() {
			log.Warn("diffusion params in request ignored: model is not a diffusion model")
		} else {
			params.Diffusion = &inference.DiffusionParams{
				Steps:       req.Diffusion.Steps,
				BlockLength: req.Diffusion.BlockLength,
				Algorithm:   req.Diffusion.Algorithm,
			}
		}
	}
	params.Streaming = req.Stream

	// Log any request-level parameter overrides of server config defaults.
	{
		var overrides []string
		if req.Stateless {
			overrides = append(overrides, "stateless=true")
		}
		if req.CullMethod != nil && *req.CullMethod != s.cfg.Inference.CullMethodDefault {
			overrides = append(overrides, fmt.Sprintf("cull_method=%q (server: %q)", truncLogVal(*req.CullMethod), s.cfg.Inference.CullMethodDefault))
		}
		if req.EnableThinking != nil && *req.EnableThinking != s.cfg.Inference.EnableThinkingDefault {
			overrides = append(overrides, fmt.Sprintf("enable_thinking=%v (server: %v)", *req.EnableThinking, s.cfg.Inference.EnableThinkingDefault))
		}
		if req.ElideThinking != nil && *req.ElideThinking != s.cfg.Inference.ShouldElideThink() {
			overrides = append(overrides, fmt.Sprintf("elide_thinking=%v (server: %v)", *req.ElideThinking, s.cfg.Inference.ShouldElideThink()))
		}
		if req.FlashAttention != nil && *req.FlashAttention != s.cfg.Inference.UseFlashAttention() {
			overrides = append(overrides, fmt.Sprintf("flash_attention=%v (server: %v)", *req.FlashAttention, s.cfg.Inference.UseFlashAttention()))
		}
		if len(overrides) > 0 {
			log.Info("[req] param overrides: %s", strings.Join(overrides, ", "))
		}
	}

	chunkID := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	created := time.Now().Unix()

	if req.Stream {
		// --- Streaming (SSE) ---
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("X-Accel-Buffering", "no")
		flusher, canFlush := w.(http.Flusher)

		sendChunk := func(delta deltaContent, finishReason *string) {
			chunk := streamChunk{
				ID:      chunkID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   req.Model,
				Choices: []streamChoice{{Index: 0, Delta: delta, FinishReason: finishReason}},
			}
			data, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", data)
			if canFlush {
				flusher.Flush()
			}
		}

		// Send role delta first
		sendChunk(deltaContent{Role: "assistant"}, nil)

		elide := s.cfg.Inference.ShouldElideThink()
		if req.ElideThinking != nil {
			elide = *req.ElideThinking
		}

		filter := &thinkFilter{open: eng.ThinkOpen(), close: eng.ThinkClose(), inThink: params.ThinkingEnabled}
		metrics, genErr := eng.Generate(msgs, params, func(token string) bool {
			if elide {
				if out := filter.feed(token); out != "" {
					sendChunk(deltaContent{Content: out}, nil)
				}
			} else {
				sendChunk(deltaContent{Content: token}, nil)
			}
			return true
		})
		if metrics != nil {
			log.Info("[metrics] prompt=%d completion=%d prefill=%.1fms decode=%.1fms total=%.1fms tok/s=%.1f cull=%.1f%%",
				metrics.PromptTokens, metrics.CompletionTokens,
				float64(metrics.PrefillDuration.Microseconds())/1000,
				float64(metrics.DecodeDuration.Microseconds())/1000,
				float64(metrics.TotalDuration.Microseconds())/1000,
				metrics.TokensPerSec(), metrics.CullRatio()*100)
		}
		if elide {
			if out := filter.flush(); out != "" {
				sendChunk(deltaContent{Content: out}, nil)
			}
		}
		if s.cfg.Inference.LogThinking {
			filter.logThinking(req.Model)
		}

		if genErr != nil && inference.IsComputeFailure(genErr) {
			// GPU backend is permanently poisoned after a compute failure.
			// Evict the engine now so the next request loads a fresh one.
			// We cannot change the HTTP status (headers already sent), but
			// finish_reason:"error" signals the failure to the client.
			log.Error("GPU compute failure for %s — evicting engine for recovery: %v", req.Model, genErr)
			s.evictEngine(req.Model)
		}

		stop := finishReason(metrics, genErr)
		sendChunk(deltaContent{}, &stop)
		fmt.Fprintf(w, "data: [DONE]\n\n")
		if canFlush {
			flusher.Flush()
		}

	} else {
		// --- Non-streaming ---
		var sb []byte

		metrics, genErr := eng.Generate(msgs, params, func(token string) bool {
			sb = append(sb, token...)
			return true
		})
		if genErr != nil {
			if inference.IsComputeFailure(genErr) {
				// GPU backend is permanently poisoned — evict now so the next
				// request loads a fresh engine.
				log.Error("GPU compute failure for %s — evicting engine for recovery: %v", req.Model, genErr)
				s.evictEngine(req.Model)
				writeError(w, http.StatusServiceUnavailable, "GPU compute failure; please retry")
			} else {
				writeError(w, http.StatusInternalServerError, genErr.Error())
			}
			return
		}

		var content string
		elide := s.cfg.Inference.ShouldElideThink()
		if req.ElideThinking != nil {
			elide = *req.ElideThinking
		}
		if elide {
			filter := &thinkFilter{open: eng.ThinkOpen(), close: eng.ThinkClose(), inThink: params.ThinkingEnabled}
			content = filter.feed(string(sb))
			content += filter.flush()
			content = strings.TrimSpace(content)
			if s.cfg.Inference.LogThinking {
				filter.logThinking(req.Model)
			}
		} else {
			content = strings.TrimSpace(string(sb))
		}

		if metrics != nil {
			log.Info("[metrics] prompt=%d completion=%d prefill=%.1fms decode=%.1fms total=%.1fms tok/s=%.1f cull=%.1f%%",
				metrics.PromptTokens, metrics.CompletionTokens,
				float64(metrics.PrefillDuration.Microseconds())/1000,
				float64(metrics.DecodeDuration.Microseconds())/1000,
				float64(metrics.TotalDuration.Microseconds())/1000,
				metrics.TokensPerSec(), metrics.CullRatio()*100)
		}

		var logProbs *choiceLogProbs
		if req.LogProbs && metrics != nil && len(metrics.TokenLogProbs) > 0 {
			entries := make([]choiceLogProbEntry, len(metrics.TokenLogProbs))
			for i, tlp := range metrics.TokenLogProbs {
				entries[i] = choiceLogProbEntry{TopLogProbs: tlp.TopProbs}
			}
			logProbs = &choiceLogProbs{Content: entries}
		}

		resp := chatCompletionResponse{
			ID:      chunkID,
			Object:  "chat.completion",
			Created: created,
			Model:   req.Model,
			Choices: []choice{{
				Index:        0,
				Message:      choiceMessage{Role: "assistant", Content: content},
				FinishReason: finishReason(metrics, genErr),
				LogProbs:     logProbs,
			}},
			Usage: usage{
				PromptTokens:           metrics.PromptTokens,
				CompletionTokens:       metrics.CompletionTokens,
				TotalTokens:            metrics.PromptTokens + metrics.CompletionTokens,
				PromptTokensPerSec:     metrics.PrefillTokensPerSec(),
				CompletionTokensPerSec: metrics.TokensPerSec(),
				TotalTokensPerSec:      metrics.TotalTokensPerSec(),
				PrefillSeconds:         metrics.PrefillDuration.Seconds(),
				DecodeSeconds:          metrics.DecodeDuration.Seconds(),
				TotalSeconds:           metrics.TotalDuration.Seconds(),
			},
		}
		util.WriteJSON(w, resp)
	}
}

// finishReason maps generation outcome to an OpenAI-compatible finish_reason string.
// Returns "error" on any error, "length" if the token budget was exhausted, and
// "stop" otherwise (stop token hit or metrics unavailable).
func finishReason(m *inference.InferenceMetrics, err error) string {
	if err != nil {
		return "error"
	}
	if m != nil && m.FinishReason == "length" {
		return "length"
	}
	return "stop"
}

func writeError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	fmt.Fprintf(w, `{"error":{"message":%q,"type":"api_error"}}`, msg)
}
