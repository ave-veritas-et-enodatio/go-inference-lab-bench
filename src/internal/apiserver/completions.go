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
	open        string // e.g. "<think>" — from arch TOML tokens.think_open
	close       string // e.g. "</think>" — from arch TOML tokens.think_close
	buf         string
	inThink     bool // true while inside think block; set initially if model starts in think mode
	think       strings.Builder
	thinkTokens int // tokens received while inThink was true at entry
	totalTokens int // total tokens received via feed()
}

func (f *thinkFilter) feed(token string) string {
	f.totalTokens++
	if f.inThink {
		f.thinkTokens++
	}
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

// matches logging policy in AGENTS.md
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
	Model              string             `json:"model"`
	Messages           []chatMessage      `json:"messages"`
	Stream             bool               `json:"stream"`
	MaxTokens          int                `json:"max_tokens"`
	Temperature        float64            `json:"temperature"`
	TopP               float64            `json:"top_p"`
	LogProbs           bool               `json:"logprobs,omitempty"`             // include log-probabilities
	TopLogProbs        int                `json:"top_logprobs,omitempty"`         // number of top alternatives (default 1)
	ChatTemplateKwargs map[string]any     `json:"chat_template_kwargs,omitempty"` // passed to the chat template (llama.cpp-compatible); reserved key: "enable_thinking" (bool)
	BenchCustom        *benchCustomParams `json:"bench_custom,omitempty"`
}

type diffusionRequestParams struct {
	Steps       int    `json:"steps,omitempty"`
	BlockLength int    `json:"block_length,omitempty"`
	Algorithm   string `json:"algorithm,omitempty"`
}

type benchCustomParams struct {
	Stateless           *bool                   `json:"stateless,omitempty"`
	CullMethod          string                  `json:"cull_method,omitempty"`
	FlashAttention      *bool                   `json:"flash_attention,omitempty"`
	ElideThinking       *bool                   `json:"elide_thinking,omitempty"`
	EnableRLBOnPrefill  bool                    `json:"enable_rlb_on_prefill,omitempty"`
	UseRLBGen           bool                    `json:"use_rlb_gen,omitempty"`
	RLBAlpha            *float64                `json:"rlb_alpha,omitempty"`              // 0.0-1.0; nil = default (1.0, no blending)
	RLBHaltRule         string                  `json:"rlb_halt_rule,omitempty"`          // halt rule name ("" = default fixed ceiling); see parseHaltRule in generate_rlb.go for menu
	RLBTerminalHaltRule string                  `json:"rlb_terminal_halt_rule,omitempty"` // halt rule for the terminal block only ("" = reuse rlb_halt_rule); Tier 1 rules only have meaningful signal at the terminal block since upstream tops aren't yet projected through lm_head
	Diffusion           *diffusionRequestParams `json:"diffusion,omitempty"`
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
	ThinkingTokens         int     `json:"thinking_tokens,omitempty"`
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

	// Resolve chat_template_kwargs: request overrides server default.
	// enable_thinking is a reserved key — we type-check it here (must be bool)
	// and mirror the resolved value into params.ThinkingEnabled for the
	// thinkFilter initial state and downstream generation logic.
	kwargs := make(map[string]any, len(req.ChatTemplateKwargs)+1)
	for k, v := range req.ChatTemplateKwargs {
		kwargs[k] = v
	}
	if raw, present := kwargs["enable_thinking"]; present {
		b, ok := raw.(bool)
		if !ok {
			writeError(w, http.StatusBadRequest, "invalid type for chat_template_kwargs.enable_thinking (expected bool)")
			return
		}
		params.ThinkingEnabled = b
	} else {
		params.ThinkingEnabled = s.cfg.Inference.EnableThinkingDefault
		kwargs["enable_thinking"] = params.ThinkingEnabled
	}
	params.ChatTemplateKwargs = kwargs

	if req.MaxTokens > 0 {
		params.MaxTokens = req.MaxTokens
	}
	if req.Temperature >= 0 {
		params.Temperature = float32(req.Temperature)
	}
	if req.TopP > 0 {
		params.TopP = float32(req.TopP)
	}
	if req.BenchCustom != nil && req.BenchCustom.Stateless != nil {
		params.Stateless = *req.BenchCustom.Stateless
	}
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
		var promptText string
		{
			sb := strings.Builder{}
			for _, m := range req.Messages {
				if m.Role == "user" {
					sb.WriteString(m.Content + "\n")
				}
			}
			promptText = sb.String()
		}

		// Rough estimate: assume ~1 token per 4 chars (English text)
		estimatedPromptTokens := max(len(promptText) / 4, 10)
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

	// Resolve cull method: request overrides server config; empty = use config; "none" = off.
	if req.BenchCustom != nil && req.BenchCustom.CullMethod != "" {
		params.CullMethod = req.BenchCustom.CullMethod
	} else {
		params.CullMethod = s.cfg.Inference.CullMethodDefault
	}
	// Resolve flash attention: request overrides server config; null = use config.
	if req.BenchCustom != nil && req.BenchCustom.FlashAttention != nil {
		params.FlashAttention = req.BenchCustom.FlashAttention
	}
	if req.BenchCustom != nil && req.BenchCustom.Diffusion != nil {
		if !eng.IsDiffusion() {
			log.Warn("diffusion params in request ignored: model is not a diffusion model")
		} else {
			params.Diffusion = &inference.DiffusionParams{
				Steps:       req.BenchCustom.Diffusion.Steps,
				BlockLength: req.BenchCustom.Diffusion.BlockLength,
				Algorithm:   req.BenchCustom.Diffusion.Algorithm,
			}
		}
	}
	if req.BenchCustom != nil && (req.BenchCustom.UseRLBGen || req.BenchCustom.EnableRLBOnPrefill) {
		rlb := &inference.RLBParams{
			Prefill: req.BenchCustom.EnableRLBOnPrefill,
		}
		if req.BenchCustom.UseRLBGen {
			rlb.Decode = true
			rlb.HaltRule = req.BenchCustom.RLBHaltRule
			rlb.TerminalHaltRule = req.BenchCustom.RLBTerminalHaltRule
			if req.BenchCustom.RLBAlpha != nil {
				rlb.Alpha = *req.BenchCustom.RLBAlpha
			}
		}
		params.RLB = rlb
	}
	params.Streaming = req.Stream

	// Log any request-level parameter overrides of server config defaults.
	{
		var overrides []string
		if req.BenchCustom != nil && req.BenchCustom.Stateless != nil && *req.BenchCustom.Stateless {
			overrides = append(overrides, "stateless=true")
		}
		if req.BenchCustom != nil && req.BenchCustom.CullMethod != "" && req.BenchCustom.CullMethod != s.cfg.Inference.CullMethodDefault {
			overrides = append(overrides, fmt.Sprintf("cull_method=%q (server: %q)", truncLogVal(req.BenchCustom.CullMethod), s.cfg.Inference.CullMethodDefault))
		}
		if raw, present := req.ChatTemplateKwargs["enable_thinking"]; present {
			if b, ok := raw.(bool); ok && b != s.cfg.Inference.EnableThinkingDefault {
				overrides = append(overrides, fmt.Sprintf("enable_thinking=%v (server: %v)", b, s.cfg.Inference.EnableThinkingDefault))
			}
		}
		if req.BenchCustom != nil && req.BenchCustom.ElideThinking != nil && *req.BenchCustom.ElideThinking != s.cfg.Inference.ShouldElideThink() {
			overrides = append(overrides, fmt.Sprintf("elide_thinking=%v (server: %v)", *req.BenchCustom.ElideThinking, s.cfg.Inference.ShouldElideThink()))
		}
		if req.BenchCustom != nil && req.BenchCustom.FlashAttention != nil && *req.BenchCustom.FlashAttention != s.cfg.Inference.UseFlashAttention() {
			overrides = append(overrides, fmt.Sprintf("flash_attention=%v (server: %v)", *req.BenchCustom.FlashAttention, s.cfg.Inference.UseFlashAttention()))
		}
		if req.BenchCustom != nil && req.BenchCustom.EnableRLBOnPrefill {
			overrides = append(overrides, "rlb.prefill=true")
		}
		if req.BenchCustom != nil && req.BenchCustom.UseRLBGen {
			overrides = append(overrides, "rlb.decode=true")
		}
		if req.BenchCustom != nil && req.BenchCustom.UseRLBGen && req.BenchCustom.RLBAlpha != nil {
			overrides = append(overrides, fmt.Sprintf("rlb.alpha=%.2f", *req.BenchCustom.RLBAlpha))
		}
		if req.BenchCustom != nil && req.BenchCustom.UseRLBGen && req.BenchCustom.RLBHaltRule != "" {
			overrides = append(overrides, fmt.Sprintf("rlb.halt_rule=%q", req.BenchCustom.RLBHaltRule))
		}
		if req.BenchCustom != nil && req.BenchCustom.UseRLBGen && req.BenchCustom.RLBTerminalHaltRule != "" {
			overrides = append(overrides, fmt.Sprintf("rlb.terminal_halt_rule=%q", req.BenchCustom.RLBTerminalHaltRule))
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
		if req.BenchCustom != nil && req.BenchCustom.ElideThinking != nil {
			elide = *req.BenchCustom.ElideThinking
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
			log.Info("[metrics] prompt=%d completion=%d think=%d prefill=%.1fms decode=%.1fms total=%.1fms tok/s=%.1f cull=%.1f%%",
				metrics.PromptTokens, metrics.CompletionTokens, filter.thinkTokens,
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
		elide := s.cfg.Inference.ShouldElideThink()
		if req.BenchCustom != nil && req.BenchCustom.ElideThinking != nil {
			elide = *req.BenchCustom.ElideThinking
		}

		// Always feed tokens through the filter for think-token counting;
		// also collect visible output when eliding.
		filter := &thinkFilter{open: eng.ThinkOpen(), close: eng.ThinkClose(), inThink: params.ThinkingEnabled}
		var visible strings.Builder
		var raw []byte

		metrics, genErr := eng.Generate(msgs, params, func(token string) bool {
			visible.WriteString(filter.feed(token))
			raw = append(raw, token...)
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
		if elide {
			content = strings.TrimSpace(visible.String() + filter.flush())
			if s.cfg.Inference.LogThinking {
				filter.logThinking(req.Model)
			}
		} else {
			content = strings.TrimSpace(string(raw))
		}

		if metrics != nil {
			log.Info("[metrics] prompt=%d completion=%d think=%d prefill=%.1fms decode=%.1fms total=%.1fms tok/s=%.1f cull=%.1f%%",
				metrics.PromptTokens, metrics.CompletionTokens, filter.thinkTokens,
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
				ThinkingTokens:         filter.thinkTokens,
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
