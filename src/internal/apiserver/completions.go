package apiserver

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"inference-lab-bench/internal/inference"
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

func (f *thinkFilter) logThinking(model string) {
	if s := f.think.String(); s != "" {
		log.Printf("[think] model=%s %s", model, s)
	}
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatCompletionRequest struct {
	Model           string       `json:"model"`
	Messages        []chatMessage `json:"messages"`
	Stream          bool          `json:"stream"`
	MaxTokens       int           `json:"max_tokens"`
	Temperature     float64       `json:"temperature"`
	TopP            float64       `json:"top_p"`
	Stateless       bool          `json:"stateless,omitempty"`        // bypass KV cache (for testing)
	EnableThinking  *bool         `json:"enable_thinking,omitempty"`  // true/false/null; null = use server default
	ElideThinking   *bool         `json:"elide_thinking,omitempty"`  // true/false/null; null = use server default
}

// --- Non-streaming response types ---

type choiceMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type choice struct {
	Index        int           `json:"index"`
	Message      choiceMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

type usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
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
	var req chatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	// Resolve default model
	if req.Model == "" || req.Model == "default" {
		req.Model = s.resolveDefaultModel()
	}
	if req.Model == "" || s.manager.Get(req.Model) == nil {
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
	if req.Temperature > 0 {
		params.Temperature = float32(req.Temperature)
	}
	if req.TopP > 0 {
		params.TopP = float32(req.TopP)
	}
	params.Stateless = req.Stateless

	// Convert messages and inject /no_think if needed
	msgs := make([]inference.ChatMessage, len(req.Messages))
	for i, m := range req.Messages {
		msgs[i] = inference.ChatMessage{Role: m.Role, Content: m.Content}
	}
	msgs = applyThinkingMode(msgs, params.ThinkingEnabled, eng.NoThinkInstruction())


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
			log.Printf("[metrics] prompt=%d completion=%d prefill=%.1fms decode=%.1fms total=%.1fms tok/s=%.1f",
				metrics.PromptTokens, metrics.CompletionTokens,
				float64(metrics.PrefillDuration.Microseconds())/1000,
				float64(metrics.DecodeDuration.Microseconds())/1000,
				float64(metrics.TotalDuration.Microseconds())/1000,
				metrics.TokensPerSec())
		}
		if elide {
			if out := filter.flush(); out != "" {
				sendChunk(deltaContent{Content: out}, nil)
			}
		}
		if s.cfg.Inference.LogThinking {
			filter.logThinking(req.Model)
		}

		stop := "stop"
		if genErr != nil {
			stop = "error"
		}
		sendChunk(deltaContent{}, &stop)
		fmt.Fprintf(w, "data: [DONE]\n\n")
		if canFlush {
			flusher.Flush()
		}

	} else {
		// --- Non-streaming ---
		var sb []byte
		completionTokens := 0

		metrics, genErr := eng.Generate(msgs, params, func(token string) bool {
			sb = append(sb, token...)
			completionTokens++
			return true
		})
		if genErr != nil {
			writeError(w, http.StatusInternalServerError, genErr.Error())
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
			log.Printf("[metrics] prompt=%d completion=%d prefill=%.1fms decode=%.1fms total=%.1fms tok/s=%.1f",
				metrics.PromptTokens, metrics.CompletionTokens,
				float64(metrics.PrefillDuration.Microseconds())/1000,
				float64(metrics.DecodeDuration.Microseconds())/1000,
				float64(metrics.TotalDuration.Microseconds())/1000,
				metrics.TokensPerSec())
		}

		resp := chatCompletionResponse{
			ID:      chunkID,
			Object:  "chat.completion",
			Created: created,
			Model:   req.Model,
			Choices: []choice{{
				Index:        0,
				Message:      choiceMessage{Role: "assistant", Content: content},
				FinishReason: "stop",
			}},
			Usage: usage{
				PromptTokens:     metrics.PromptTokens,
				CompletionTokens: metrics.CompletionTokens,
				TotalTokens:      metrics.PromptTokens + metrics.CompletionTokens,
			},
		}
		util.WriteJSON(w, resp)
	}
}

// applyThinkingMode appends the model-specific no-think instruction to the last user message
// when thinking is disabled. The instruction is auto-detected from the model's vocabulary.
func applyThinkingMode(msgs []inference.ChatMessage, thinking bool, noThinkInstr string) []inference.ChatMessage {
	if thinking || noThinkInstr == "" {
		return msgs
	}
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "user" {
			out := make([]inference.ChatMessage, len(msgs))
			copy(out, msgs)
			out[i].Content = out[i].Content + "\n" + noThinkInstr
			return out
		}
	}
	return msgs
}

func writeError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	fmt.Fprintf(w, `{"error":{"message":%q,"type":"api_error"}}`, msg)
}
