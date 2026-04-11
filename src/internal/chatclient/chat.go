package chatclient

import (
	"bufio"
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	log "inference-lab-bench/internal/log"
	"inference-lab-bench/internal/util"

	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/memory"
	"github.com/tmc/langchaingo/memory/sqlite3"

	_ "github.com/mattn/go-sqlite3"
)

// Config holds chat client configuration from chat_config.toml.
type Config struct {
	BaseURL        string   `toml:"base_url"`
	Token          string   `toml:"token"`
	Model          string   `toml:"model"`
	SystemPrompt   []string `toml:"system_prompt"`
	ThinkingBudget *int     `toml:"thinking_budget"`
}

// Options holds runtime parameters for the chat client.
type Options struct {
	ConfigPath    string // path to chat_config.toml
	ApiConfigPath string // path to api_config.toml (for auto base URL resolution)
	HistoryDbPath string // path to chat history SQLite database
}

// thinkingTransport injects thinking_budget into every chat completion request body.
type thinkingTransport struct {
	base   http.RoundTripper
	budget int
}

func (t *thinkingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.Body != nil {
		body, err := io.ReadAll(req.Body)
		req.Body.Close()
		if err == nil {
			var m map[string]any
			if json.Unmarshal(body, &m) == nil {
				m["thinking_budget"] = t.budget
				if b, err := json.Marshal(m); err == nil {
					body = b
				}
			}
			req.Body = io.NopCloser(bytes.NewReader(body))
			req.ContentLength = int64(len(body))
		}
	}
	return t.base.RoundTrip(req)
}

func insertChatMessage(ctx context.Context, db *sql.DB, content string, msgType llms.ChatMessageType) {
	db.ExecContext(ctx,
		"INSERT INTO langchaingo_messages(session, content, type) VALUES (?, ?, ?)",
		"example", content, msgType)
}

// ResolveAutoBaseURL reads api_config.toml to construct the API base URL.
func ResolveAutoBaseURL(apiConfigPath string) (string, error) {
	type apiCfg struct {
		Server struct {
			Port int `toml:"port"`
		} `toml:"server"`
	}
	var ac apiCfg
	if err := util.LoadTOML(apiConfigPath, &ac); err != nil {
		return "", err
	}
	port := ac.Server.Port
	if port == 0 {
		port = 11116
	}
	return fmt.Sprintf("http://localhost:%d/api/v1", port), nil
}

// Run starts the interactive chat client.
func Run(opts Options) error {
	var cfg Config
	if err := util.LoadTOML(opts.ConfigPath, &cfg); err != nil {
		return err
	}

	// Resolve "auto" base URL from api_config.toml
	if cfg.BaseURL == "auto" {
		var err error
		cfg.BaseURL, err = ResolveAutoBaseURL(opts.ApiConfigPath)
		if err != nil {
			return fmt.Errorf("resolving auto base URL: %w", err)
		}
	}

	llmOpts := []openai.Option{
		openai.WithBaseURL(cfg.BaseURL),
		openai.WithToken(cfg.Token),
		openai.WithModel(cfg.Model),
	}
	if cfg.ThinkingBudget != nil {
		llmOpts = append(llmOpts, openai.WithHTTPClient(&http.Client{
			Transport: &thinkingTransport{
				base:   http.DefaultTransport,
				budget: *cfg.ThinkingBudget,
			},
		}))
	}
	llm, err := openai.New(llmOpts...)
	if err != nil {
		return fmt.Errorf("creating LLM client: %w", err)
	}

	db, err := sql.Open("sqlite3", opts.HistoryDbPath)
	if err != nil {
		return fmt.Errorf("opening database: %w", err)
	}

	chatHistory := sqlite3.NewSqliteChatMessageHistory(
		sqlite3.WithSession("example"),
		sqlite3.WithDB(db),
	)
	conversationBuffer := memory.NewConversationBuffer(memory.WithChatHistory(chatHistory))
	llmChain := chains.NewConversation(llm, conversationBuffer)
	ctx := context.Background()

	// Seed system prompt if DB is empty
	var count int
	if err := db.QueryRowContext(ctx, "SELECT count(id) FROM langchaingo_messages").Scan(&count); err == nil && count == 0 {
		if len(cfg.SystemPrompt) > 0 {
			insertChatMessage(ctx, db, strings.Join(cfg.SystemPrompt, "\n"), llms.ChatMessageTypeSystem)
		}
	}

	fmt.Println("Chat started. Type 'exit' or 'quit' to end.")
	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Print("> ")
		input, err := reader.ReadString('\n')
		if err != nil {
			log.Error("Error reading input: %v", err)
			continue
		}
		input = strings.TrimSpace(input)

		if input == "exit" || input == "quit" {
			fmt.Println("Goodbye!")
			break
		}
		if input == "" {
			continue
		}

		out, err := chains.Run(ctx, llmChain, input)
		if err != nil {
			log.Error("Error generating response: %v", err)
			continue
		}

		fmt.Println(out)
	}

	return nil
}
