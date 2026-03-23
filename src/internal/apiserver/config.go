package apiserver

import (
	"fmt"

	"github.com/BurntSushi/toml"
)

type Config struct {
	Server    ServerConfig    `toml:"server"`
	Models    ModelsConfig    `toml:"models"`
	Inference InferenceConfig `toml:"inference"`
}

type ServerConfig struct {
	Host      string `toml:"host"`
	Port      int    `toml:"port"`
	AuthToken string `toml:"auth_token"`
}

type ModelsConfig struct {
	Directory string `toml:"directory"`
	Default   string `toml:"default"` // "first", "last", or explicit model name
}

type InferenceConfig struct {
	NThreads             int    `toml:"n_threads"`
	LogThinking             bool   `toml:"log_thinking"`
	EnableThinkingDefault bool  `toml:"enable_thinking_default"`
	MaxSeqLen            int    `toml:"max_seq_len"`    // KV cache size in tokens (default: 4096)
	ElideThinkingDefault *bool  `toml:"elide_thinking_default"` // nil = default true; strip <think>...</think> from output
}

// ShouldElideThink returns true if <think> content should be stripped from output (default: true).
func (c *InferenceConfig) ShouldElideThink() bool {
	if c.ElideThinkingDefault == nil {
		return true
	}
	return *c.ElideThinkingDefault
}

func LoadConfig(path string) (*Config, error) {
	cfg := &Config{
		Server: ServerConfig{
			Host: "0.0.0.0",
			Port: 11116,
		},
		Models: ModelsConfig{
			Directory: "models",
			Default:   "first",
		},
	}
	md, err := toml.DecodeFile(path, cfg)
	if err != nil {
		return nil, fmt.Errorf("parsing config: %w", err)
	}

	// Warn on unknown keys
	if undecoded := md.Undecoded(); len(undecoded) > 0 {
		for _, key := range undecoded {
			fmt.Printf("warning: unknown config key: %s\n", key)
		}
	}

	if err := cfg.validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}
	return cfg, nil
}

func (c *Config) validate() error {
	// server
	if c.Server.Host == "" {
		return fmt.Errorf("server.host must not be empty")
	}
	if c.Server.Port < 1 || c.Server.Port > 65535 {
		return fmt.Errorf("server.port must be 1-65535, got %d", c.Server.Port)
	}

	// models
	if c.Models.Directory == "" {
		return fmt.Errorf("models.directory must not be empty")
	}

	// inference
	if c.Inference.NThreads < 0 {
		return fmt.Errorf("inference.n_threads must be >= 0, got %d", c.Inference.NThreads)
	}

	return nil
}
