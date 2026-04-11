package main

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"

	log "inference-lab-bench/internal/log"
	"inference-lab-bench/internal/chatclient"
	"inference-lab-bench/internal/util"
)

var chatCmd = &cobra.Command{
	Use:   "chat",
	Short: "Interactive chat client",
	RunE:  runChat,
}

var (
	chatLogPath  string
	chatLogLevel string
	chatLogFileLine bool
)

func init() {
	chatCmd.Flags().StringVar(&chatLogPath, "log", "", "path to log file (tee with stderr)")
	chatCmd.Flags().StringVar(&chatLogLevel, "log-level", "INFO", "stderr log level ("+strings.Join(log.ValidLevelNames, "|")+")")
	chatCmd.Flags().BoolVar(&chatLogFileLine, "log-file-line", false, "include file and line number in log messages")
	rootCmd.AddCommand(chatCmd)
}

func runChat(cmd *cobra.Command, args []string) error {
	level, ok := log.ParseLevel(chatLogLevel)
	if !ok {
		return fmt.Errorf("invalid --log-level %q: valid values: %s", chatLogLevel, strings.Join(log.ValidLevelNames, ", "))
	}
	if err := log.InitLogger(chatLogPath, level, chatLogFileLine); err != nil {
		return fmt.Errorf("init logger: %w", err)
	}
	paths := util.ResolvePaths()
	return chatclient.Run(chatclient.Options{
		ConfigPath:    filepath.Join(paths.ConfigDir, "chat_config.toml"),
		ApiConfigPath: filepath.Join(paths.ConfigDir, "api_config.toml"),
		HistoryDbPath: filepath.Join(paths.ExeDir, "chat_history.db"),
	})
}
