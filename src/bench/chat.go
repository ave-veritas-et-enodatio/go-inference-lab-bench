package main

import (
	"path/filepath"

	"github.com/spf13/cobra"

	"inference-lab-bench/internal/chatclient"
	"inference-lab-bench/internal/util"
)

var chatCmd = &cobra.Command{
	Use:   "chat",
	Short: "Interactive chat client",
	RunE:  runChat,
}

func init() {
	rootCmd.AddCommand(chatCmd)
}

func runChat(cmd *cobra.Command, args []string) error {
	paths := util.ResolvePaths()
	return chatclient.Run(chatclient.Options{
		ConfigPath:    filepath.Join(paths.ConfigDir, "chat_config.toml"),
		ApiConfigPath: filepath.Join(paths.ConfigDir, "api_config.toml"),
		HistoryDbPath: filepath.Join(paths.ExeDir, "chat_history.db"),
	})
}
