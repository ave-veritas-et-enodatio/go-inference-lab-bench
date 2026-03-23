package main

import (
	"log"
	"path/filepath"

	"github.com/spf13/cobra"

	"inference-lab-bench/internal/apiserver"
	"inference-lab-bench/internal/model"
	"inference-lab-bench/internal/util"
)

var serveAPICmd = &cobra.Command{
	Use:   "serve-api",
	Short: "Run the inference API server",
	Run:   runServeAPI,
}

var (
	apiConfigPath string
	apiHost       string
	apiPort       int
)

func init() {
	paths := util.ResolvePaths()
	serveAPICmd.Flags().StringVar(&apiConfigPath, "config", filepath.Join(paths.ConfigDir, "api_config.toml"), "path to API config file")
	serveAPICmd.Flags().StringVar(&apiHost, "host", "", "override listen host")
	serveAPICmd.Flags().IntVar(&apiPort, "port", 0, "override listen port")
	rootCmd.AddCommand(serveAPICmd)
}

func runServeAPI(cmd *cobra.Command, args []string) {
	paths := util.ResolvePaths()
	if err := util.EnsureDiagDir(paths); err != nil {
		log.Fatalf("pre-init: %v", err)
	}

	cfg, err := apiserver.LoadConfig(apiConfigPath)
	if err != nil {
		log.Fatalf("loading config: %v", err)
	}
	if apiHost != "" {
		cfg.Server.Host = apiHost
	}
	if apiPort != 0 {
		cfg.Server.Port = apiPort
	}

	modelsDir := cfg.Models.Directory
	if !filepath.IsAbs(modelsDir) {
		modelsDir = filepath.Join(paths.ExeDir, modelsDir)
	}
	manager, err := model.NewManager(modelsDir)
	if err != nil {
		log.Fatalf("initializing model manager: %v", err)
	}

	srv := apiserver.NewServer(cfg, manager)
	if err := srv.Run(); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
