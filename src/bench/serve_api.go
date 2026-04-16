package main

import (
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"

	log "inference-lab-bench/internal/log"
	ggmlmod "inference-lab-bench/internal/ggml"
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
	apiConfigPath  string
	apiHost        string
	apiPort        int
	serveLogPath   string
	serveLogLevel  string
	serveLogFileLine bool
)

func init() {
	paths, err := util.ResolvePaths()
	if err != nil {
		log.Fatal("resolve paths: %v", err)
	}
	serveAPICmd.Flags().StringVar(&apiConfigPath, "config", filepath.Join(paths.ConfigDir, "api_config.toml"), "path to API config file")
	serveAPICmd.Flags().StringVar(&apiHost, "host", "", "override listen host")
	serveAPICmd.Flags().IntVar(&apiPort, "port", 0, "override listen port")
	serveAPICmd.Flags().StringVar(&serveLogPath, "log", "", "path to log file (tee with stderr)")
	serveAPICmd.Flags().StringVar(&serveLogLevel, "log-level", "INFO", "stderr log level ("+strings.Join(log.ValidLevelNames, "|")+")")
	serveAPICmd.Flags().BoolVar(&serveLogFileLine, "log-file-line", false, "show file and line in log messages")
	rootCmd.AddCommand(serveAPICmd)
}

func runServeAPI(cmd *cobra.Command, args []string) {
	level, ok := log.ParseLevel(serveLogLevel)
	if !ok {
		log.Fatal("invalid --log-level %q: valid values: %s", serveLogLevel, strings.Join(log.ValidLevelNames, ", "))
	}
	if err := log.InitLogger(serveLogPath, level, serveLogFileLine); err != nil {
		log.Fatal("init logger: %v", err)
	}
	ggmlmod.InitLogging()
	paths, err := util.ResolvePaths()
	if err != nil {
		log.Fatal("resolve paths: %v", err)
	}
	if err := util.EnsureDiagDir(paths); err != nil {
		log.Fatal("pre-init: %v", err)
	}

	cfg, err := apiserver.LoadConfig(apiConfigPath)
	if err != nil {
		log.Fatal("loading config: %v", err)
	}
	if apiHost != "" {
		cfg.Server.Host = apiHost
	}
	if apiPort != 0 {
		cfg.Server.Port = apiPort
	}

	manager, err := model.NewManager(paths.ModelsDir)
	if err != nil {
		log.Fatal("initializing model manager: %v", err)
	}

	srv := apiserver.NewServer(paths, cfg, manager)
	if err := srv.Run(); err != nil {
		log.Fatal("server error: %v", err)
	}
}
