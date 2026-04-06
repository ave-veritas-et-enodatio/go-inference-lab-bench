package main

import (
	"strings"

	"github.com/spf13/cobra"

	log "inference-lab-bench/internal/log"
	"inference-lab-bench/internal/archeditor"
	"inference-lab-bench/internal/util"
)

var defEditorCmd = &cobra.Command{
	Use:   "arch-editor",
	Short: "Launch interactive TOML definition editor",
	Run:   runArchEditor,
}

var (
	editorArchDir   string
	editorStaticDir string
	editorPort      int
	editorLogPath   string
	editorLogLevel  string
)

func init() {
	defEditorCmd.Flags().StringVar(&editorArchDir, "arch", "", "path to models/arch directory (default: models/arch next to executable)")
	defEditorCmd.Flags().StringVar(&editorStaticDir, "static", "", "path to static files directory (default: editor next to arch)")
	defEditorCmd.Flags().IntVar(&editorPort, "port", 8080, "localhost port to listen on")
	defEditorCmd.Flags().StringVar(&editorLogPath, "log", "", "path to log file (tee with stderr)")
	defEditorCmd.Flags().StringVar(&editorLogLevel, "log-level", "INFO", "stderr log level ("+strings.Join(log.ValidLevelNames, "|")+")")
	rootCmd.AddCommand(defEditorCmd)
}

func runArchEditor(cmd *cobra.Command, args []string) {
	level, ok := log.ParseLevel(editorLogLevel)
	if !ok {
		log.Fatal("invalid --log-level %q: valid values: %s", editorLogLevel, strings.Join(log.ValidLevelNames, ", "))
	}
	if err := log.InitLogger(editorLogPath, level); err != nil {
		log.Fatal("init logger: %v", err)
	}
	paths := util.ResolvePaths()
	if editorArchDir == "" {
		editorArchDir = paths.ArchDir
	}
	if editorStaticDir == "" {
		editorStaticDir = editorArchDir + "/editor"
	}

	if err := archeditor.Run(editorArchDir, editorStaticDir, editorPort); err != nil {
		log.Fatal("arch-editor: %v", err)
	}
}
