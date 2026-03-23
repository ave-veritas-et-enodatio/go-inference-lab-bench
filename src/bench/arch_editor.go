package main

import (
	"log"

	"github.com/spf13/cobra"

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
)

func init() {
	defEditorCmd.Flags().StringVar(&editorArchDir, "arch", "", "path to models/arch directory (default: models/arch next to executable)")
	defEditorCmd.Flags().StringVar(&editorStaticDir, "static", "", "path to static files directory (default: editor next to arch)")
	defEditorCmd.Flags().IntVar(&editorPort, "port", 8080, "localhost port to listen on")
	rootCmd.AddCommand(defEditorCmd)
}

func runArchEditor(cmd *cobra.Command, args []string) {
	paths := util.ResolvePaths()
	if editorArchDir == "" {
		editorArchDir = paths.ArchDir
	}
	if editorStaticDir == "" {
		editorStaticDir = editorArchDir + "/editor"
	}

	if err := archeditor.Run(editorArchDir, editorStaticDir, editorPort); err != nil {
		log.Fatalf("arch-editor: %v", err)
	}
}
