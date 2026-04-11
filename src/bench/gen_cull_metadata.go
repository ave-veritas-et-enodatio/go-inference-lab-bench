package main

import (
	"os"

	"github.com/spf13/cobra"

	log "inference-lab-bench/internal/log"
	"inference-lab-bench/internal/inference/culling"
)

var cullMethod string
var cullCPU bool

var genCullingMetadataCmd = &cobra.Command{
	Use:   "gen-cull-metadata <model.gguf>",
	Short: "Generate culling metadata sidecar for a model",
	Long:  "Computes and stores per-model culling metadata consumed by ApplyCulling at inference time.",
	Args:  cobra.ExactArgs(1),
	Run:   runGenCullMetadata,
}

func init() {
	const options = culling.InattentionCullMethod + "|" + culling.RandomCullMethod
	genCullingMetadataCmd.Flags().StringVar(&cullMethod, "cull-method", "", "culling method name ["+options+"] (required)")
	genCullingMetadataCmd.Flags().BoolVar(&cullCPU, "cpu", false, "use CPU for generation sweep (slower, for debugging/validation)")
	genCullingMetadataCmd.MarkFlagRequired("cull-method")
	rootCmd.AddCommand(genCullingMetadataCmd)
}

func runGenCullMetadata(cmd *cobra.Command, args []string) {
	modelPath := args[0]
	if _, err := os.Stat(modelPath); err != nil {
		log.Fatal("model path: %v", err)
	}

	metaPath := culling.ComputeCullingMeta(modelPath, cullMethod, !cullCPU)
	if metaPath != nil {
		log.Info("culling metadata written to: %s", *metaPath)
	} else {
		log.Fatal("no culling metadata generated for: %s", modelPath)
	}
}
