package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"

	"inference-lab-bench/internal/inference/arch"
)

var defToSVGCmd = &cobra.Command{
	Use:   "gen-arch-diagram [flags] <input.toml> [output.svg]",
	Short: "Generate SVG architecture diagram from a TOML definition",
	Args:  cobra.RangeArgs(1, 2),
	Run:   runRenderDiagram,
}

var (
	svgLayers      int
	svgBlockSVGDir string
)

func init() {
	defToSVGCmd.Flags().IntVar(&svgLayers, "layers", 0, "layer count for pattern strip (0 = omit)")
	defToSVGCmd.Flags().StringVar(&svgBlockSVGDir, "blocks", "", "directory containing block SVG fragments (default: block_svg/ next to input)")
	rootCmd.AddCommand(defToSVGCmd)
}

func runRenderDiagram(cmd *cobra.Command, args []string) {
	inputPath := args[0]
	outputPath := ""
	if len(args) > 1 {
		outputPath = args[1]
	}
	if outputPath == "" {
		outputPath = strings.TrimSuffix(inputPath, filepath.Ext(inputPath)) + ".svg"
	}
	if svgBlockSVGDir == "" {
		svgBlockSVGDir = filepath.Join(filepath.Dir(inputPath), "block_svg")
	}

	data, err := os.ReadFile(inputPath)
	if err != nil {
		log.Fatalf("reading %s: %v", inputPath, err)
	}

	def, err := arch.Parse(data)
	if err != nil {
		log.Fatalf("parsing %s: %v", inputPath, err)
	}

	f, err := os.Create(outputPath)
	if err != nil {
		log.Fatalf("creating %s: %v", outputPath, err)
	}
	defer f.Close()

	opts := arch.ArchDiagramOptions{LayerCount: svgLayers}
	if err := arch.RenderArchDiagram(def, svgBlockSVGDir, f, opts); err != nil {
		os.Remove(outputPath)
		log.Fatalf("generating SVG: %v", err)
	}

	fmt.Fprintf(os.Stderr, "wrote %s\n", outputPath)

	// Generate module-map layers diagram if layer count is known
	if svgLayers > 0 {
		weights := arch.ResolveWeightsFromDef(def, svgLayers)
		// Strip FFNAlt so the standard layers diagram shows dense FFN only.
		if def.FFNAlt != nil {
			for i := range weights.Layers {
				weights.Layers[i].FFNAlt = nil
			}
		}
		mm := arch.BuildModuleMap(weights)
		layersPath := filepath.Join(filepath.Dir(outputPath), def.Architecture.Name+".layers.svg")
		if err := arch.RenderModuleMapDiagram(mm, layersPath, strings.Title(def.Architecture.Name)+" Layers", nil); err != nil {
			log.Fatalf("generating layers SVG: %v", err)
		}
		fmt.Fprintf(os.Stderr, "wrote %s\n", layersPath)
	}

	// Generate alt FFN variant diagram if the arch has [ffn_alt]
	if def.FFNAlt != nil {
		altSuffix := "-" + def.FFNAlt.Builder
		altBase := strings.TrimSuffix(inputPath, ".arch.toml")
		altArchPath := altBase + altSuffix + ".arch.svg"

		af, err := os.Create(altArchPath)
		if err != nil {
			log.Fatalf("creating %s: %v", altArchPath, err)
		}
		defer af.Close()

		altOpts := arch.ArchDiagramOptions{LayerCount: svgLayers, UseFFNAlt: true}
		if err := arch.RenderArchDiagram(def, svgBlockSVGDir, af, altOpts); err != nil {
			os.Remove(altArchPath)
			log.Fatalf("generating alt FFN SVG: %v", err)
		}
		fmt.Fprintf(os.Stderr, "wrote %s\n", altArchPath)

		// Alt layers diagram (with FFNAlt → MoE modules)
		if svgLayers > 0 {
			altWeights := arch.ResolveWeightsFromDef(def, svgLayers)
			altMM := arch.BuildModuleMap(altWeights)
			altLayersPath := filepath.Join(filepath.Dir(outputPath),
				def.Architecture.Name+altSuffix+".layers.svg")
			if err := arch.RenderModuleMapDiagram(altMM, altLayersPath,
				strings.Title(def.Architecture.Name)+" Layers ("+def.FFNAlt.Builder+" variant)", nil); err != nil {
				log.Fatalf("generating alt layers SVG: %v", err)
			}
			fmt.Fprintf(os.Stderr, "wrote %s\n", altLayersPath)
		}
	}
}
