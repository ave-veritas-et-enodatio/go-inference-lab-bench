package main

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"

	"inference-lab-bench/internal/inference/arch"
	"inference-lab-bench/internal/inference/archdiagram"
	log "inference-lab-bench/internal/log"
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
		log.Fatal("reading %s: %v", inputPath, err)
	}

	def, err := arch.Parse(data)
	if err != nil {
		log.Fatal("parsing %s: %v", inputPath, err)
	}

	if svgLayers == 0 && def.Example.NLayers > 0 {
		svgLayers = def.Example.NLayers
	}

	f, err := os.Create(outputPath)
	if err != nil {
		log.Fatal("creating %s: %v", outputPath, err)
	}
	defer f.Close()

	opts := archdiagram.ArchDiagramOptions{LayerCount: svgLayers}
	if err := archdiagram.RenderArchDiagram(def, svgBlockSVGDir, f, opts); err != nil {
		os.Remove(outputPath)
		log.Fatal("generating SVG: %v", err)
	}

	log.Info("wrote %s", outputPath)

	// Generate module-map layers diagram if layer count is known
	if svgLayers > 0 {
		weights := archdiagram.ResolveWeightsForDiagram(def, svgLayers)
		// Strip FFNAlt so the standard layers diagram shows dense FFN only.
		if def.FFNAlt != nil {
			for i := range weights.Layers {
				weights.Layers[i].FFNAlt = nil
			}
		}
		mm := arch.BuildModuleMap(def, weights)
		layersPath := filepath.Join(filepath.Dir(outputPath), def.Architecture.Name+".layers.svg")
		subtitle := " Layers"
		lf, err := os.Create(layersPath)
		if err != nil {
			log.Fatal("creating %s: %v", layersPath, err)
		}
		defer lf.Close()
		if err := archdiagram.RenderLayersDiagram(def, mm, lf, subtitle); err != nil {
			log.Fatal("generating layers SVG: %v", err)
		}
		log.Info("wrote %s", layersPath)
	}

	// Generate vision-tower diagram if the arch has a [vision] section.
	if def.Vision != nil {
		visionPath := filepath.Join(filepath.Dir(outputPath), def.Architecture.Name+".vision.svg")
		vf, err := os.Create(visionPath)
		if err != nil {
			log.Fatal("creating %s: %v", visionPath, err)
		}
		defer vf.Close()
		if err := archdiagram.RenderVisionDiagram(def, vf); err != nil {
			os.Remove(visionPath)
			log.Fatal("generating vision SVG: %v", err)
		}
		log.Info("wrote %s", visionPath)

		// Fully-exploded per-layer vision diagram. The runtime block_count is
		// not statically known (no GGUF here), so it uses the example depth —
		// same reason the decoder layers diagram uses example.n_layers. Skipped
		// when no example vision depth is set (don't invent a count).
		if def.Example.VisionNLayers > 0 {
			visionLayersPath := filepath.Join(filepath.Dir(outputPath), def.Architecture.Name+".vision.layers.svg")
			vlf, err := os.Create(visionLayersPath)
			if err != nil {
				log.Fatal("creating %s: %v", visionLayersPath, err)
			}
			defer vlf.Close()
			if err := archdiagram.RenderVisionLayersDiagram(def, def.Example.VisionNLayers, vlf); err != nil {
				os.Remove(visionLayersPath)
				log.Fatal("generating vision layers SVG: %v", err)
			}
			log.Info("wrote %s", visionLayersPath)
		}
	}

	// Generate alt FFN variant diagram if the arch has [ffn_alt]
	if def.FFNAlt != nil {
		altSuffix := "-" + def.FFNAlt.Builder
		altBase := strings.TrimSuffix(inputPath, ".arch.toml")
		altArchPath := altBase + altSuffix + ".arch.svg"

		af, err := os.Create(altArchPath)
		if err != nil {
			log.Fatal("creating %s: %v", altArchPath, err)
		}
		defer af.Close()

		altOpts := archdiagram.ArchDiagramOptions{LayerCount: svgLayers, UseFFNAlt: true}
		if err := archdiagram.RenderArchDiagram(def, svgBlockSVGDir, af, altOpts); err != nil {
			os.Remove(altArchPath)
			log.Fatal("generating alt FFN SVG: %v", err)
		}
		log.Info("wrote %s", altArchPath)

		// Alt layers diagram (with FFNAlt → MoE modules)
		if svgLayers > 0 {
			altWeights := archdiagram.ResolveWeightsForDiagram(def, svgLayers)
			altMM := arch.BuildModuleMap(def, altWeights)
			altLayersPath := filepath.Join(filepath.Dir(outputPath),
				def.Architecture.Name+altSuffix+".layers.svg")
			subtitle := " Layers (" + def.FFNAlt.Builder + " variant)"
			alf, err := os.Create(altLayersPath)
			if err != nil {
				log.Fatal("creating %s: %v", altLayersPath, err)
			}
			defer alf.Close()
			if err := archdiagram.RenderLayersDiagram(def, altMM, alf, subtitle); err != nil {
				log.Fatal("generating alt layers SVG: %v", err)
			}
			log.Info("wrote %s", altLayersPath)
		}
	}
}
