package culling

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	log "inference-lab-bench/internal/log"
	"inference-lab-bench/internal/inference/arch"
	"inference-lab-bench/internal/util"
)

// WriteCullDiagnostics saves the culled module map and SVG diagram alongside the model GGUF.
// Writes timestamped files plus non-timestamped "latest" copies for browser refresh.
func WriteCullDiagnostics(mm *arch.ModuleMap, modelPath string, tensorDims arch.TensorDimsMap, engagement *arch.EngagementData) {
	paths := util.ResolvePaths()
	base := strings.TrimSuffix(filepath.Base(modelPath), filepath.Ext(modelPath))
	base = filepath.Join(paths.DiagDir, base)

	ts := time.Now().Format("20060102-150405")

	tomlPath := fmt.Sprintf("%s.%s.cullmap.toml", base, ts)
	svgPath := fmt.Sprintf("%s.%s.cullmap.svg", base, ts)
	latestToml := base + ".cullmap.toml"
	latestSvg := base + ".cullmap.svg"

	if err := mm.Save(tomlPath); err != nil {
		log.Warn("failed to write cullmap TOML: %v", err)
		return
	}

	title := strings.TrimSuffix(filepath.Base(modelPath), filepath.Ext(modelPath))
	title = title + " [" + mm.Method + "]"
	if err := arch.RenderModuleMapDiagram(mm, svgPath, title, tensorDims, engagement); err != nil {
		log.Warn("failed to render cullmap SVG: %v", err)
		return
	}

	// Copy timestamped files to non-timestamped "latest" for browser refresh.
	util.CopyFile(tomlPath, latestToml)
	util.CopyFile(svgPath, latestSvg)

	log.Info("cullmap written: %s", tomlPath)
}

// WriteEngagementDiag writes an engagement-shaded module map SVG and an HTML auto-refresh
// wrapper to the diagnostics directory. Called after stateless inference to visualize
// per-layer engagement. The HTML file polls the SVG every second for live updates.
func WriteEngagementDiag(mm *arch.ModuleMap, modelPath string, tensorDims arch.TensorDimsMap, engagement *arch.EngagementData) {
	if mm == nil {
		return
	}
	paths := util.ResolvePaths()
	os.MkdirAll(paths.DiagDir, 0755) // ensure dir exists

	stem := strings.TrimSuffix(filepath.Base(modelPath), filepath.Ext(modelPath))
	base := filepath.Join(paths.DiagDir, stem)

	svgPath := base + ".engagement.svg"
	htmlPath := base + ".engagement.html"

	title := fmt.Sprintf("%s [engagement %s]", stem, time.Now().Format("15:04:05"))
	if err := arch.RenderModuleMapDiagram(mm, svgPath, title, tensorDims, engagement); err != nil {
		log.Warn("engagement SVG failed: %v", err)
		return
	}

	svgFile := filepath.Base(svgPath)
	html := strings.NewReplacer("{{TITLE}}", title, "{{SVG}}", svgFile).Replace(engagementHTML)
	os.WriteFile(htmlPath, []byte(html), 0644)

	log.Info("engagement written: %s", svgPath)
}

const engagementHTML = `<!DOCTYPE html>
<html><head><title>{{TITLE}}</title>
<style>body{margin:0;background:#fafafa;display:flex;justify-content:center}</style>
</head><body>
<img id="svg" src="{{SVG}}" style="max-width:100%">
<script>
const img=document.getElementById('svg'),base=img.src.split('?')[0];
setInterval(()=>{img.src=base+'?t='+Date.now()},1000);
</script>
</body></html>`
