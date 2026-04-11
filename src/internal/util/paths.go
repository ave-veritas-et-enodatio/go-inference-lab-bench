package util

import (
	"fmt"
	"inference-lab-bench/internal/log"
	"os"
	"path/filepath"
	"sync"
)

// Custom file extensions used across the project.
const (
	ExtArchToml = ".arch.toml"  // architecture definition
	ExtCullMeta = ".cullmeta"   // culling metadata sidecar
)

// benchPaths holds standard directory paths derived from the executable location.
// All subcommands use this to find config, models, and arch definitions without
// hardcoding paths or requiring flags.
type BenchPaths struct {
	ExeDir    string // directory containing the bench binary
	ConfigDir string // ExeDir/config - configuration files
	ModelsDir string // ExeDir/models - model files
	ArchDir   string // ExeDir/models/arch - architecture definitions
	DiagDir   string // ExeDir/diag - diagnostic output
}

func makeBenchPaths(exeDir string) BenchPaths {
	return BenchPaths{
		ExeDir:    exeDir,
		ConfigDir: filepath.Join(exeDir, "config"),
		ModelsDir: filepath.Join(exeDir, "models"),
		ArchDir:   filepath.Join(exeDir, "models/arch"),
		DiagDir:   filepath.Join(exeDir, "diag"),
	}
}

var (
	resolvedPaths BenchPaths
	resolveOnce   sync.Once
)

func IsDir(path string) bool {
	stat, err := os.Stat(path)
	return err == nil && stat.IsDir()
}

func getExeDir() string {
	if exe, err := os.Executable(); err == nil {
		exeDir := filepath.Dir(exe)
		return exeDir
	} else {
		log.Fatal("failed to determine executable path: %v", err)
	}
	return ""
}

// ResolvePaths computes standard paths relative to the running executable.
// Falls back to the current directory if the executable path cannot be determined.
// The result is computed once and cached for the lifetime of the process.
// handles hacky debugging envar BENCH_EXE_DIR because debug.json configs
// don't let you separate the source level CWD from runtime CWD
func ResolvePaths() BenchPaths {
	resolveOnce.Do(func() {
		if exeDir := os.Getenv("BENCH_EXE_DIR"); exeDir != "" {
			// hacky debug envar was provided
			log.Info("using BENCH_EXE_DIR=%s instead of getExeDir()", exeDir)
			resolvedPaths = makeBenchPaths(exeDir)
		} else {
			// normal path
			resolvedPaths = makeBenchPaths(getExeDir())
		}
		configRel, _ := filepath.Rel(resolvedPaths.ExeDir, resolvedPaths.ConfigDir)
		if !IsDir(resolvedPaths.ConfigDir) {
			if cwd, err := os.Getwd(); err == nil {
				log.Error("%s does not contain %s falling back to %s",
					resolvedPaths.ExeDir, configRel, cwd)
				resolvedPaths = makeBenchPaths(cwd)
			} else {
				log.Fatal("failed to determine current directory: %v", err)
			}
		}
		if !IsDir(resolvedPaths.ConfigDir) {
			log.Error("%s does not contain %s",
				resolvedPaths.ExeDir, configRel)
		}
	})
	return resolvedPaths
}

// EnsureDiagDir creates the diagnostics output directory if it doesn't exist.
// Call once at server startup — not on every request.
func EnsureDiagDir(paths BenchPaths) error {
	if err := os.MkdirAll(paths.DiagDir, 0755); err != nil {
		return fmt.Errorf("creating diag dir %s: %w", paths.DiagDir, err)
	}
	return nil
}
