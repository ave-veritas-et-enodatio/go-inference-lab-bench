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
	ExtArchToml       = ".arch.toml" // architecture definition
	ExtSafetensorsDir = ".st"        // safetensors model directory
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
	resolveErr    error
	resolveOnce   sync.Once
)

func IsDir(path string) bool {
	stat, err := os.Stat(path)
	return err == nil && stat.IsDir()
}

func getExeDir() (string, error) {
	exe, err := os.Executable()
	if err != nil {
		return "", fmt.Errorf("determine executable path: %w", err)
	}
	return filepath.Dir(exe), nil
}

// ResolvePaths computes standard paths relative to the running executable.
// Falls back to the current directory if the executable path cannot be determined.
// The result is computed once and cached for the lifetime of the process —
// subsequent calls return the same (paths, err) pair.
// Handles hacky debugging envar BENCH_EXE_DIR because debug.json configs
// don't let you separate the source level CWD from runtime CWD.
func ResolvePaths() (BenchPaths, error) {
	resolveOnce.Do(func() {
		if exeDir := os.Getenv("BENCH_EXE_DIR"); exeDir != "" {
			// hacky debug envar was provided
			log.Info("using BENCH_EXE_DIR=%s instead of getExeDir()", exeDir)
			resolvedPaths = makeBenchPaths(exeDir)
		} else {
			// normal path
			exeDir, err := getExeDir()
			if err != nil {
				resolveErr = err
				return
			}
			resolvedPaths = makeBenchPaths(exeDir)
		}
		configRel, _ := filepath.Rel(resolvedPaths.ExeDir, resolvedPaths.ConfigDir)
		if !IsDir(resolvedPaths.ConfigDir) {
			cwd, err := os.Getwd()
			if err != nil {
				resolveErr = fmt.Errorf("determine current directory: %w", err)
				return
			}
			log.Error("%s does not contain %s falling back to %s",
				resolvedPaths.ExeDir, configRel, cwd)
			resolvedPaths = makeBenchPaths(cwd)
		}
		if !IsDir(resolvedPaths.ConfigDir) {
			log.Error("%s does not contain %s",
				resolvedPaths.ExeDir, configRel)
		}
	})
	return resolvedPaths, resolveErr
}

// EnsureDiagDir creates the diagnostics output directory if it doesn't exist.
// Call once at server startup — not on every request.
func EnsureDiagDir(paths BenchPaths) error {
	if err := os.MkdirAll(paths.DiagDir, 0755); err != nil {
		return fmt.Errorf("creating diag dir %s: %w", paths.DiagDir, err)
	}
	return nil
}
