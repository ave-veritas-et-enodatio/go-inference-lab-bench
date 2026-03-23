package util

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// Custom file extensions used across the project.
const (
	ExtArchToml = ".arch.toml"  // architecture definition
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

var (
	resolvedPaths BenchPaths
	resolveOnce   sync.Once
)

// ResolvePaths computes standard paths relative to the running executable.
// Falls back to the current directory if the executable path cannot be determined.
// The result is computed once and cached for the lifetime of the process.
func ResolvePaths() BenchPaths {
	resolveOnce.Do(func() {
		exeDir := "."
		if exe, err := os.Executable(); err == nil {
			exeDir = filepath.Dir(exe)
		}
		resolvedPaths = BenchPaths{
			ExeDir:    exeDir,
			ConfigDir: filepath.Join(exeDir, "config"),
			ModelsDir: filepath.Join(exeDir, "models"),
			ArchDir:   filepath.Join(exeDir, "models", "arch"),
			DiagDir:   filepath.Join(exeDir, "diag"),
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
