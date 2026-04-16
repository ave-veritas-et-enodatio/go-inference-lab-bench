package model

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	log "inference-lab-bench/internal/log"
	"inference-lab-bench/internal/util"
)

type ModelFormat string

const (
	FormatGGUF        ModelFormat = "gguf"
	FormatSafetensors ModelFormat = "safetensors"
)

type ModelInfo struct {
	ID       string // filename without .gguf / directory name without .st
	Path     string // absolute path to .gguf file or .st/ directory
	Format   ModelFormat
	Metadata *GGUFMetadata
	LoadedAt int64 // unix timestamp when this model was first discovered
}

type Manager struct {
	mu      sync.RWMutex
	models  map[string]*ModelInfo
	dir     string
	archDir string
}

func NewManager(dir string) (*Manager, error) {
	archDir := filepath.Join(dir, "arch")
	scanArchDefinitions(archDir)

	m := &Manager{
		models:  make(map[string]*ModelInfo),
		dir:     dir,
		archDir: archDir,
	}
	if err := m.scan(); err != nil {
		return nil, err
	}
	return m, nil
}

// scanArchDefinitions reads *.arch.toml filenames in archDir and populates
// supportedArchitectures. The architecture name is the filename stem
// (e.g. "llada-moe.arch.toml" → "llada-moe").
func scanArchDefinitions(archDir string) {
	paths, _ := filepath.Glob(filepath.Join(archDir, "*"+util.ExtArchToml))
	for _, path := range paths {
		name := strings.TrimSuffix(filepath.Base(path), util.ExtArchToml)
		if name != "" {
			supportedArchitectures[name] = true
		}
	}
}

func (m *Manager) scan() error {
	pattern := filepath.Join(m.dir, "*.gguf")
	paths, err := filepath.Glob(pattern)
	if err != nil {
		return fmt.Errorf("scanning models dir: %w", err)
	}
	for _, path := range paths {
		base := filepath.Base(path)
		id := strings.TrimSuffix(base, ".gguf")
		meta, err := ParseGGUF(path)
		if err != nil {
			log.Info("skipping %s: %v", base, err)
			continue
		}
		if !supportedArchitectures[meta.Architecture] {
			log.Warn("skipping %s: unsupported architecture %q", id, meta.Architecture)
			continue
		}
		m.models[id] = &ModelInfo{ID: id, Path: path, Format: FormatGGUF, Metadata: meta, LoadedAt: time.Now().Unix()}
		log.Info("loaded model: %s (%d tensors)", id, len(meta.Tensors))
	}

	// Safetensors directory scanning.
	entries, err := os.ReadDir(m.dir)
	if err != nil {
		return fmt.Errorf("scanning models dir for safetensors: %w", err)
	}
	log.Debug("safetensors scan: %d entries in %s", len(entries), m.dir)
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), util.ExtSafetensorsDir) {
			continue
		}
		stDir := filepath.Join(m.dir, e.Name())
		fi, err := os.Stat(stDir)
		if err != nil || !fi.IsDir() {
			continue
		}
		id := strings.TrimSuffix(e.Name(), util.ExtSafetensorsDir)
		log.Debug("safetensors candidate: id=%q dir=%s", id, stDir)
		if strings.Contains(id, "..") {
			log.Debug("safetensors skip (has ..): %s", id)
			continue
		}
		meta, err := ParseSafetensorsDir(stDir, m.archDir)
		if err != nil {
			log.Info("skipping %s: %v", e.Name(), err)
			continue
		}
		log.Debug("safetensors parsed: id=%q arch=%q tensors=%d", id, meta.Architecture, len(meta.Tensors))
		if !supportedArchitectures[meta.Architecture] {
			log.Warn("skipping %s: unsupported architecture %q", id, meta.Architecture)
			continue
		}
		if existing, exists := m.models[id]; exists {
			log.Debug("skipping %s: GGUF version already loaded (format=%s)", e.Name(), existing.Format)
			continue
		}
		m.models[id] = &ModelInfo{ID: id, Path: stDir, Format: FormatSafetensors, Metadata: meta, LoadedAt: time.Now().Unix()}
		log.Info("loaded model: %s (%d tensors)", id, len(meta.Tensors))
	}
	return nil
}

// List rescans the models directory and returns all discovered models.
// New GGUF files added since the last call are automatically picked up;
// removed files are dropped from the list.
func (m *Manager) List() []*ModelInfo {
	// Snapshot existing models under read lock (fast).
	m.mu.RLock()
	snapshot := make(map[string]*ModelInfo, len(m.models))
	for k, v := range m.models {
		snapshot[k] = v
	}
	m.mu.RUnlock()

	// All filesystem work happens outside any lock.
	scanArchDefinitions(filepath.Join(m.dir, "arch"))
	fresh := make(map[string]*ModelInfo)
	paths, _ := filepath.Glob(filepath.Join(m.dir, "*.gguf"))
	for _, path := range paths {
		id := strings.TrimSuffix(filepath.Base(path), ".gguf")
		if existing := snapshot[id]; existing != nil {
			fresh[id] = existing
			continue
		}
		meta, err := ParseGGUF(path)
		if err != nil {
			continue
		}
		if !supportedArchitectures[meta.Architecture] {
			continue
		}
		fresh[id] = &ModelInfo{ID: id, Path: path, Format: FormatGGUF, Metadata: meta, LoadedAt: time.Now().Unix()}
		log.Info("discovered model: %s (%d tensors)", id, len(meta.Tensors))
	}

	// Safetensors directory scanning (mirrors GGUF pattern above).
	dirEntries, _ := os.ReadDir(m.dir)
	for _, e := range dirEntries {
		if !strings.HasSuffix(e.Name(), util.ExtSafetensorsDir) {
			continue
		}
		stDir := filepath.Join(m.dir, e.Name())
		fi, err := os.Stat(stDir)
		if err != nil || !fi.IsDir() {
			continue
		}
		id := strings.TrimSuffix(e.Name(), util.ExtSafetensorsDir)
		if strings.Contains(id, "..") {
			continue
		}
		if existing := snapshot[id]; existing != nil {
			fresh[id] = existing
			continue
		}
		meta, err := ParseSafetensorsDir(stDir, m.archDir)
		if err != nil {
			continue
		}
		if !supportedArchitectures[meta.Architecture] {
			continue
		}
		fresh[id] = &ModelInfo{ID: id, Path: stDir, Format: FormatSafetensors, Metadata: meta, LoadedAt: time.Now().Unix()}
		log.Info("discovered model: %s (%d tensors)", id, len(meta.Tensors))
	}

	// Write lock only for the pointer swap.
	m.mu.Lock()
	m.models = fresh
	m.mu.Unlock()

	out := make([]*ModelInfo, 0, len(fresh))
	for _, info := range fresh {
		out = append(out, info)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// Get returns the model with the given ID. If the model is not in the
// in-memory list, falls back to checking the models directory for a
// newly added GGUF file.
func (m *Manager) Get(id string) *ModelInfo {
	m.mu.RLock()
	info := m.models[id]
	m.mu.RUnlock()
	if info != nil {
		return info
	}
	return m.tryLoadOne(id)
}

// tryLoadOne attempts to load a single model by ID from the models directory.
// Checks for .gguf first, then falls back to .st directory.
func (m *Manager) tryLoadOne(id string) *ModelInfo {
	// GGUF path
	ggufPath := filepath.Join(m.dir, id+".gguf")
	if fi, err := os.Stat(ggufPath); err == nil && !fi.IsDir() {
		meta, err := ParseGGUF(ggufPath)
		if err == nil {
			m.mu.Lock()
			defer m.mu.Unlock()
			if existing := m.models[id]; existing != nil {
				return existing
			}
			scanArchDefinitions(filepath.Join(m.dir, "arch"))
			if !supportedArchitectures[meta.Architecture] {
				return nil
			}
			info := &ModelInfo{ID: id, Path: ggufPath, Format: FormatGGUF, Metadata: meta, LoadedAt: time.Now().Unix()}
			m.models[id] = info
			log.Info("discovered model: %s (%d tensors)", id, len(meta.Tensors))
			return info
		}
	}

	// Safetensors directory fallback
	stDir := filepath.Join(m.dir, id+util.ExtSafetensorsDir)
	if fi, err := os.Stat(stDir); err == nil && fi.IsDir() {
		meta, err := ParseSafetensorsDir(stDir, m.archDir)
		if err == nil {
			m.mu.Lock()
			defer m.mu.Unlock()
			if existing := m.models[id]; existing != nil {
				return existing
			}
			scanArchDefinitions(filepath.Join(m.dir, "arch"))
			if !supportedArchitectures[meta.Architecture] {
				return nil
			}
			info := &ModelInfo{ID: id, Path: stDir, Format: FormatSafetensors, Metadata: meta, LoadedAt: time.Now().Unix()}
			m.models[id] = info
			log.Info("discovered model: %s (%d tensors)", id, len(meta.Tensors))
			return info
		}
	}

	return nil
}
