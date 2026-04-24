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
	mu       sync.RWMutex
	models   map[string]*ModelInfo
	dir      string
	archDir  string
	preferST bool // prefer safetensors (.st/) over GGUF when both exist
}

// isArchSupported reports whether an architecture name is registered.
func isArchSupported(arch string) bool {
	archMu.RLock()
	defer archMu.RUnlock()
	return supportedArchitectures[arch]
}

func NewManager(dir string, preferST bool) (*Manager, error) {
	archDir := filepath.Join(dir, "arch")
	scanArchDefinitions(archDir)

	m := &Manager{
		models:   make(map[string]*ModelInfo),
		dir:      dir,
		archDir:  archDir,
		preferST: preferST,
	}
	if err := m.scan(); err != nil {
		return nil, err
	}
	return m, nil
}

// scanArchDefinitions reads *.arch.toml filenames in archDir and populates
// supportedArchitectures. The architecture name is the filename stem
// (e.g. "llada-moe.arch.toml" → "llada-moe"). Callers may be concurrent
// (List() is called from HTTP handlers); archMu serializes map writes.
func scanArchDefinitions(archDir string) {
	// Glob outside the lock — filesystem I/O, not shared state.
	paths, _ := filepath.Glob(filepath.Join(archDir, "*"+util.ExtArchToml))
	archMu.Lock()
	defer archMu.Unlock()
	for _, path := range paths {
		name := strings.TrimSuffix(filepath.Base(path), util.ExtArchToml)
		if name != "" {
			supportedArchitectures[name] = true
		}
	}
}

func (m *Manager) scan() error {
	if m.preferST {
		if err := m.scanSafetensors(); err != nil {
			return err
		}
		m.scanGGUF()
	} else {
		m.scanGGUF()
		if err := m.scanSafetensors(); err != nil {
			return err
		}
	}
	return nil
}

func (m *Manager) scanGGUF() {
	paths, err := filepath.Glob(filepath.Join(m.dir, "*.gguf"))
	if err != nil {
		return
	}
	for _, path := range paths {
		base := filepath.Base(path)
		id := strings.TrimSuffix(base, ".gguf")
		if _, exists := m.models[id]; exists {
			log.Debug("skipping %s: %s version already loaded", base, m.models[id].Format)
			continue
		}
		meta, err := ParseGGUF(path)
		if err != nil {
			log.Info("skipping %s: %v", base, err)
			continue
		}
		if !isArchSupported(meta.Architecture) {
			log.Warn("skipping %s: unsupported architecture %q", id, meta.Architecture)
			continue
		}
		m.models[id] = &ModelInfo{ID: id, Path: path, Format: FormatGGUF, Metadata: meta, LoadedAt: time.Now().Unix()}
		log.Info("loaded model: %s (%d tensors)", id, len(meta.Tensors))
	}
}

func (m *Manager) scanSafetensors() error {
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
		if existing, exists := m.models[id]; exists {
			log.Debug("skipping %s: %s version already loaded", e.Name(), existing.Format)
			continue
		}
		meta, err := ParseSafetensorsDir(stDir, m.archDir)
		if err != nil {
			log.Info("skipping %s: %v", e.Name(), err)
			continue
		}
		log.Debug("safetensors parsed: id=%q arch=%q tensors=%d", id, meta.Architecture, len(meta.Tensors))
		if !isArchSupported(meta.Architecture) {
			log.Warn("skipping %s: unsupported architecture %q", id, meta.Architecture)
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
	if m.preferST {
		m.listSafetensors(snapshot, fresh)
		m.listGGUF(snapshot, fresh)
	} else {
		m.listGGUF(snapshot, fresh)
		m.listSafetensors(snapshot, fresh)
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

func (m *Manager) listGGUF(snapshot, fresh map[string]*ModelInfo) {
	paths, _ := filepath.Glob(filepath.Join(m.dir, "*.gguf"))
	for _, path := range paths {
		id := strings.TrimSuffix(filepath.Base(path), ".gguf")
		if existing := snapshot[id]; existing != nil && existing.Format == FormatGGUF {
			fresh[id] = existing
			continue
		}
		if _, exists := fresh[id]; exists {
			continue
		}
		meta, err := ParseGGUF(path)
		if err != nil {
			continue
		}
		if !isArchSupported(meta.Architecture) {
			continue
		}
		fresh[id] = &ModelInfo{ID: id, Path: path, Format: FormatGGUF, Metadata: meta, LoadedAt: time.Now().Unix()}
		log.Info("discovered model: %s (%d tensors)", id, len(meta.Tensors))
	}
}

func (m *Manager) listSafetensors(snapshot, fresh map[string]*ModelInfo) {
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
		if existing := snapshot[id]; existing != nil && existing.Format == FormatSafetensors {
			fresh[id] = existing
			continue
		}
		if _, exists := fresh[id]; exists {
			continue
		}
		meta, err := ParseSafetensorsDir(stDir, m.archDir)
		if err != nil {
			continue
		}
		if !isArchSupported(meta.Architecture) {
			continue
		}
		fresh[id] = &ModelInfo{ID: id, Path: stDir, Format: FormatSafetensors, Metadata: meta, LoadedAt: time.Now().Unix()}
		log.Info("discovered model: %s (%d tensors)", id, len(meta.Tensors))
	}
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
// Checks the preferred format first, then falls back to the other.
func (m *Manager) tryLoadOne(id string) *ModelInfo {
	if m.preferST {
		if info := m.tryLoadOneST(id); info != nil {
			return info
		}
		return m.tryLoadOneGGUF(id)
	}
	if info := m.tryLoadOneGGUF(id); info != nil {
		return info
	}
	return m.tryLoadOneST(id)
}

func (m *Manager) tryLoadOneGGUF(id string) *ModelInfo {
	ggufPath := filepath.Join(m.dir, id+".gguf")
	fi, err := os.Stat(ggufPath)
	if err != nil || fi.IsDir() {
		return nil
	}
	meta, err := ParseGGUF(ggufPath)
	if err != nil {
		return nil
	}
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

func (m *Manager) tryLoadOneST(id string) *ModelInfo {
	stDir := filepath.Join(m.dir, id+util.ExtSafetensorsDir)
	fi, err := os.Stat(stDir)
	if err != nil || !fi.IsDir() {
		return nil
	}
	meta, err := ParseSafetensorsDir(stDir, m.archDir)
	if err != nil {
		return nil
	}
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
