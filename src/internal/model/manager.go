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

// mmprojGGUFPrefix marks a GGUF that holds the vision/audio tower + projector
// for a paired multimodal decoder GGUF, by llama.cpp convention (see
// convert_hf_to_gguf.py --mmproj). Files with this prefix must NOT appear in
// the standalone model list — they're not directly loadable as conversation
// targets. They're consumed alongside the matching decoder GGUF when vision
// support is wired up. Until then, the discovery scan filters them out.
const mmprojGGUFPrefix = "mmproj-"

// isMMProjGGUF reports whether the given .gguf basename is an mmproj sidecar.
func isMMProjGGUF(base string) bool {
	return strings.HasPrefix(base, mmprojGGUFPrefix)
}

// modelDescriptorTokens lists the case-insensitive name fragments that
// mark the transition from a model's family-and-size identifier to its
// quantization / chat-tune / format suffix. Stripping at the leftmost
// match yields a "family.size" key that's stable across quantizations
// and instruction-tunes — useful for matching loosely-named mmproj
// sidecars whose vision tower is identical regardless of the decoder's
// quant level. Exact-token matches; `Q\d` and quant-format tags get
// special handling in stripModelDescriptors.
var modelDescriptorTokens = map[string]bool{
	"it":       true, // "instruct-tuned" Google shorthand (Gemma)
	"instruct": true,
	"chat":     true,
	"coder":    true,
}

// quantFormatTokens lists the well-known quant/precision tags that
// follow the same truncation rule as the descriptor tokens above.
// Case-insensitive prefix or whole-token match.
var quantFormatTokens = []string{
	"MXFP4",
	"BF16",
	"F16",
	"F32",
	"GGUF",
}

// stripModelDescriptors truncates a model name at the leftmost
// descriptor / quant / format token (case-insensitive), returning the
// family-and-size prefix. Tokens are hyphen-separated (HuggingFace
// convention). Examples:
//
//	"gemma-4-E4B-it-Q4_K_M"           → "gemma-4-E4B"
//	"Llama-3.2-3B-Instruct"           → "Llama-3.2-3B"
//	"qwen35-9b-opus46-mix-i1-Q4_K_M"  → "qwen35-9b-opus46-mix-i1"
//	"gemma-4-E4B"                     → "gemma-4-E4B" (no descriptor)
//
// Q<digit> tokens always match; quant-format tokens match
// case-insensitively as whole tokens or as a prefix joined to other
// quant suffixes by `_` (e.g. "Q4_K_M", "MXFP4_MOE").
func stripModelDescriptors(name string) string {
	tokens := strings.Split(name, "-")
	for i, tok := range tokens {
		if isDescriptorToken(tok) {
			if i == 0 {
				return name // never strip the entire name to empty
			}
			return strings.Join(tokens[:i], "-")
		}
	}
	return name
}

func isDescriptorToken(tok string) bool {
	if tok == "" {
		return false
	}
	lower := strings.ToLower(tok)
	if modelDescriptorTokens[lower] {
		return true
	}
	// Q<digit> pattern: Q4_K_M, Q5_K_S, Q8_0, etc.
	if (tok[0] == 'Q' || tok[0] == 'q') && len(tok) >= 2 && tok[1] >= '0' && tok[1] <= '9' {
		return true
	}
	// Whole-token or prefix match against quant-format tokens.
	upper := strings.ToUpper(tok)
	for _, qf := range quantFormatTokens {
		if upper == qf || strings.HasPrefix(upper, qf+"_") {
			return true
		}
	}
	return false
}

// mmprojMatchKey strips the standard mmproj-* / -mmproj / .gguf
// decorations from a sidecar filename to produce its "what model is
// this for" identifier. Examples:
//
//	"mmproj-gemma-4-E2B.gguf"     → "gemma-4-E2B"
//	"gemma-4-E4B-mmproj-f16.gguf" → "gemma-4-E4B-f16"
//	"qwen36-mmproj.gguf"          → "qwen36"
func mmprojMatchKey(filename string) string {
	base := filepath.Base(filename)
	// Strip .gguf extension if present.
	base = strings.TrimSuffix(base, ".gguf")
	// Remove "mmproj-" and "-mmproj" markers. The hyphenated forms collapse
	// surrounding hyphens; the bare "mmproj" form just disappears.
	for _, pat := range []string{"mmproj-", "-mmproj"} {
		base = strings.ReplaceAll(base, pat, "")
	}
	base = strings.ReplaceAll(base, "mmproj", "")
	return base
}

// findMmprojForGGUF returns the absolute path of the mmproj sidecar
// whose match-key starts with the descriptor-stripped form of the
// decoder model. Convention from llama.cpp's `convert_hf_to_gguf.py
// --mmproj`: sidecar names contain "mmproj" but vary in placement
// (prefix / suffix / quant-tagged), so we strip both the decoder name
// and the candidate names down to family-and-size identifiers before
// matching. Returns "" when no sidecar matches or when m.enableMmproj
// is false.
//
// First match wins. With multiple matching sidecars (e.g. f16 + Q4)
// for the same family, the first one returned by the directory glob
// is used — typically deterministic given filesystem ordering, but
// the user can disambiguate by deleting the one they don't want.
func (m *Manager) findMmprojForGGUF(ggufPath string) string {
	if !m.enableMmproj {
		return ""
	}
	dir := filepath.Dir(ggufPath)
	decoderID := strings.TrimSuffix(filepath.Base(ggufPath), ".gguf")
	stripped := stripModelDescriptors(decoderID)
	if stripped == "" {
		return ""
	}

	candidates, err := filepath.Glob(filepath.Join(dir, "*mmproj*.gguf"))
	if err != nil {
		return ""
	}
	// First pass: exact full-stem match (incl. any precision/quant suffix like
	// -f16/-f32). When both gemma-4-E4B-it-f16.gguf and -f32.gguf coexist with
	// matching-suffix mmproj sidecars, the stripped-prefix fallback below would
	// bind both decoders to whichever sidecar the glob returns first (it strips
	// the suffix at the leftmost descriptor, e.g. "it"). The exact match keeps
	// the precision paired: -f32 decoder -> -f32 mmproj.
	for _, c := range candidates {
		if mmprojMatchKey(c) == decoderID {
			log.Info("bound mmproj sidecar %s to decoder %s (exact)", filepath.Base(c), filepath.Base(ggufPath))
			return c
		}
	}
	for _, c := range candidates {
		key := mmprojMatchKey(c)
		// Match the decoder's stripped name as a prefix of the candidate's
		// match-key. Hyphen-boundary check prevents "gemma-4-E2" matching
		// "gemma-4-E2B-f16".
		if !strings.HasPrefix(key, stripped) {
			continue
		}
		if len(key) > len(stripped) && key[len(stripped)] != '-' {
			continue
		}
		log.Info("bound mmproj sidecar %s to decoder %s", filepath.Base(c), filepath.Base(ggufPath))
		return c
	}
	// Sidecar files exist in this directory but none matched the decoder's
	// family-and-size key — likely a misnamed pairing. Without this the model
	// would load text-only with no diagnostic. Skip the warning when there are
	// no candidates at all (the ordinary text-only case).
	if len(candidates) > 0 {
		log.Warn("no mmproj sidecar matched decoder %s (key %q); %d sidecar(s) present in %s but none paired — loading text-only", filepath.Base(ggufPath), stripped, len(candidates), dir)
	}
	return ""
}

type ModelInfo struct {
	ID       string // filename without .gguf / directory name without .st
	Path     string // absolute path to .gguf file or .st/ directory
	Format   ModelFormat
	Metadata *GGUFMetadata
	LoadedAt int64 // unix timestamp when this model was first discovered

	// MmprojPath is the absolute path to a paired `mmproj-<id>.gguf`
	// sidecar containing the vision/audio tower + projector tensors,
	// when one exists alongside a Format=FormatGGUF model. Empty for
	// safetensors models (their .st/ directory carries vision tensors
	// inline) and for text-only GGUFs. Populated at discovery time;
	// the engine loader picks it up transparently — apiserver sees one
	// model, the loader binds the sidecar's vision tower to it.
	MmprojPath string

	// MmprojEnabled is the user-facing --auto-mmproj gate snapshot taken
	// at discovery time. The GGUF path doesn't need to consult it
	// downstream (MmprojPath is already nulled out when the gate is off),
	// but the safetensors path does — vision tensors are inline in the
	// .st/ directory, so the loader must be told whether to wire them up.
	MmprojEnabled bool
}

type Manager struct {
	mu           sync.RWMutex
	models       map[string]*ModelInfo
	dir          string
	archDir      string
	preferST     bool // prefer safetensors (.st/) over GGUF when both exist
	enableMmproj bool // scan for paired mmproj-*.gguf sidecars at discovery
}

// isArchSupported reports whether an architecture name is registered.
func isArchSupported(arch string) bool {
	archMu.RLock()
	defer archMu.RUnlock()
	return supportedArchitectures[arch]
}

func NewManager(dir string, preferST, enableMmproj bool) (*Manager, error) {
	archDir := filepath.Join(dir, "arch")
	scanArchDefinitions(archDir)

	m := &Manager{
		models:       make(map[string]*ModelInfo),
		dir:          dir,
		archDir:      archDir,
		preferST:     preferST,
		enableMmproj: enableMmproj,
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
		if isMMProjGGUF(base) {
			log.Debug("skipping %s: mmproj sidecar, not a standalone model", base)
			continue
		}
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
		m.models[id] = &ModelInfo{ID: id, Path: path, Format: FormatGGUF, Metadata: meta, LoadedAt: time.Now().Unix(), MmprojPath: m.findMmprojForGGUF(path), MmprojEnabled: m.enableMmproj}
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
		m.models[id] = &ModelInfo{ID: id, Path: stDir, Format: FormatSafetensors, Metadata: meta, LoadedAt: time.Now().Unix(), MmprojEnabled: m.enableMmproj}
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
		base := filepath.Base(path)
		if isMMProjGGUF(base) {
			continue
		}
		id := strings.TrimSuffix(base, ".gguf")
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
		fresh[id] = &ModelInfo{ID: id, Path: path, Format: FormatGGUF, Metadata: meta, LoadedAt: time.Now().Unix(), MmprojPath: m.findMmprojForGGUF(path), MmprojEnabled: m.enableMmproj}
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
		fresh[id] = &ModelInfo{ID: id, Path: stDir, Format: FormatSafetensors, Metadata: meta, LoadedAt: time.Now().Unix(), MmprojEnabled: m.enableMmproj}
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
	info := &ModelInfo{ID: id, Path: ggufPath, Format: FormatGGUF, Metadata: meta, LoadedAt: time.Now().Unix(), MmprojPath: m.findMmprojForGGUF(ggufPath), MmprojEnabled: m.enableMmproj}
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
	info := &ModelInfo{ID: id, Path: stDir, Format: FormatSafetensors, Metadata: meta, LoadedAt: time.Now().Unix(), MmprojEnabled: m.enableMmproj}
	m.models[id] = info
	log.Info("discovered model: %s (%d tensors)", id, len(meta.Tensors))
	return info
}
