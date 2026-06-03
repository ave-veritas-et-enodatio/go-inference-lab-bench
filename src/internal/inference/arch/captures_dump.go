package arch

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
)

// DumpNamedTensors writes each capture in caps.NamedTensors to one file
// per capture under outDir, plus a manifest.json listing names, lengths,
// and per-tensor statistics (min, max, mean, finite count) for quick
// at-a-glance comparison without re-reading the binary blobs.
//
// File format per tensor: raw little-endian float32 array, name-derived
// filename (path-safe). Matches the shape llama-mtmd produces when its
// `cb()` callbacks are wired to a similar dump, so a side-by-side diff
// against the same image becomes `cmp` + a small numpy script.
//
// Used by the Phase 8 numerical-equivalence workflow. No-op when caps
// or caps.NamedTensors is empty.
func DumpNamedTensors(caps *ForwardCaptures, outDir string) error {
	if caps == nil || len(caps.NamedTensors) == 0 {
		return nil
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return fmt.Errorf("mkdir %s: %w", outDir, err)
	}

	// Sort names so manifest order is stable run-to-run.
	names := make([]string, 0, len(caps.NamedTensors))
	for k := range caps.NamedTensors {
		names = append(names, k)
	}
	sort.Strings(names)

	manifest := make([]captureManifestEntry, 0, len(names))
	for _, name := range names {
		data := caps.NamedTensors[name]
		fname := safeName(name) + ".f32"
		path := filepath.Join(outDir, fname)
		if err := writeFloat32LE(path, data); err != nil {
			return fmt.Errorf("write %s: %w", name, err)
		}
		manifest = append(manifest, captureStats(name, fname, data))
	}

	manifestPath := filepath.Join(outDir, "manifest.json")
	mf, err := os.Create(manifestPath)
	if err != nil {
		return fmt.Errorf("create manifest: %w", err)
	}
	defer mf.Close()
	enc := json.NewEncoder(mf)
	enc.SetIndent("", "  ")
	return enc.Encode(manifest)
}

type captureManifestEntry struct {
	Name      string  `json:"name"`
	File      string  `json:"file"`
	NElements int     `json:"n_elements"`
	Min       float64 `json:"min"`
	Max       float64 `json:"max"`
	Mean      float64 `json:"mean"`
	NFinite   int     `json:"n_finite"`
	NNaN      int     `json:"n_nan"`
	NInf      int     `json:"n_inf"`
}

func captureStats(name, file string, data []float32) captureManifestEntry {
	e := captureManifestEntry{Name: name, File: file, NElements: len(data)}
	if len(data) == 0 {
		return e
	}
	e.Min = math.Inf(1)
	e.Max = math.Inf(-1)
	var sum float64
	for _, v := range data {
		f := float64(v)
		switch {
		case math.IsNaN(f):
			e.NNaN++
		case math.IsInf(f, 0):
			e.NInf++
		default:
			e.NFinite++
			sum += f
			if f < e.Min {
				e.Min = f
			}
			if f > e.Max {
				e.Max = f
			}
		}
	}
	if e.NFinite > 0 {
		e.Mean = sum / float64(e.NFinite)
	} else {
		// Avoid carrying +Inf/-Inf into the manifest for the all-non-finite
		// edge case — replace with NaN sentinel.
		e.Min = math.NaN()
		e.Max = math.NaN()
	}
	return e
}

func writeFloat32LE(path string, data []float32) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	buf := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	_, err = f.Write(buf)
	return err
}

// safeName converts a capture name (e.g. "vision.layer_0", "decoder.inp_embd_post_splice")
// into a path-safe filename component. Replaces dots with underscores; rejects
// path-traversal characters defensively.
func safeName(name string) string {
	out := make([]byte, 0, len(name))
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z',
			r >= 'A' && r <= 'Z',
			r >= '0' && r <= '9',
			r == '_', r == '-':
			out = append(out, byte(r))
		case r == '.':
			out = append(out, '_')
		default:
			out = append(out, '_')
		}
	}
	if len(out) == 0 {
		return "unnamed"
	}
	return string(out)
}
