package model

// CullingMap holds the pre-computed culling map for a model.
// Granularity: one entry per (layer, head) pair.
// A value of 1.0 means "always active"; values < threshold get culled.
//
// The binary sidecar file format (*.culling) is:
//   [4 bytes] magic: 0x43554C4C ("CULL")
//   [4 bytes] version: 1 (uint32 LE)
//   [4 bytes] n_layers (uint32 LE)
//   [4 bytes] n_heads  (uint32 LE)
//   [n_layers * n_heads * 4 bytes] float32 LE culling values, row-major [layer][head]
type CullingMap struct {
	NLayers int
	NHeads  int
	Values  []float32 // [layer * n_heads + head]
}

// LiveRegionMap returns a flat bool slice [layer * n_heads + head] indicating
// which (layer, head) pairs should be computed for the given prompt.
// If no culling map is loaded, all regions are live (all-ones behaviour).
func (m *CullingMap) LiveRegionMap(cullingVector []float32, threshold float32) []bool {
	total := m.NLayers * m.NHeads
	live := make([]bool, total)
	if len(m.Values) == 0 {
		// No culling data — all regions active
		for i := range live {
			live[i] = true
		}
		return live
	}
	for i, val := range m.Values {
		score := val
		if len(cullingVector) > i {
			score *= cullingVector[i]
		}
		live[i] = score >= threshold
	}
	return live
}

// LoadCullingMap loads a .culling sidecar file.
// Returns an all-ones map if the file does not exist.
func LoadCullingMap(path string, nLayers, nHeads int) (*CullingMap, error) {
	// Stub: return all-ones map until generation tool is built
	m := &CullingMap{
		NLayers: nLayers,
		NHeads:  nHeads,
	}
	// Values left nil → LiveRegionMap returns all-true
	return m, nil
}
