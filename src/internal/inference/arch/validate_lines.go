package arch

import (
	"bufio"
	"bytes"
	"fmt"
	"strings"
)

// ResolveErrorLines maps validation errors to TOML source line numbers.
func ResolveErrorLines(tomlSource []byte, errs []ValidationError) []string {
	index := buildLineIndex(tomlSource)

	out := make([]string, len(errs))
	for i, ve := range errs {
		if line, ok := index[ve.KeyPath]; ok {
			out[i] = fmt.Sprintf("  line %d: %s: %s", line, ve.KeyPath, ve.Message)
		} else {
			out[i] = fmt.Sprintf("  %s: %s", ve.KeyPath, ve.Message)
		}
	}
	return out
}

// buildLineIndex scans TOML source and builds a map from dotted key paths to line numbers.
// Tracks [section] headers to build the full path for each key = value line.
func buildLineIndex(source []byte) map[string]int {
	index := make(map[string]int)
	scanner := bufio.NewScanner(bytes.NewReader(source))

	var sectionPath string
	lineNum := 0

	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())

		// Skip comments and blank lines
		if line == "" || line[0] == '#' {
			continue
		}

		// Section header: [section] or [section.subsection]
		// NOTE: [[array]] headers are not used in the arch TOML DSL. If added,
		// strip the extra '[' here and add per-element counter logic.
		if line[0] == '[' {
			end := strings.IndexByte(line, ']')
			if end > 0 {
				sectionPath = strings.TrimSpace(line[1:end])
				// Also index the section itself
				index[sectionPath] = lineNum
			}
			continue
		}

		// Key = value
		eq := strings.IndexByte(line, '=')
		if eq > 0 {
			key := strings.TrimSpace(line[:eq])
			fullPath := key
			if sectionPath != "" {
				fullPath = sectionPath + "." + key
			}
			index[fullPath] = lineNum
		}
	}

	return index
}
