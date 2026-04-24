package util

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"

	"github.com/BurntSushi/toml"

	log "inference-lab-bench/internal/log"
)

// WriteJSON writes a JSON response with the appropriate Content-Type header.
func WriteJSON(w http.ResponseWriter, v any) error {
	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(v)
}

// LoadTOML reads a TOML file and decodes it into the provided value.
func LoadTOML(path string, v any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("reading %s: %w", path, err)
	}
	if _, err := toml.Decode(string(data), v); err != nil {
		return fmt.Errorf("parsing %s: %w", path, err)
	}
	return nil
}

func WriteTOML(path string, v any) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("creating %s: %w", path, err)
	}
	defer f.Close()
	return toml.NewEncoder(f).Encode(v)
}

func CopyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		log.Warn("copyFile open %s: %v", src, err)
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		log.Warn("copyFile create %s: %v", dst, err)
		return err
	}
	defer out.Close()
	if _, err := io.Copy(out, in); err != nil {
		log.Warn("copyFile %s → %s: %v", src, dst, err)
		return err
	}
	return nil
}
