// Package archeditor serves a local web app for editing TOML architecture definitions.
package archeditor

import (
	"fmt"
	"log"
	"mime"
	"net/http"
	"os"
	"path/filepath"
)

// Run starts the editor HTTP server on localhost at the given port.
// archDir is the path to the models/arch directory.
// staticDir is the path to the directory containing index.html, editor.js, etc.
func Run(archDir, staticDir string, port int) error {
	// Ensure correct MIME types — Go's default detection can return text/plain
	// for .js on macOS, which browsers reject with nosniff enforcement.
	mime.AddExtensionType(".js", "application/javascript; charset=utf-8")
	mime.AddExtensionType(".css", "text/css; charset=utf-8")

	api := &apiServer{archDir: archDir}

	mux := http.NewServeMux()

	// Static editor files
	mux.Handle("GET /static/", http.StripPrefix("/static/", http.FileServer(http.Dir(staticDir))))
	mux.HandleFunc("GET /", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		data, err := os.ReadFile(filepath.Join(staticDir, "index.html"))
		if err != nil {
			http.Error(w, "index.html not found", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(data)
	})

	// Arch directory: /arch/*.toml and /arch/block_svg/*.svg
	// Block /arch/editor/ to prevent the editor's own files from being served here.
	mux.HandleFunc("GET /arch/editor/", func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	})
	mux.Handle("GET /arch/", http.StripPrefix("/arch/", http.FileServer(http.Dir(archDir))))

	// API
	mux.HandleFunc("GET /api/arch", api.handleListArchs)
	mux.HandleFunc("POST /api/parse", api.handleParse)
	mux.HandleFunc("POST /api/serialize", api.handleSerialize)
	mux.HandleFunc("POST /api/validate", api.handleValidate)

	addr := fmt.Sprintf("127.0.0.1:%d", port)
	log.Printf("arch-editor listening on http://%s", addr)
	return http.ListenAndServe(addr, mux)
}
