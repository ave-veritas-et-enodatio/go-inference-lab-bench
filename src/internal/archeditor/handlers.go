package archeditor

import (
	"encoding/json"
	"net/http"

	"inference-lab-bench/internal/inference/arch"
	"inference-lab-bench/internal/util"
)

type apiServer struct {
	archDir string
}

// decodeTOMLBody reads and decodes a JSON request with a "toml" field.
// Returns the TOML string and true on success, or writes an error response and returns false.
func decodeTOMLBody(w http.ResponseWriter, r *http.Request) (string, bool) {
	var body struct {
		TOML string `json:"toml"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return "", false
	}
	return body.TOML, true
}

func (s *apiServer) handleListArchs(w http.ResponseWriter, _ *http.Request) {
	names, err := arch.ListArchitectures(s.archDir)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	util.WriteJSON(w, names)
}

func (s *apiServer) handleParse(w http.ResponseWriter, r *http.Request) {
	tomlStr, ok := decodeTOMLBody(w, r)
	if !ok {
		return
	}
	def, err := arch.Parse([]byte(tomlStr))
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnprocessableEntity)
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}
	util.WriteJSON(w, def)
}

func (s *apiServer) handleSerialize(w http.ResponseWriter, r *http.Request) {
	def := &arch.ArchDef{}
	if err := json.NewDecoder(r.Body).Decode(def); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	// Ensure maps are initialized (JSON decode may leave them nil)
	if def.Params.Keys == nil {
		def.Params.Keys = map[string]string{}
	}
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.Write(SerializeToTOML(def))
}

func (s *apiServer) handleValidate(w http.ResponseWriter, r *http.Request) {
	tomlStr, ok := decodeTOMLBody(w, r)
	if !ok {
		return
	}
	_, err := arch.Parse([]byte(tomlStr))
	type response struct {
		Valid  bool     `json:"valid"`
		Errors []string `json:"errors,omitempty"`
	}
	resp := response{Valid: err == nil}
	if err != nil {
		resp.Errors = []string{err.Error()}
	}
	util.WriteJSON(w, resp)
}
