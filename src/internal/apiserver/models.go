package apiserver

import (
	"net/http"

	"inference-lab-bench/internal/util"
)

type modelObject struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type listModelsResponse struct {
	Object string        `json:"object"`
	Data   []modelObject `json:"data"`
}

func (s *Server) handleListModels(w http.ResponseWriter, r *http.Request) {
	models := s.manager.List()
	data := make([]modelObject, 0, len(models))
	for _, m := range models {
		data = append(data, modelObject{
			ID:      m.ID,
			Object:  "model",
			Created: m.LoadedAt,
			OwnedBy: "local",
		})
	}
	util.WriteJSON(w, listModelsResponse{Object: "list", Data: data})
}
