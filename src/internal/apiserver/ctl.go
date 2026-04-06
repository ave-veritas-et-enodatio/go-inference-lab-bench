package apiserver

import (
	"context"
	"net/http"
	"time"

	log "inference-lab-bench/internal/log"
)

func (s *Server) handleCtl(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	if !q.Has("quit") {
		writeError(w, http.StatusBadRequest, "unknown command")
		return
	}

	now := q.Has("now")
	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(`{"ok":true}`)) //nolint:errcheck

	go func() {
		if now {
			log.Info("[ctl] quit now — shutting down immediately")
			ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
			defer cancel()
			s.httpServer.Shutdown(ctx) //nolint:errcheck
		} else {
			log.Info("[ctl] quit — waiting for pending inference to finish")
			s.pending.Wait()
			log.Info("[ctl] all inference done — shutting down")
			s.httpServer.Shutdown(context.Background()) //nolint:errcheck
		}
	}()
}
