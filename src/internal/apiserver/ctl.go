package apiserver

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	log "inference-lab-bench/internal/log"
)

func (s *Server) handleCtl(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()

	// help command or no command -> list available commands
	if q.Has("help") || !q.Has("quit") && !q.Has("memstats") {
		w.Header().Set("Content-Type", "application/json")

		commands := []map[string]interface{}{
			{
				"quit": map[string]interface{}{
					"desc": "Gracefully shut down the server after pending inference completes",
					"options": map[string]interface{}{
						"now": "Shutdown immediately (100ms timeout) instead of waiting",
					},
				},
			},
			{
				"memstats": map[string]interface{}{
					"desc": "Print current VRAM/RAM usage from loaded models",
				},
			},
			{
				"help": map[string]interface{}{
					"desc": "Show help about available commands",
				},
			},
		}

		data, _ := json.Marshal(commands)
		w.Write(data)
		return
	}

	// memstats command
	if q.Has("memstats") {
		// Collect memory stats from loaded engines
		totalStats := s.MemoryStats()

		w.Header().Set("Content-Type", "application/json")
		resp := map[string]interface{}{
			"vram": map[string]interface{}{
				"alloc": totalStats.VRAM.Allocated,
				"total":  totalStats.VRAM.Total,
				"usage": totalStats.VRAM.AllocatedPct(),
			},
			"ram": map[string]interface{}{
				"alloc": totalStats.RAM.Allocated,
				"total":  totalStats.RAM.Total,
				"usage": totalStats.RAM.AllocatedPct(),
			},
			"is_uma":      totalStats.IsUMA,
		}

		data, _ := json.Marshal(resp)
		w.Write(data)

		log.Info("[ctl] memstats: allocatedVRAM=%d VRAM=%d vramUsage=%.2f%% allocatedRAM=%d RAM=%d ramUsage=%.2f%% is_uma=%v",
			totalStats.VRAM.Allocated, totalStats.VRAM.Total, totalStats.VRAM.AllocatedPct(),
			totalStats.RAM.Allocated, totalStats.RAM.Total, totalStats.RAM.AllocatedPct(),
			totalStats.IsUMA,
		)
		return
	}

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
