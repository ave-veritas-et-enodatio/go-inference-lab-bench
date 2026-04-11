package apiserver

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	ggml "inference-lab-bench/internal/ggml"
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
		var allStats []ggml.MemoryStats

		for _, eng := range s.engines {
			if ws := eng.WeightStore(); ws != nil && ws.Buffer != nil {
				totalVRAM := ggml.DevMemory(ws.GPU)
				totalRAM := ggml.DevMemory(ws.CPU)

				bufSize := ws.Buffer.Size()

				allStats = append(allStats, ggml.MemoryStats{
					VRAM: ggml.MemoryStat{ Allocated: int64(bufSize), Total: totalVRAM },
					RAM: ggml.MemoryStat{ Allocated: int64(0), Total: totalRAM },
					IsUMA: ggml.IsUMA(ws.GPU),
				})
			}
		}

		// Aggregate stats
		var totalAllocatedRAM int64
		var totalAllocatedVRAM int64
		var totalVRAM int64
		var totalRAM int64
		var isUMA bool

		for _, stat := range allStats {
			totalAllocatedVRAM += stat.VRAM.Allocated
			if totalVRAM == 0 {
				totalVRAM += stat.VRAM.Total
			}
			totalAllocatedRAM += stat.RAM.Allocated
			if totalRAM == 0 {
				totalRAM += stat.RAM.Total
			}
			isUMA = isUMA || stat.IsUMA
		}

		vramUsage := 0.0
		if totalVRAM > 0 {
			vramUsage = float64(totalAllocatedVRAM) / float64(totalVRAM) * 100
		}
		ramUsage := 0.0
		if totalRAM > 0 {
			ramUsage = float64(totalAllocatedRAM) / float64(totalRAM) * 100
		}

		w.Header().Set("Content-Type", "application/json")
		resp := map[string]interface{}{
			"memstats":    allStats,
			"vram": map[string]interface{}{
				"alloc": totalAllocatedVRAM,
				"total":  totalVRAM,
				"usage": vramUsage,
			},
			"ram": map[string]interface{}{
				"alloc": totalAllocatedRAM,
				"total":  totalRAM,
				"usage": ramUsage,
			},
			"is_uma":      isUMA,
		}

		data, _ := json.Marshal(resp)
		w.Write(data)

		log.Info("[ctl] memstats: allocatedVRAM=%d VRAM=%d vramUsage=%.2f%% allocatedRAM=%d RAM=%d ramUsage=%.2f%% is_uma=%v",
			totalAllocatedVRAM, totalVRAM, vramUsage, totalAllocatedRAM, totalRAM, ramUsage, isUMA)

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
