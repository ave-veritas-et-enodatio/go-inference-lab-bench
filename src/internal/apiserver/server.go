package apiserver

import (
	"errors"
	"fmt"
	"net/http"
	"path/filepath"
	"sync"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"

	"inference-lab-bench/internal/inference"
	log "inference-lab-bench/internal/log"
	"inference-lab-bench/internal/model"
	"inference-lab-bench/internal/util"
)

const (
	apiRoot             = "/api/v1/"
	diagRoot            = "/diag/"
	ctlRoot             = "/ctl/"
	modelsEndpoint      = apiRoot + "models"
	completionsEndpoint = apiRoot + "chat/completions"
)

type Server struct {
	cfg        *Config
	manager    *model.Manager
	engines    map[string]*inference.Engine // model ID → loaded engine
	archDir    string                       // absolute path to arch definitions
	httpServer *http.Server                 // for graceful shutdown
	pending    sync.WaitGroup               // tracks in-flight inference requests
}

func NewServer(cfg *Config, manager *model.Manager) *Server {
	absModelsDir := cfg.Models.Directory
	if !filepath.IsAbs(absModelsDir) {
		if abs, err := filepath.Abs(absModelsDir); err == nil {
			absModelsDir = abs
		}
	}
	return &Server{
		cfg:     cfg,
		manager: manager,
		engines: make(map[string]*inference.Engine),
		archDir: filepath.Join(absModelsDir, "arch"),
	}
}

// evictEngine closes and removes the engine for modelID. Safe to call if the
// engine is not loaded. Used to recover from a poisoned Metal backend.
func (s *Server) evictEngine(modelID string) {
	if eng, ok := s.engines[modelID]; ok {
		log.Warn("evicting poisoned engine for %s", modelID)
		eng.Close()
		delete(s.engines, modelID)
	}
}

// Engine returns or lazily creates an inference engine for the given model ID.
// Only one engine is kept resident at a time — switching models evicts the previous one.
func (s *Server) Engine(modelID string) (*inference.Engine, error) {
	if eng, ok := s.engines[modelID]; ok {
		return eng, nil
	}
	info := s.manager.Get(modelID)
	if info == nil {
		log.Error("model not found: %s", modelID)
		return nil, fmt.Errorf("model not found: %s", modelID)
	}
	// Evict previous engines unless configured to keep all loaded.
	if s.cfg.Inference.SingleResidentModel == nil || *s.cfg.Inference.SingleResidentModel {
		for id, eng := range s.engines {
			log.Info("evicting engine for %s", id)
			eng.Close()
			delete(s.engines, id)
		}
	}
	log.Info("loading inference engine for %s ...", modelID)
	eng, err := inference.NewEngine(info, s.archDir, s.cfg.Inference.MaxSeqLen, s.cfg.Inference.UseFlashAttention())
	if err != nil {
		return nil, fmt.Errorf("inference engine: %w", err)
	}
	s.engines[modelID] = eng
	log.Info("inference engine ready for %s", modelID)
	return eng, nil
}

func (s *Server) Run() error {
	r := chi.NewRouter()
	r.Use(middleware.Recoverer)
	r.Use(s.authMiddleware)

	r.Get(modelsEndpoint, s.handleListModels)
	r.Post(completionsEndpoint, s.handleChatCompletions)

	r.Get(ctlRoot, s.handleCtl)

	// Diagnostic file explorer: serves contents of bin/diag/
	paths := util.ResolvePaths()
	diagFS := http.StripPrefix("/diag/", http.FileServer(http.Dir(paths.DiagDir)))
	r.Get(diagRoot+"*", diagFS.ServeHTTP)
	r.Get(diagRoot, diagFS.ServeHTTP)

	addr := fmt.Sprintf("%s:%d", s.cfg.Server.Host, s.cfg.Server.Port)
	protoAddr := "http://" + addr
	s.httpServer = &http.Server{Addr: addr, Handler: r}
	log.Info("api: %s%s", protoAddr, apiRoot)
	log.Info("endpoints:\n    models: %s%s\n    completions: %s%s", protoAddr, modelsEndpoint, protoAddr, completionsEndpoint)
	log.Info("diagnostics: %s%s", protoAddr, diagRoot)
	log.Info("control: %s%s", protoAddr, ctlRoot)
	if err := s.httpServer.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

// resolveDefaultModel returns the model ID based on the "default" config field.
// "first" → first model in list, "last" → last, otherwise treated as explicit name.
// If an explicit name is configured but the model file is not found, falls back to
// first model and logs a warning.
func (s *Server) resolveDefaultModel() string {
	dflt := s.cfg.Models.Default
	models := s.manager.List()
	if len(models) == 0 {
		return ""
	}
	switch dflt {
	case "", "first":
		return models[0].ID
	case "last":
		return models[len(models)-1].ID
	default:
		// Check if the configured default model actually exists
		info := s.manager.Get(dflt)
		if info == nil {
			log.Warn("configured default model %q not found; falling back to first available model", dflt)
			return models[0].ID
		}
		return dflt
	}
}

func (s *Server) authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		token := s.cfg.Server.AuthToken
		if token == "" {
			next.ServeHTTP(w, r)
			return
		}
		auth := r.Header.Get("Authorization")
		expected := "Bearer " + token
		if auth != expected {
			writeError(w, http.StatusUnauthorized, "invalid auth token")
			return
		}
		next.ServeHTTP(w, r)
	})
}
