package inference

import (
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"time"

	ggufparser "github.com/gpustack/gguf-parser-go"

	"inference-lab-bench/internal/inference/arch"
	"inference-lab-bench/internal/inference/culling"
	log "inference-lab-bench/internal/log"
	"inference-lab-bench/internal/model"
	"inference-lab-bench/internal/util"
)

// ErrComputeFailed is returned by Generate when the ggml Metal backend fails to
// execute a graph. The underlying Metal context is permanently poisoned after
// this error — the server must evict and re-create the engine to recover.
var ErrComputeFailed = arch.ErrComputeFailed

// IsComputeFailure reports whether err is (or wraps) a Metal compute failure.
func IsComputeFailure(err error) bool {
	return errors.Is(err, ErrComputeFailed)
}

// Engine runs inference for a single loaded model.
type Engine struct {
	model       *arch.GenericModel
	tokenizer   *Tokenizer
	nLayers     int
	maxSeqLen   int
	cullingMeta *culling.CullingMeta // optional sidecar metadata (nil = none)
	flashAttn   bool                 // server default: use FA2 when head geometry allows
}

// NewEngine creates an inference engine for the given model.
// archDir is the directory containing architecture definition TOML files.
// maxSeqLen is the KV cache size in tokens (0 = default 8192).
// flashAttention is the server default for FA2 (true = enable when head geometry allows).
// Automatically loads a .modulemap sidecar file if present next to the GGUF.
func NewEngine(info *model.ModelInfo, archDir string, maxSeqLen int, flashAttention bool) (*Engine, error) {
	f, err := ggufparser.ParseGGUFFile(info.Path)
	if err != nil {
		return nil, fmt.Errorf("parsing GGUF for tokenizer: %w", err)
	}

	tok, err := NewTokenizerFromGGUF(f, info.Path)
	if err != nil {
		return nil, fmt.Errorf("building tokenizer: %w", err)
	}

	// DSL-driven model loading — reuse the already-parsed GGUF to avoid double-parsing.
	m, err := arch.NewGenericModel(info.Metadata.Architecture, info.Path, archDir, f)
	if err != nil {
		return nil, fmt.Errorf("creating model: %w", err)
	}

	if maxSeqLen <= 0 {
		maxSeqLen = defaultMaxSeqLen
	}

	validateArchTokens(m.Def.Tokens, tok)

	// Load culling metadata sidecar if present: <model_stem>.<method>.cullmeta
	var cullingMeta *culling.CullingMeta
	dir := filepath.Dir(info.Path)
	stem := strings.TrimSuffix(filepath.Base(info.Path), filepath.Ext(info.Path))
	metaPaths, _ := filepath.Glob(filepath.Join(dir, stem+".*"+util.ExtCullMeta))
	if len(metaPaths) > 0 {
		cm, err := culling.LoadCullingMeta(metaPaths[0])
		if err != nil {
			log.Warn("failed to load culling metadata: %v", err)
		} else if cm != nil {
			cullingMeta = cm
			log.Info("loaded culling metadata: %s (method=%s)", metaPaths[0], cm.Method)
		}
	}

	// Resolve flash attention: check head geometry against Metal FA2 supported dims.
	headDim := m.HeadDim
	flashAttn := flashAttention && arch.FlashAttnSupported(headDim)
	switch {
	case !flashAttention:
		log.Info("flash attention: disabled by config")
	case !arch.FlashAttnSupported(headDim):
		log.Warn("flash attention: disabled (fallback) - head_dim=%d not in FA2 supported set", headDim)
	default:
		log.Info("flash attention: enabled (head_dim=%d)", headDim)
	}

	return &Engine{
		model:       m,
		tokenizer:   tok,
		nLayers:     info.Metadata.NLayers,
		maxSeqLen:   maxSeqLen,
		cullingMeta: cullingMeta,
		flashAttn:   flashAttn,
	}, nil
}

// validateArchTokens checks each non-empty token string from the arch TOML
// against the loaded vocabulary. Warns if a string is not a direct vocab entry
// or encodes to zero tokens.
func validateArchTokens(tokens arch.TokensDef, tok *Tokenizer) {
	check := func(name, value string) {
		if value == "" {
			return
		}
		ids := tok.Encode(value, false)
		if len(ids) == 0 {
			log.Warn("arch token %s=%q encodes to zero tokens — not in vocabulary", name, value)
		} else if !tok.VocabContains(value) {
			log.Warn("arch token %s=%q is not a single vocabulary entry (encodes to %d BPE sub-tokens)", name, value, len(ids))
		}
	}
	check("think_open", tokens.ThinkOpen)
	check("think_close", tokens.ThinkClose)
	for i, s := range tokens.ExtraEOS {
		check(fmt.Sprintf("extra_eos[%d]", i), s)
	}
}

// ThinkOpen returns the TOML-defined think opening tag, or "".
func (e *Engine) ThinkOpen() string { return e.model.Def.Tokens.ThinkOpen }

// ThinkClose returns the TOML-defined think closing tag, or "".
func (e *Engine) ThinkClose() string { return e.model.Def.Tokens.ThinkClose }

// Close frees all GPU resources.
func (e *Engine) Close() {
	if e.model != nil {
		e.model.Close()
	}
}

// WeightStore returns the model's weight store (for memory stats).
// Returns nil if the engine is not loaded.
func (e *Engine) WeightStore() *arch.WeightStore {
	if e.model == nil {
		return nil
	}
	return e.model.Store
}

// GenerateParams controls the generation loop.
type GenerateParams struct {
	MaxTokens       int
	Temperature     float32
	TopP            float32
	Stateless       bool   // bypass KV cache (for testing/comparison)
	CullMethod      string // "" or "none" = no culling; "random" = random test pattern
	ThinkingEnabled bool   // if false, strip think_open prefix the GGUF template may append
	LogProbs        bool   // include log-probabilities in response
	TopLogProbs     int    // number of top log-probabilities per token (0 = just the chosen token)
	FlashAttention  *bool  // nil = use server default; true/false = per-request override
}

// DefaultParams returns sensible defaults.
func DefaultParams() GenerateParams {
	return GenerateParams{
		MaxTokens:       512,
		Temperature:     0.7,
		TopP:            0.9,
		ThinkingEnabled: true,
	}
}

const defaultMaxSeqLen = 8192

// Generate runs the full chat generation loop.
// Uses KV/SSM cache by default; set params.Stateless for full-sequence recomputation.
// Returns metrics for the generation request.
func (e *Engine) Generate(
	messages []ChatMessage,
	params GenerateParams,
	onToken func(token string) bool,
) (*InferenceMetrics, error) {
	promptIDs, err := e.tokenizer.EncodeChat(messages, params.ThinkingEnabled)
	if err != nil {
		return nil, fmt.Errorf("encoding chat: %w", err)
	}
	if len(promptIDs) == 0 {
		return nil, fmt.Errorf("empty prompt after encoding")
	}
	// The template handles enable_thinking natively — when disabled, it emits an
	// empty think block (e.g. "<think>\n\n</think>\n\n") which is the correct
	// prompt. No post-render stripping needed; this matches llama.cpp behavior.

	// Resolve effective flash attention: per-request override applies config flag;
	// geometry check is enforced at load time (e.flashAttn already incorporates it).
	flashAttn := e.flashAttn
	if params.FlashAttention != nil {
		flashAttn = *params.FlashAttention && arch.FlashAttnSupported(e.model.HeadDim)
	}
	params.FlashAttention = &flashAttn
	log.Info("flash_attention_used=%v", *params.FlashAttention)

	// Per-query culling: clone canonical map, apply method, compile mask.
	var cullMap *arch.ModuleMap
	var mask *arch.CullingMask
	if params.CullMethod != "" && params.CullMethod != "none" {
		prompt := lastUserContent(messages)
		cullMap, mask = culling.ApplyCulling(
			e.model.CanonicalModuleMap, params.CullMethod, promptIDs, prompt, e.cullingMeta)
	}

	maxTokens := params.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 512
	}

	stopSet := e.tokenizer.BuildStopSet(e.model.Def.Tokens.ExtraEOS)

	metrics := &InferenceMetrics{
		PromptTokens: len(promptIDs),
	}
	if mask != nil {
		metrics.ZeroedTensors = mask.NumZeroed()
	}
	// Count total tensors: globals + layers
	metrics.TotalTensors = len(e.model.Weights.Global)
	for _, lw := range e.model.Weights.Layers {
		metrics.TotalTensors += len(lw.Common) + len(lw.Block) + len(lw.FFN)
	}

	start := time.Now()

	var genErr error
	if params.Stateless {
		log.Info("stateless inference (no KV cache), prompt=%d tokens", len(promptIDs))
		genErr = e.generateStateless(promptIDs, maxTokens, stopSet, params, mask, onToken, metrics)
	} else {
		genErr = e.generateCached(promptIDs, maxTokens, stopSet, params, mask, onToken, metrics)
	}

	metrics.TotalDuration = time.Since(start)

	// Write diagnostic files.
	if cullMap != nil {
		culling.WriteCullDiagnostics(cullMap, e.model.ModelPath, e.model.TensorDims, metrics.Engagement)
	}

	return metrics, genErr
}

func (e *Engine) generateCached(
	promptIDs []int32, maxTokens int, stopSet map[int32]bool,
	params GenerateParams, mask *arch.CullingMask,
	onToken func(string) bool, metrics *InferenceMetrics,
) error {
	cache, err := e.model.NewCache(e.maxSeqLen)
	if err != nil {
		return fmt.Errorf("creating cache: %w", err)
	}
	defer cache.Free()

	prefillStart := time.Now()
	logits, err := e.model.ForwardCached(cache, promptIDs, mask, *params.FlashAttention)
	if err != nil {
		return fmt.Errorf("prefill forward: %w", err)
	}
	metrics.PrefillDuration = time.Since(prefillStart)

	decodeStart := time.Now()
	hitStop := false
	for range maxTokens {
		nextID := e.sample(logits, params)
		if stopSet[nextID] {
			hitStop = true
			break
		}
		if params.LogProbs {
			metrics.TokenLogProbs = append(metrics.TokenLogProbs,
				ComputeTopLogProbs(logits, nextID, params.TopLogProbs, e.tokenizer.TokenString))
		}
		if !onToken(e.tokenizer.TokenString(nextID)) {
			hitStop = true
			break
		}
		metrics.CompletionTokens++
		logits, err = e.model.ForwardCached(cache, []int32{nextID}, mask, *params.FlashAttention)
		if err != nil {
			return fmt.Errorf("decode forward: %w", err)
		}
	}
	metrics.DecodeDuration = time.Since(decodeStart)
	if hitStop {
		metrics.FinishReason = "stop"
	} else {
		metrics.FinishReason = "length"
	}
	return nil
}

func (e *Engine) generateStateless(
	promptIDs []int32, maxTokens int, stopSet map[int32]bool,
	params GenerateParams, mask *arch.CullingMask,
	onToken func(string) bool, metrics *InferenceMetrics,
) error {
	tokenIDs := make([]int32, len(promptIDs))
	copy(tokenIDs, promptIDs)

	hitStop := false
	for range maxTokens {
		stepStart := time.Now()
		logits, engagement, err := e.model.ForwardStateless(tokenIDs, mask, nil, *params.FlashAttention)
		if err != nil {
			return fmt.Errorf("forward pass: %w", err)
		}
		if metrics.CompletionTokens == 0 {
			metrics.PrefillDuration = time.Since(stepStart)
			metrics.Engagement = engagement
			logEngagement(engagement)
		} else {
			metrics.DecodeDuration += time.Since(stepStart)
			metrics.Engagement = engagement // update with latest for live SVG
		}
		// Live engagement SVG update (every forward pass in stateless mode).
		if engagement != nil {
			culling.WriteEngagementDiag(e.model.CanonicalModuleMap, e.model.ModelPath, e.model.TensorDims, engagement)
		}
		nextID := e.sample(logits, params)
		if stopSet[nextID] {
			hitStop = true
			break
		}
		if params.LogProbs {
			metrics.TokenLogProbs = append(metrics.TokenLogProbs,
				ComputeTopLogProbs(logits, nextID, params.TopLogProbs, e.tokenizer.TokenString))
		}
		if !onToken(e.tokenizer.TokenString(nextID)) {
			hitStop = true
			break
		}
		metrics.CompletionTokens++
		tokenIDs = append(tokenIDs, nextID)
	}
	if hitStop {
		metrics.FinishReason = "stop"
	} else {
		metrics.FinishReason = "length"
	}
	return nil
}

// logEngagement logs per-layer engagement (1 - cosSim) at DEBUG level.
func logEngagement(ed *arch.EngagementData) {
	if ed == nil {
		return
	}
	var parts []string
	for il := range ed.BlockCosSim {
		bc := ed.BlockCosSim[il]
		fc := ed.FFNCosSim[il]
		parts = append(parts, fmt.Sprintf("L%d blk=%.4f ffn=%.4f", il, 1-bc, 1-fc))
	}
	log.Debug("engagement (1-cos): %s", strings.Join(parts, " | "))
}

func (e *Engine) sample(logits []float32, params GenerateParams) int32 {
	if params.Temperature <= 0 {
		return Greedy(logits)
	}
	return TopP(logits, params.Temperature, params.TopP)
}

// lastUserContent returns the content of the last user message, used as diagnostic context for culling.
func lastUserContent(msgs []ChatMessage) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "user" {
			return msgs[i].Content
		}
	}
	return ""
}
