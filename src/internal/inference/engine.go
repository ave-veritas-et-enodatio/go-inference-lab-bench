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
	"inference-lab-bench/internal/log"
	"inference-lab-bench/internal/model"
	"inference-lab-bench/internal/util"
	"inference-lab-bench/internal/ggml"
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
	diagDir     string               // directory for diagnostic output (resolved by caller)
}

// NewEngine creates an inference engine for the given model.
// archDir is the directory containing architecture definition TOML files.
// diagDir is the directory where diagnostic files (cullmap, engagement) are written.
// maxSeqLen is the KV cache size in tokens (0 = default 8192).
// flashAttention is the server default for FA2 (true = enable when head geometry allows).
// Automatically loads a .modulemap sidecar file if present next to the GGUF.
func NewEngine(memStats ggml.MemoryStats, info *model.ModelInfo, archDir, diagDir string, maxSeqLen int, flashAttention bool) (*Engine, error) {
	var (
		m   *arch.GenericModel
		tok *Tokenizer
		err error
	)

	archName := info.Metadata.Architecture
	archDef, err := arch.Load(archDir, archName)
	if err != nil {
		return nil, fmt.Errorf("loading arch def %q: %w", archName, err)
	}

	switch info.Format {
	case model.FormatSafetensors:
		// Load tokenizer from the tokenizer.gguf sidecar in the safetensors directory.
		// This is a minimal GGUF containing only tokenizer metadata (no weights).
		tokPath := filepath.Join(info.Path, "tokenizer.gguf")
		tokF, e := ggufparser.ParseGGUFFile(tokPath)
		if e != nil {
			return nil, fmt.Errorf("parsing tokenizer sidecar: %w", e)
		}

		tok, err = NewTokenizerFromGGUF(tokF, tokPath)
		if err != nil {
			return nil, fmt.Errorf("building tokenizer: %w", err)
		}

		m, err = arch.NewGenericModelFromSafetensors(memStats, maxSeqLen, archDef, info.Path, archDir)
		if err != nil {
			return nil, fmt.Errorf("creating model: %w", err)
		}

	case model.FormatGGUF, "":
		// Existing GGUF path — functionally identical to prior code.
		f, err := ggufparser.ParseGGUFFile(info.Path)
		if err != nil {
			return nil, fmt.Errorf("parsing GGUF for tokenizer: %w", err)
		}

		tok, err = NewTokenizerFromGGUF(f, info.Path)
		if err != nil {
			return nil, fmt.Errorf("building tokenizer: %w", err)
		}

		// DSL-driven model loading — reuse the already-parsed GGUF to avoid double-parsing.
		m, err = arch.NewGenericModelFromGGUF(memStats, maxSeqLen, archDef, info.Path, archDir, f)
		if err != nil {
			return nil, fmt.Errorf("creating model: %w", err)
		}

	default:
		return nil, fmt.Errorf("unsupported model format: %s", info.Format)
	}

	if maxSeqLen <= 0 {
		maxSeqLen = defaultMaxSeqLen
	}

	validateArchTokens(m.Def.Tokens, tok)

	// For safetensors, NLayers may be 0 if config.json lacked it. Fall back to
	// the resolved model param (config.json → params stmap).
	nLayers := info.Metadata.NLayers
	if nLayers == 0 {
		if n, err := m.Params.GetInt("n_layers"); err == nil {
			nLayers = n
		}
	}

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
		nLayers:     nLayers,
		maxSeqLen:   maxSeqLen,
		cullingMeta: cullingMeta,
		flashAttn:   flashAttn,
		diagDir:     diagDir,
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

// IsDiffusion reports whether the loaded model uses diffusion generation.
func (e *Engine) IsDiffusion() bool {
	return e.model != nil && e.model.Def.Architecture.Generation == "diffusion"
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

func (e *Engine) MemoryStats() ggml.MemoryStats {
	if ws := e.WeightStore(); ws != nil {
		return ggml.DevMemory(ws.GPU, ws.CPU)
	}
	return ggml.MemoryStats{}
}

// DiffusionParams groups diffusion-model-specific generation parameters.
// Ignored for autoregressive models.
type DiffusionParams struct {
	Steps       int    // 0 = use default (64)
	BlockLength int    // 0 = single block (global); >0 = block-based left-to-right
	Algorithm   string // "" or "confidence" = max-softmax; stub for future variants
}

// GenerateParams controls the generation loop.
type GenerateParams struct {
	MaxTokens          int
	Temperature        float32
	TopP               float32
	Stateless          bool             // bypass KV cache (for testing/comparison)
	Streaming          bool             // true if the caller is using SSE streaming
	CullMethod         string           // "" or "none" = no culling; "random" = random test pattern
	ThinkingEnabled    bool             // mirrors ChatTemplateKwargs["enable_thinking"]; used for thinkFilter init + internal logic
	ChatTemplateKwargs map[string]any   // passed to the chat template renderer; llama.cpp-compatible shape
	LogProbs           bool             // include log-probabilities in response
	TopLogProbs        int              // number of top log-probabilities per token (0 = just the chosen token)
	FlashAttention     *bool            // nil = use server default; true/false = per-request override
	Diffusion          *DiffusionParams // nil = not diffusion (ignored for autoregressive models)
	RLB                *RLBParams       // nil = RLB disabled; non-nil = run recurrent logic block generation with these params
}

// DefaultParams returns sensible defaults.
func DefaultParams() GenerateParams {
	return GenerateParams{
		MaxTokens:       512,
		Temperature:     0,
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
	// Ensure chat_template_kwargs carries enable_thinking for callers (e.g.
	// CLI paths) that only set ThinkingEnabled. The HTTP handler populates
	// both already; this is the backstop for direct Engine users.
	kwargs := params.ChatTemplateKwargs
	if kwargs == nil {
		kwargs = map[string]any{}
	}
	if _, ok := kwargs["enable_thinking"]; !ok {
		kwargs["enable_thinking"] = params.ThinkingEnabled
	}
	promptIDs, err := e.tokenizer.EncodeChat(messages, kwargs)
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

	// RLB doesn't apply to diffusion models (no SSM state to blend, no
	// per-block recurrence concept). If both are requested, drop RLB and
	// fall through to the diffusion path.
	if params.RLB != nil && e.IsDiffusion() {
		log.Warn("rlb_gen requested on diffusion model; ignoring RLB, performing diffusion generation")
		params.RLB = nil
	}

	var genErr error
	if params.RLB != nil {
		if params.Stateless {
			log.Warn("rlb_gen requires cached mode; ignoring stateless=true")
		}
		log.Info("rlb generation: prompt=%d tokens", len(promptIDs))
		genErr = e.generateRLB(promptIDs, maxTokens, stopSet, params, mask, onToken, metrics)
	} else if e.IsDiffusion() {
		if params.Streaming {
			T := 64
			if params.Diffusion != nil && params.Diffusion.Steps > 0 {
				T = params.Diffusion.Steps
			}
			log.Warn("streaming request on diffusion model; response will be returned as a burst after %d denoising steps", T)
		}
		genErr = e.generateDiffusion(promptIDs, maxTokens, stopSet, params, mask, onToken, metrics)
	} else if params.Stateless {
		log.Info("stateless inference (no KV cache), prompt=%d tokens", len(promptIDs))
		genErr = e.generateStateless(promptIDs, maxTokens, stopSet, params, mask, onToken, metrics)
	} else {
		genErr = e.generateCached(promptIDs, maxTokens, stopSet, params, mask, onToken, metrics)
	}

	metrics.TotalDuration = time.Since(start)

	// Write diagnostic files.
	if cullMap != nil {
		culling.WriteCullDiagnostics(e.diagDir, cullMap, e.model.ModelPath, e.model.TensorDims, metrics.Engagement, e.model.Def.Architecture.NonCausal, e.model.Def.Architecture.Generation)
	}

	return metrics, genErr
}

// enablePerTokenLogging gates verbose per-token diagnostic logging on the
// hot sampling path. Off by default — flipping it to true produces one log
// line per generated token with the sampled ID and top logit, useful when
// debugging flat-distribution or off-by-one-token bugs without rebuilding
// the engine harness. Promote to a config/runtime setting if it proves
// useful often enough to warrant live toggling.
const enablePerTokenLogging = false

func (e *Engine) sample(logits []float32, params GenerateParams) (int32, error) {
	if err := ValidateLogits(logits); err != nil {
		return 0, fmt.Errorf("sample: %w", err)
	}
	var next int32
	if params.Temperature <= 0 {
		next = Greedy(logits)
	} else {
		next = TopP(logits, params.Temperature, params.TopP)
	}
	if enablePerTokenLogging {
		log.Debug("sample: next=%d top_logit=%.4f n_vocab=%d temp=%.2f top_p=%.2f",
			next, logits[next], len(logits), params.Temperature, params.TopP)
	}
	return next, nil
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
