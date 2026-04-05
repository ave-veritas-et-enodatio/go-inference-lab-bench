# TODO

## Current - Gemma4 Support

### Critical Issues
* None Open

### Serious Issues
* None Open

## Future

### Deferred Work
1. **Structured logging** — replace `fmt.Fprintf(os.Stderr, ...)` with a real logging package (slog or zerolog). Leveled output, consistent format, eliminate Makefile/test_inference.sh stderr redirects.
  1. creating a simple internal logging interface and backing it with a known package would be a good intermediate step
  2. we could back it with a simple implementation and then opt for a full logging package if the delta merited the dependency weight   
2. **Chat client streaming + acontextual mode** — add `--no-history` flag for stateless per-prompt testing with real-time SSE streaming output. Replaces need for test_inference.sh for interactive debugging of thinking models. refactor chat client to separate implementation from CLI entry point
3. Batch inference
4. Multiple concurrent models
5. Linux/CUDA support (rename GPU init, add CUDA backend)
6. **Diffusion generation loop** — iterative masked denoising for non-causal models (LLaDA-MoE). Architecture definition at `models/arch/llada-moe.arch.toml.nyi`; builder support (attention QK-norm, non-causal mask, MoE FFN) already in place. Needs new generation strategy in engine.go.
