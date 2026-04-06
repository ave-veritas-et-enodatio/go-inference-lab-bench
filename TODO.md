# TODO

## Current - Gemma4 Support

### Critical Issues
* None Open

### Serious Issues
* None Open

## Future

### Deferred Work
1. **Chat client streaming + acontextual mode** — add `--no-history` flag for stateless per-prompt testing with real-time SSE streaming output. Replaces need for test_inference.sh for interactive debugging of thinking models. refactor chat client to separate implementation from CLI entry point
2. Batch inference
3. Multiple concurrent models
4. Linux/CUDA support (rename GPU init, add CUDA backend)
5. **Diffusion generation loop** — iterative masked denoising for non-causal models (LLaDA-MoE). Architecture definition at `models/arch/llada-moe.arch.toml.nyi`; builder support (attention QK-norm, non-causal mask, MoE FFN) already in place. Needs new generation strategy in engine.go.
6. arch.toml editor web app
