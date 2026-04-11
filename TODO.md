# TODO

## Critical Issues

* None open

## Serious Issues
* None open

## Tech Debt
* None open

## Features

## Deferred Work
1. **Diffusion generation loop** — iterative masked denoising for LLaDA-MoE; builder support in place, needs new generation strategy in `engine.go`.
2. image support
3. **Chat client streaming + acontextual mode** — `--no-history` flag, real-time SSE output; refactor client to separate impl from CLI entry point.
4. Batch inference - (pad_token handling will be needed)
5. Multiple concurrent models
6. Linux/CUDA support (rename GPU init, add CUDA backend)
