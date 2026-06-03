# Go Inference Lab Bench

From-scratch Go LLM inference engine for R&D into inference mechanics. Multi-model API server, data-driven architecture definition via TOML DSL, KV-cached and stateless inference, and visualization tooling.

<a href="models/arch/qwen35.arch.svg"><image src="models/arch/qwen35.arch.svg" height="600" alt="Qwen 3.5 Architecture"></a><a href="models/arch/qwen35.layers.svg"><image src="models/arch/qwen35.layers.svg" height="600" alt="Qwen 3.5 Layers"></a>

## Features

- Full inference loop in Go; thin C-only ggml op wrapper for GPU acceleration
- Data-driven architecture definition via TOML DSL — adding architectures is primarily a data-writing operation
- Zero model-specific Go code — chat templates from GGUF `tokenizer.chat_template` via gonja; BOS/EOS from GGUF metadata
- KV-cached and stateless inference
- OpenAI-compatible API (`/api/v1/chat/completions`) with extensions: `stateless`, `enable_thinking`, `elide_thinking`, `logprobs`
- Non-streaming responses include `usage` with token counts, throughput (tokens/sec), and timing
- 7 working architectures (GGUF)
  - Llama 3B
  - Gemma-4 (with vision support),
  - Qwen3.5 & Qwen3.5-MoE (with vision support),
  - DeepSeek-V2,
  - LLaDA, LLaDA-MoE (Diffusion text generation)
- Inference equivalence testing against llama-server (validates logprobs match within FP variance)
- Support for loading from swiftensors (Hugging Face format)
  - Gemma-4, LLaDA, Qwen3.5
- SVG architecture visualizer: `bench gen-arch-diagram` generates `*.arch.svg`, `*.layers.svg` from TOML
  - Gemma-4 and Qwen3.5 also have `*.vision.svg` and `*.vision.layers.svg` 

For architecture details, invariants, and development workflow: **[AGENTS.md](AGENTS.md)** and **[ARCHITECTURE.md](ARCHITECTURE.md)**.

## IMPORTANT NOTE ABOUT CLAUDE.md

Do not grow it past 50 lines. Keep it tight and focused on critical persistent behaviors.
It is injected *every turn* so filling it with heartfelt pleas for various behaviors is a tremendous waste of tokens.

## Quick Start

- download one or more gguf model to `models/[name].gguf`
- download one or more safetensors file set to `models/[name].st/`
- after adding any `models/[name].st/` directory, run `make st-tok-ggufs` to
  build the required `tokenizer.gguf` sidecar in each `.st/` dir (safetensors
  files carry no tokenizer; bench loads it from this sidecar). `make serve`
  runs this automatically, but `./bin/bench serve-api` directly will not.
- if the same `[name]` exists as both a `.gguf` and a `.st/` the .gguf will be used
- there's also a make rule for generating `[name].gguf` from `[name].st/`
  - `make models/[name].gguf`


```bash
# Build (first run compiles ggml, ~1 min)
make

# Run server (logs to bin/bench.log; INFO+ also on stderr)
make serve

# Test inference
./test_inference.sh "What is 2+2?"
./test_inference.sh --loop                # interactive (acontextual)
ALL_MODELS=true ./test_inference.sh "Hi"   # test every loaded model

# Validate logprob equivalence against llama-server (requires Homebrew llama.cpp)
make equiv-test
```

Requirements: macOS M3 Max or better, Go, GNU make, clang, CMake, Python3 (for test scripts). Linux/Windows support coming.

## API

```bash
# List models
curl localhost:11116/api/v1/models

# Chat completion (cached — default, fast)
curl -X POST localhost:11116/api/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","messages":[{"role":"user","content":"Hi"}],"stream":true}'

# Stateless mode (no KV cache)
curl -X POST localhost:11116/api/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","messages":[{"role":"user","content":"Hi"}],"stateless":true}'

# Diffusion generation (LLaDA models only; ignored on autoregressive models)
curl -X POST localhost:11116/api/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","messages":[{"role":"user","content":"Hi"}],"diffusion":{"steps":64,"block_length":64}}'

# Control endpoints
curl localhost:11116/ctl/?memstats   # memory statistics
curl localhost:11116/ctl/?quit       # wait for in-flight inference then shut down
curl localhost:11116/ctl/?quit&now   # immediate shutdown
```

Control endpoint: `/ctl/` (`?memstats` = memory stats; `?quit` = graceful shutdown; `?quit&now` = immediate).

## Bench Commands

```
bench serve-api              Run inference API server
bench gen-arch-diagram       Generate SVG diagrams from TOML
```

See `bin/bench` without arguments for complete help. For detailed config options, see `config/api_config.toml`.

## Third Party Acknowledgements

- [ggml](https://github.com/ggml-org/ggml.git) — Georgi Gerganov's C++ tensor library (inference backend via thin binding layer)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — Georgi Gerganov's C++ inference engine (reference for block implementations; equivalence testing)
- [gguf-parser-go](https://github.com/gpustack/gguf-parser-go) — Frank Mai's Pure Go GGUF file parser
- [gonja](https://github.com/nikolalohinski/gonja) — Nikola Lohinski's Go Jinja2 template engine (chat template execution from GGUF metadata)
- [BurntSushi/toml](https://github.com/BurntSushi/toml) — Andrew Gallant's TOML parser (architecture DSL and config files)
- [go-chi](https://github.com/go-chi/chi) — HTTP router (Vojtech Vitek's OpenAI API implementation)
- [cobra](https://github.com/spf13/cobra) — Steve Francia's CLI framework
