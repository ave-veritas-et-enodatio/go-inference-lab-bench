# Inference Lab Bench

From-scratch Go LLM inference engine for R&D into inference mechanics. Multi-model API server, data-driven architecture definition via TOML DSL, KV-cached and stateless inference, and visualization tooling.

<a href="models/arch/qwen35.arch.svg"><image src="models/arch/qwen35.arch.svg" height="600" alt="Qwen 3.5 Architecture"></a><a href="models/arch/qwen35.layers.svg"><image src="models/arch/qwen35.layers.svg" height="600" alt="Qwen 3.5 Layers"></a>

## Features

- Full inference loop in Go; thin C-only ggml op wrapper for GPU acceleration
- Data-driven architecture definition via TOML DSL — adding architectures is primarily a data-writing operation
- Zero model-specific Go code — chat templates from GGUF `tokenizer.chat_template` via gonja; BOS/EOS from GGUF metadata
- KV-cached and stateless inference
- OpenAI-compatible API (`/api/v1/chat/completions`) with extensions: `stateless`, `enable_thinking`, `elide_thinking`, `logprobs`
- Non-streaming responses include `usage` with token counts, throughput (tokens/sec), and timing
- 7 working architectures: Llama 3B, Qwen3.5 4B/9B, Qwen3.5-MoE 30B-A3B, DeepSeek-V2-Lite, Gemma4 4B/26B, LLaDA-MoE 7B; LLaDA 8B built but not yet tested against a live model
- Diffusion generation for LLaDA models — iterative masked denoising with configurable steps and block length
- SVG architecture visualizer: `bench gen-arch-diagram` generates `*.arch.svg` and `*.layers.svg` from TOML
- Inference equivalence testing against llama-server (validates logprobs match within FP variance)
- Support for swiftensor models

Known working models:
- [qwen35-9b-opus46-mix-i1-Q4_K_M.gguf](https://huggingface.co/slyfox1186/qwen35-9b-opus46-mix-i1-GGUF/resolve/main/qwen35-9b-opus46-mix-i1-Q4_K_M.gguf) - hybrid SSM + attention
- [Llama-3.2-3B-Instruct-f16.gguf](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-f16.gguf) — standard attention
- [llama-3.2-3b-instruct-q4_k_m.gguf](https://huggingface.co/hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF/resolve/main/llama-3.2-3b-instruct-q4_k_m.gguf) — quantized
- [gemma-4-E4B-it-Q4_K_M.gguf](https://huggingface.co/lmstudio-community/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf) — ISWA + GeGLU
- [gemma-4-26B-A4B-it-MXFP4_MOE](https://huggingface.co/noctrex/gemma-4-26B-A4B-it-MXFP4_MOE-GGUF/resolve/main/gemma-4-26B-A4B-it-MXFP4_MOE.gguf) — MoE ISWA + GeGLU
- [LLaDA-MoE-7B-A1B-Instruct.i1-Q4_K_M](https://huggingface.co/mradermacher/LLaDA-MoE-7B-A1B-Instruct-i1-GGUF/resolve/main/LLaDA-MoE-7B-A1B-Instruct.i1-Q4_K_M.gguf) — MoE, Text Diffusion
- [LLaDA-8B-Instruct.i1-Q4_K_M.gguf](https://huggingface.co/mradermacher/LLaDA-8B-Instruct-i1-GGUF/resolve/main/LLaDA-8B-Instruct.i1-Q4_K_M.gguf) — Dense, Text Diffusion
- [Qwen3.5-9B.st/](https://www.modelscope.cn/Qwen/Qwen3.5-9B.git) - (Instruct) Official safetensors, hybrid SSM + attention, git lfs needed
- [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct/tree/main) - Official safetensors, text diffusion, git xet needed

For architecture details, invariants, and development workflow: **[AGENTS.md](AGENTS.md)** and **[ARCHITECTURE.md](ARCHITECTURE.md)**.

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
bash test_inference.sh "What is 2+2?"
bash test_inference.sh --loop                # interactive (acontextual)
ALL_MODELS=true bash test_inference.sh "Hi"   # test every loaded model

# Validate logprob equivalence against llama-server (requires Homebrew llama.cpp)
make equiv-test

# Interactive chat (server must be running)
make chat
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
bench chat                   Interactive chat client
bench gen-arch-diagram       Generate SVG diagrams from TOML
```

See `bin/bench` without arguments for complete help. For detailed config options, see `config/api_config.toml`.

## Third Party Acknowledgements

- [ggml](https://github.com/ggml-org/ggml.git) — Georgi Gerganov's C++ tensor library (inference backend via thin binding layer)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — Georgi Gerganov's C++ inference engine (reference for block implementations; equivalence testing)
- [gguf-parser-go](https://github.com/gpustack/gguf-parser-go) — Frank Mai's Pure Go GGUF file parser
- [gonja](https://github.com/nikolalohinski/gonja) — Nikola Lohinski's Go Jinja2 template engine (chat template execution from GGUF metadata)
- [BurntSushi/toml](https://github.com/BurntSushi/toml) — Andrew Gallant's TOML parser (architecture DSL and config files)
- [go-chi](https://github.com/go-chi/chi) — HTTP router (OpenAI API implementation)
- [go-sqlite3](https://github.com/mattn/go-sqlite3) — Matt N.'s sqlite3 driver (chat client history)
- [langchaingo](https://github.com/tmc/langchaingo) — Travis Cline's OpenAI client library (chat client)
- [cobra](https://github.com/spf13/cobra) — Steve Francia's CLI framework
