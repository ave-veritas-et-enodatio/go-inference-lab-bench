# Go Inference Lab Bench

A [from-scratch Go language LLM lab bench](https://bennherrera.dev/writing/go-inference-lab-bench/) for doing R&D related to the mechanics of inference.

<a href="models/arch/qwen35.arch.svg"><image src="models/arch/qwen35.arch.svg" height="600" alt="Qwen 3.5 Architecture"></a><a href="models/arch/qwen35.layers.svg"><image src="models/arch/qwen35.layers.svg" height="600" alt="Qwen 3.5 Layers"></a>

Key features:
* Does not use [llama.cpp](#third-party-acknowledgements) binding (too black of a box)
  * Full inference loop written in Go and available for exploration and modification 
  * Built on a fine-grain, thin-layer [ggml](#third-party-acknowledgements) binding for GPU acceleration
* Data driven model architecture definitions (see `models/arch/*.arch.toml`)
  * Defined declaratively in toml, block builders implement the graph-level ops in Go.
  * Graphic architecture visualization from definitions (see `models/arch/` for `*.arch.svg` and `*.layers.svg`)
    * These are generated via ```bin/bench gen-arch-diagram <path-to-arch-toml>``` 
  * Architecture definition editor (WIP) ```bin/bench arch-editor``` then browse to [http://localhost:8080](http://localhost:8080)
* Adding new model architectures is primarily a data writing operation
  * Some models may require a new block definition to be coded in go
  * This is still much less cumbersome than 'entire new source file for each architecture'
* Inference service provided via local http server (standard OpenAI API at /api/v1)
  * Logprobs support (`logprobs` + `top_logprobs` in request)
  * Graceful shutdown via `/ctl?quit` (waits for in-flight inference) or `/ctl?quit&now`
* Acontextual testing script for running repeated tests under identical starting conditions
* Inference equivalence testing against llama-server (validates logprobs match within FP variance)
* Simple chat client for contextual chat testing
* Go language choice provides multiple benefits
  * Easy for humans: clean, easy-to-read syntax that builds and runs fast
  * Eash for agents: congitive load is lower than python and much lower than c++
  * Easy to share results: zip up the bin/ dir minus the .ggufs, send zip + model download link, others can verify results.

## Requirements
* macOS M3 Max or better
  * Linux and Windows support coming, but not here yet 
* Go language
* Gnu make
* For ggml third party library
  * A C++ compiler (clang strongly recommended)
  * CMake
* Python3 (used with no dependencies or external files)
  * Used for json processing in `test_inference.sh`
  * If you only use chat client (```bin/bench chat```) you can skip this

## Quick Start

Download one or more supported GGUF models to `models/`

Supported architectures:
* Qwen3.5 Dense and MoE
* Llama
* Deepseek2

I've been using these:
* [Qwen3.5-4B_Abliterated.f16.gguf](https://huggingface.co/mradermacher/Qwen3.5-4B_Abliterated-GGUF/resolve/main/Qwen3.5-4B_Abliterated.f16.gguf)
  * Qwen3.5, hybrid SSM + attention
* [Qwen3.5-9B-abliterated.f16.gguf](https://huggingface.co/mradermacher/Qwen3.5-9B-abliterated-GGUF/resolve/main/Qwen3.5-9B-abliterated.f16.gguf)
  * Qwen3.5, hybrid SSM + attention
* [Llama-3.2-3B-Instruct-f16.gguf](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-f16.gguf)
  * Llama 3.2, standard attention 
* [llama-3.2-3b-instruct-q4_k_m.gguf](https://huggingface.co/hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF/resolve/main/llama-3.2-3b-instruct-q4_k_m.gguf)
  * Llama 3.2, standard attention  (quantized)

```bash
# Build (first run compiles ggml, takes ~1 min, after that much faster)
make

# Run server
make serve

# Test
[FORCE_NEW_SERVER=<false|true>] ./test_inference.sh "What is 2+2?"        # → 4
[FORCE_NEW_SERVER=<false|true>] ./test_inference.sh --loop                # interactive loop
```

* if FORCE_NEW_SERVER=true it will kill any running inference server and start its own
* see script text for other envar controls
* Loop mode is *acontextual* each prompt is presented in isolation with no history
  * it's intended for repeating identical tests, not chatting.

To validate inference correctness against llama-server (requires Homebrew `llama.cpp`):
```bash
make equiv-test   # compares top-1 logprobs — all models should match within FP variance
```

For interactive chatting with persistent context:
```bash
make chat   # builds + runs bench chat (server must already be running)
```

## Model Architecture SVG Visualizer 

```make arch-diagrams``` (re)builds SVG graph renderings of model architectures defined in `models/arch/*.arch.toml`
* Uses `bench gen-arch-diagram` to generate architecture diagrams from TOML definitions

## API

```bash
# List models
curl localhost:11116/api/v1/models

# Chat completion (cached — default, fast)
curl -X POST localhost:11116/api/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": true
  }'

# With logprobs (top 3 alternatives per token)
curl -X POST localhost:11116/api/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hi"}],
    "logprobs": true,
    "top_logprobs": 3
  }'

# Stateless mode (no KV cache — for correctness testing / comparison)
curl -X POST localhost:11116/api/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hi"}],
    "stateless": true
  }'

# Graceful shutdown
curl localhost:11116/ctl?quit       # waits for in-flight inference to finish
curl localhost:11116/ctl?quit&now   # immediate shutdown
```

NOTE: see `test_inference.sh` - the commands above are for doc purposes, not normal workflow

## Project Layout
for file-level details see `##Source Layout` in [AGENTS.md](AGENTS.md)

```
Makefile                        Top-level: delegates to src/, symlinks bin/
config/                         bench subcommand config files
models/
  *.gguf                        Model files (not committed)
  arch/                         Architecture TOML definitions (*.arch.toml) + generated SVGs (*.arch.svg, *.layers.svg)
    model_arch_toml_dsl_spec.md               DSL specification
    block_svg/                  Hand-crafted SVG fragments per block builder
    editor/                     Web-based TOML editor (HTML/JS/CSS)
src/
  Makefile                      Go build, ggml build, test
  go.mod / go.sum               Module: inference-lab-bench
  bench/                        CLI entry point (cobra subcommands — thin wrappers that dispatch to internal/)
  internal/
    util/                       Project-wide utilities (LoadTOML, WriteJSON, BenchPaths, extension constants)
    apiserver/                  HTTP handlers (OpenAI-compatible: /api/v1/*)
    chatclient/                 Interactive chat client implementation
    model/                      GGUF scanning + metadata
    archeditor/                 Arch editor web server
    inference/                  Inference engine
      arch/                     Model architecture handling, forward pass
      ggml/                     Go wrappers for ggml ops (~36 functions)
  ggml_lib/                     C op wrappers + ggml build
  third_party/ggml/             ggml git submodule
test_inference.sh               Test harness
test_llama_equiv.sh             Validates inference equivalence with llama-server
```

## Config

`config/api_config.toml`:
```toml
[server]
host = "0.0.0.0"
port = 11116
auth_token = ""       # optional Bearer token

[models]
directory = "models"
default = "first"     # "first", "last", or explicit model filename (without .gguf)

[inference]
max_seq_len = 8192                       # KV cache size in tokens
enable_thinking_default = true           # passed to template as enable_thinking variable
elide_thinking_default = true            # strip <think>...</think> from output
log_thinking = false                     # log <think>...</think> content to stderr
```

## Bench Commands

```
bench serve-api              Run inference API server
bench chat                   Interactive chat client
bench gen-arch-diagram       Generate SVG architecture + layer diagrams from TOML definitions
bench arch-editor            Launch web-based TOML architecture editor
```

Run `bin/bench` without arguments for complete help.

## Adding a model architecture
Adding architectures is doable in about an hour with AI assistance, if you need to code a new block. If you can use existing blocks, it's around 15 minutes.

* see `models/arch/model_arch_toml_dsl_spec.md` for the block and architecture format spec
* If a new block type is needed:
  * add a Go file that implements it in `src/internal/inference/arch/block_<name>`
  * add an SVG snippet for it to `models/arch/block_svg/<name>.svg`
  * edit `models/arch/editor/editor.js` to include BUILDER_ table entries for your new block type
* Add the new architecture
  * create a new toml file for the architecture at `models/arch/<arch-name>.arch.toml`
  * may be easiest to derive from existing .arch.toml - AI assistance is a big help here
  * try using the visual editor (it's experimental, YMMV)
  * supported architectures are auto-detected from `.arch.toml` files — no code changes needed
  * run and test
  * run `./test_llama_equiv.sh` to verify the new architecture's inference behavior does not diverge from llama.cpp's
    * unless that's the goal, of course. 

## Third Party Acknowledgements
* [ggml](https://github.com/ggml-org/ggml.git) - [Georgi Gerganov](https://github.com/ggerganov)'s C++ tensor library for machine learning
  * Used in inference engine via thin binding layer
* [llama.cpp](https://github.com/ggml-org/llama.cpp) - Georgi Gerganov's C++ inference engine
  * Reference for block implementations
* [gguf-parser-go](github.com/gpustack/gguf-parser-go) - [Frank Mai](https://github.com/thxCode)'s Pure Go GGUF file parser
* [gonja](https://github.com/NikolaLohinski/gonja) - [Nikola Lohinski](https://github.com/NikolaLohinski) and [Axel H](https://github.com/noirbizarre)'s Jinja2 template processor in pure Go
  * Used to handle rendering prompt templates from GGUF files
* [BurntSushi/toml](github.com/BurntSushi/toml) - [Andrew Gallant](https://github.com/BurntSushi)'s TOML parser
  * Model architecture DSL and config files
* [regexp2](github.com/dlclark/regexp2) - [Doug Clark](https://github.com/dlclark)'s regular expression engine 
* [go-chi](github.com/go-chi/chi) - HTTP services router
  * Used by server implementation of OpenAI API
* [go-sqlite3](https://github.com/mattn/go-sqlite3) - [Matt N.](https://github.com/mattn)'s  sqlite3 driver
  * Used in the simple chat client for accumulating history
* [langchaingo](github.com/tmc/langchaingo) - [Travis Cline](https://github.com/tmc)'s OpenAI API client interaction library
  * Used in the simple chat client
* [cobra](github.com/spf13/cobra) - [Steve Francia](https://github.com/spf13)'s CLI parameter handler
