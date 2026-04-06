MAKE           := /usr/bin/make
GRAPH_TARGETS  := $(patsubst models/arch/%.arch.toml,models/arch/%.arch.svg,$(wildcard models/arch/*.arch.toml))

.PHONY: all build test integration-test equiv-test llama-equiv-test stateless-equiv-test update-dependencies serve chat arch-editor model-diagrams clean ggml

all: build

GGML_DIR := src/third_party/ggml
GGML_MARKER := $(GGML_DIR)/.git/HEAD
GGML_VERSION ?= v0.9.11
GGML_REPO := https://github.com/ggerganov/ggml.git
ggml: $(GGML_MARKER)

build: ggml
	$(MAKE) -C src all
	@[[ -L bin/config ]] || (cd bin && ln -s ../config .)
	@[[ -L bin/models ]] || (cd bin && ln -s ../models .)

$(GGML_MARKER):
	@mkdir -p $(dir $(GGML_DIR))
	@git clone $(GGML_REPO) $(GGML_DIR)
	@cd $(GGML_DIR) && git checkout $(GGML_VERSION)

serve: build
	./bin/bench serve-api --log bin/bench.log --log-level INFO

chat: build
	./bin/bench chat

arch-editor: build
	./bin/bench arch-editor

llama-equiv-test: build
	./test_equiv.sh llama

stateless-equiv-test: build
	./test_equiv.sh stateless

equiv-test: build
	./test_equiv.sh stateless
	./test_equiv.sh llama

PROMPT ?= explain pi in one short sentence.
integration-test: build
	@(ls models/*.gguf 2>&1) > /dev/null || (echo "$@: No model/.gguf files." 1>&2 && exit 1)
	FORCE_NEW_SERVER=true ALL_MODELS=true THINK=false MAX_TOKENS=500 ./test_inference.sh "$(PROMPT)"

models/arch/%.arch.svg: models/arch/%.arch.toml
	@[[ -x ./bin/bench ]] || (echo "make build first." 1>&2 && exit 1)
	./bin/bench gen-arch-diagram --layers 32 $(<) $(@)

arch-diagrams: $(GRAPH_TARGETS)

update-attributions:
	claude -p "read AGENTS.md and ARCHITECTURE.md, then read the direct imports section of src/go.mod and update 'Third Party Acknowledgements' at the end of README.md" \
	      --allowedTools "Read,Edit,Write,Glob,Grep" \
				--model sonnet

update-dependencies:
	$(MAKE) -C src $@

test:
	$(MAKE) -C src $@

clean:
	$(MAKE) -C src $@
