DIAGRAM_TARGETS  := $(patsubst models/arch/%.arch.toml,models/arch/%.arch.svg,$(wildcard models/arch/*.arch.toml))
ST_TOK_TARGETS  := $(patsubst models/%.st/tokenizer.json,models/%.st/tokenizer.gguf,$(wildcard models/*.st/tokenizer.json))
ST_GGUF_TARGETS := $(patsubst models/%.st,models/%.gguf,$(wildcard models/*.st))
.PHONY: all arch-diagrams arch-diagram-targets build test integration-test equiv-test update-dependencies serve chat model-diagrams clean agents st-tok-ggufs st-ggufs

all: build

build:
	$(MAKE) -C src all
	@[[ -L bin/config ]] || (cd bin && ln -s ../config .)
	@[[ -L bin/models ]] || (cd bin && ln -s ../models .)

LOG_LEVEL ?= INFO
serve: build st-tok-ggufs
	./bin/bench serve-api --log bin/bench.log --log-level $(LOG_LEVEL)

chat: build
	./bin/bench chat

equiv-test: build
	./test_equiv.sh stateless
	./test_equiv.sh llama
	./test_equiv.sh standard-attention
	./test_equiv.sh gguf-st

PROMPT ?= explain pi in one short sentence.
integration-test: build
	@(ls models/*.gguf 2>&1) > /dev/null || (echo "$@: No model/.gguf files." 1>&2 && exit 1)
	@(rm -f ./bin/itest.log 2>&1) > /dev/null || true
	LOG=./bin/itest.log FORCE_NEW_SERVER=true ALL_MODELS=true THINK=true MAX_TOKENS=1000 ./test_inference.sh "$(PROMPT)"
	@grep '\[ERROR\]' ./bin/itest.log && exit 1 || true

st-tok-ggufs: $(ST_TOK_TARGETS)
st-ggufs: $(ST_GGUF_TARGETS)

models/arch/%.arch.svg: models/arch/%.arch.toml
	@[[ -x ./bin/bench ]] || (echo "make build first." 1>&2 && exit 1)
	./bin/bench gen-arch-diagram $(<) $(@)

arch-diagram-targets: $(DIAGRAM_TARGETS)

arch-diagrams: build
	@$(MAKE) -B arch-diagram-targets

update-attributions:
	claude -p "read AGENTS.md and ARCHITECTURE.md, then read the direct imports section of src/go.mod and update 'Third Party Acknowledgements' at the end of README.md" \
	      --allowedTools "Read,Edit,Write,Glob,Grep" \
				--model sonnet

update-dependencies:
	@$(MAKE) -C src $@

test:
	@$(MAKE) -C src $@

clean:
	@$(MAKE) -C src $@

# used only for reference - llama.cpp is not built or used for inference
llama-cpp:
	@$(MAKE) -C tools $@

# leave this rule at the end - SECONDEXPANSION applies to everything after it
# handles on-demand generation of f16 ggufs from safetensor directories.
# DO NOT make st-ggufs a dependency of anything else. this is not a 1-second process
.SECONDEXPANSION:

#	Handles creation of tokenizer-only thing gguf sidecar in [model].st/ safetensor directories.
# 1. it's quick and cheap
# 2. the sidecar is required for loading a safetensor model directly in the server.
models/%.st/tokenizer.gguf: models/%.st/tokenizer.json models/%.st/tokenizer_config.json $$(wildcard models/%.st/*.jinja)
	./tools/hf_to_gguf.sh --bench-tokenizer $(dir $(<))

models/%.gguf: $$(wildcard models/%.st/*.json) $$(wildcard models/%.st/*.safetensors) $$(wildcard models/%.st/*.jinja) $$(wildcard models/%.st/*.py)
	./tools/hf_to_gguf.sh $(dir $(<)) --outtype f16 --outfile $(@)
