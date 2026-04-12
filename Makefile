MAKE           := /usr/bin/make
GRAPH_TARGETS  := $(patsubst models/arch/%.arch.toml,models/arch/%.arch.svg,$(wildcard models/arch/*.arch.toml))

.PHONY: all build test integration-test equiv-test update-dependencies serve chat arch-editor model-diagrams clean

all: build

build:
	$(MAKE) -C src all
	@[[ -L bin/config ]] || (cd bin && ln -s ../config .)
	@[[ -L bin/models ]] || (cd bin && ln -s ../models .)

serve: build
	./bin/bench serve-api --log bin/bench.log --log-level INFO

chat: build
	./bin/bench chat

arch-editor: build
	./bin/bench arch-editor

equiv-test: build
	./test_equiv.sh stateless
	./test_equiv.sh llama
	./test_equiv.sh standard-attention

PROMPT ?= explain pi in one short sentence.
integration-test: build
	@(ls models/*.gguf 2>&1) > /dev/null || (echo "$@: No model/.gguf files." 1>&2 && exit 1)
	FORCE_NEW_SERVER=true ALL_MODELS=true THINK=true MAX_TOKENS=1000 ./test_inference.sh "$(PROMPT)"

models/arch/%.arch.svg: models/arch/%.arch.toml
	@[[ -x ./bin/bench ]] || (echo "make build first." 1>&2 && exit 1)
	./bin/bench gen-arch-diagram $(<) $(@)

arch-diagrams: build $(GRAPH_TARGETS)

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
