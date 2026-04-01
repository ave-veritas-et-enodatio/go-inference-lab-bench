MAKE           := /usr/bin/make
GRAPH_TARGETS  := $(patsubst models/arch/%.arch.toml,models/arch/%.arch.svg,$(wildcard models/arch/*.arch.toml))

.PHONY: all build test integration-test equiv-test update-dependencies serve chat arch-editor model-diagrams clean

all: build

build: src/third_party/ggml/.git
	$(MAKE) -C src all
	@[[ -L bin/config ]] || (cd bin && ln -s ../config .)
	@[[ -L bin/models ]] || (cd bin && ln -s ../models .)

src/third_party/ggml/.git:
	git submodule update --init

serve: build
	./bin/bench serve-api 2> bin/serve-api.log

chat: build
	./bin/bench chat

arch-editor: build
	./bin/bench arch-editor

PROMPT ?= explain pi in one short sentence.
integration-test: build
	@(ls models/*.gguf 2>&1) > /dev/null || (echo "$@: No model/.gguf files." 1>&2 && exit 1)
	FORCE_NEW_SERVER=true ALL_MODELS=true THINK=false MAX_TOKENS=500 ./test_inference.sh "$(PROMPT)"

equiv-test: build
	./test_llama_equiv.sh

models/arch/%.arch.svg: models/arch/%.arch.toml
	@[[ -x ./bin/bench ]] || (echo "make build first." 1>&2 && exit 1)
	./bin/bench gen-arch-diagram --layers 32 $(<) $(@)

arch-diagrams: $(GRAPH_TARGETS)


update-dependencies:
	$(MAKE) -C src $@

test:
	$(MAKE) -C src $@

clean:
	$(MAKE) -C src $@
