MAKE           := /usr/bin/make
GRAPH_TARGETS  := $(patsubst models/arch/%.arch.toml,models/arch/%.arch.svg,$(wildcard models/arch/*.arch.toml))

.PHONY: all build test serve chat arch-editor model-diagrams clean

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

test:
	$(MAKE) -C src test

models/arch/%.arch.svg: models/arch/%.arch.toml
	@[[ -x ./bin/bench ]] || (echo "make build first." 1>&2 && exit 1)
	./bin/bench gen-arch-diagram --layers 32 $(<) $(@)

arch-diagrams: $(GRAPH_TARGETS)

clean:
	$(MAKE) -C src clean
