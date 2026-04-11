MAKE         := /usr/bin/make
GGML_LIB_DIR := ggml_lib
BINDIR       := ../bin

.PHONY: all ggml ggml_lib bench test clean update-dependencies

all: bench

# DO NOT MANUALLY EDIT - use make update-dependencies
GGML_VERSION := v0.9.11
GGML_REPO := https://github.com/ggerganov/ggml.git

GGML_DIR := third_party/ggml
GGML_MARKER := $(GGML_DIR)/.git/HEAD
ggml: $(GGML_MARKER)

$(GGML_MARKER):
		@mkdir -p $(dir $(GGML_DIR))
		@[[ -d $(GGML_DIR) ]] && (cd $(GGML_DIR) && git fetch --tags) || git clone $(GGML_REPO) $(GGML_DIR)
		@cd $(GGML_DIR) && git checkout $(GGML_VERSION)

ggml_lib: ggml
	$(MAKE) -C $(GGML_LIB_DIR)

bench: ggml_lib
	@mkdir -p $(BINDIR)
	go build -o $(BINDIR)/ ./bench

update-dependencies:
	go mod tidy
	go get -u ./...
	go mod tidy
	# update to the latest tagged version
	@cd $(GGML_DIR) && git fetch --tags && git tag | sort -Vr | head -1 | tee | xargs git checkout
	# update Makefile with the new version tag.
	@cd $(GGML_DIR) && (\
	  VTAG=$$(git describe --tag) && \
		sed "s/^GGML_VERSION := $(GGML_VERSION)/GGML_VERSION := $$VTAG/" ../../Makefile > ../../Makefile.tmp && \
		mv -f ../../Makefile.tmp ../../Makefile \
	)

test: bench
	go vet ./...
	go test ./... --count=1

clean:
	$(MAKE) -C $(GGML_LIB_DIR) clean
	rm -f $(BINDIR)/bench
