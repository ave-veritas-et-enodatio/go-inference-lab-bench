GGML_LIB_DIR := ggml_lib
BINDIR       := ../bin

.PHONY: all ggml ggml_lib bench test clean update-dependencies update-ggml-dependency

all: bench

# DO NOT MANUALLY EDIT vvv - use make update-dependencies
GGML_VERSION := v0.10.0
GGML_REPO := https://github.com/ggerganov/ggml.git

GGML_DIR := third_party/ggml
GGML_MARKER := $(GGML_DIR)/.git/HEAD
ggml: $(GGML_MARKER)

$(GGML_MARKER):
		@mkdir -p $(dir $(GGML_DIR))
		@[[ -d $(GGML_DIR) ]] && (cd $(GGML_DIR) && git fetch --tags) || git clone $(GGML_REPO) $(GGML_DIR)
		@cd $(GGML_DIR) && git -c advice.detachedHead=false checkout $(GGML_VERSION)

ggml_lib: ggml
	@echo "make -C $(GGML_LIB_DIR)"
	@$(MAKE) -C $(GGML_LIB_DIR)

bench: ggml_lib
	@mkdir -p $(BINDIR)
	go build -o $(BINDIR)/ ./bench

update-dependencies: update-ggml-dependency
	go mod tidy
	go get -u ./...
	go mod tidy

update-ggml-dependency:
	# update to the latest tagged version
	@git -C $(GGML_DIR) fetch --tags
	@git -C $(GGML_DIR) tag --sort=committerdate  | tail -1 | xargs git -C $(GGML_DIR) -c advice.detachedHead=false checkout
	# update Makefile with the new version tag.
	@(\
	  VTAG=$$(git -C $(GGML_DIR) describe --tag) && \
		sed "s/^GGML_VERSION := $(GGML_VERSION)/GGML_VERSION := $$VTAG/" Makefile > Makefile.tmp && \
		mv -f Makefile.tmp Makefile && \
		echo "GGML_VERSION: $(GGML_VERSION) -> $$VTAG" \
	)

test: bench
	go vet ./...
	go test ./... --count=1

clean:
	@$(MAKE) -C $(GGML_LIB_DIR) clean
	rm -f $(BINDIR)/bench
