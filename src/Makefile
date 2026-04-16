GGML_LIB_DIR := ggml_lib
BINDIR       := ../bin

.PHONY: all ggml ggml_lib bench test clean update-dependencies update-ggml-dependency

all: bench

# DO NOT MANUALLY EDIT vvv - use make update-dependencies
GGML_VERSION := v0.9.11-27-g58c38058
GGML_REPO := https://github.com/ggerganov/ggml.git

GGML_DIR := third_party/ggml
GGML_MARKER := $(GGML_DIR)/.git/HEAD
ggml: $(GGML_MARKER)

$(GGML_MARKER):
		@mkdir -p $(dir $(GGML_DIR))
		@[[ -d $(GGML_DIR) ]] && (cd $(GGML_DIR) && git fetch --tags) || git clone $(GGML_REPO) $(GGML_DIR)
		@cd $(GGML_DIR) && git -c advice.detachedHead=false checkout $(GGML_VERSION)

ggml_lib: ggml
	@$(MAKE) -C $(GGML_LIB_DIR)

bench: ggml_lib
	@mkdir -p $(BINDIR)
	go build -o $(BINDIR)/ ./bench

MAKEFILE := $(shell pwd)/Makefile
update-dependencies:
	go mod tidy
	go get -u ./...
	go mod tidy
# pinned to v0.9.11-g58c38058 until v0.9.12 comes out.
#	@$(MAKE) update-ggml-dependency

update-ggml-dependency:
	# update to the latest tagged version
	@cd $(GGML_DIR) && git tag --sort=-committerdate  | head -1 | xargs git -c advice.detachedHead=false checkout
	# update Makefile with the new version tag.
	@cd $(GGML_DIR) && (\
	  VTAG=$$(git describe --tag) && \
		sed "s/^GGML_VERSION := $(GGML_VERSION)/GGML_VERSION := $$VTAG/" $(MAKEFILE) > $(MAKEFILE).tmp && \
		mv -f $(MAKEFILE).tmp $(MAKEFILE) && \
		echo "GGML_VERSION := $$VTAG" \
	)

test: bench
	go vet ./...
	go test ./... --count=1

clean:
	@$(MAKE) -C $(GGML_LIB_DIR) clean
	rm -f $(BINDIR)/bench
