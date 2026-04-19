.PHONY: all llama-cpp update-dependencies setup

# LLAMA.CPP USED FOR REFERENCE ONLY.

# DO NOT MANUALLY EDIT vvv - use make update-dependencies
LLAMA_VERSION := b8809
LLAMA_REPO := https://github.com/ggerganov/llama.cpp.git

LLAMA_DIR := llama.cpp
LLAMA_MARKER := $(LLAMA_DIR)/.git/HEAD

all: setup

setup:
	./setup_venv.sh

llama-cpp: $(LLAMA_MARKER)

$(LLAMA_MARKER):
		@mkdir -p $(dir $(LLAMA_DIR))
		@[[ -d $(LLAMA_DIR) ]] && (cd $(LLAMA_DIR) && git fetch --tags) || git clone $(LLAMA_REPO) $(LLAMA_DIR)
		@cd $(LLAMA_DIR) && git -c advice.detachedHead=false checkout $(LLAMA_VERSION)

MAKEFILE := $(shell pwd)/Makefile
update-dependencies: llama-cpp
	# update llama.cpp to the latest tagged version
	@cd $(LLAMA_DIR) && git tag --sort=-committerdate  | head -1 | xargs git -c advice.detachedHead=false checkout
	# update our borrowed script and requirements
	@( \
	  cp -f $(LLAMA_DIR)/convert_hf_to_gguf.py ./ && \
		cp -f $(LLAMA_DIR)/requirements/*.txt ./requirements/ && \
		git clean -fxd requirements/ > /dev/null \
	)
	# update Makefile with the new version tag.
	@( \
		cd $(LLAMA_DIR) && \
		VTAG=$$(git describe --tag) && \
		sed "s/^LLAMA_VERSION := $(LLAMA_VERSION)/LLAMA_VERSION := $$VTAG/" $(MAKEFILE) > $(MAKEFILE).tmp && \
		mv -f $(MAKEFILE).tmp $(MAKEFILE) && \
		echo "LLAMA_VERSION := $$VTAG" \
	)
