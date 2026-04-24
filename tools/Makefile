.PHONY: all llama-cpp update-dependencies setup

# LLAMA.CPP USED FOR REFERENCE ONLY.

# DO NOT MANUALLY EDIT vvv - use make update-dependencies
LLAMA_VERSION := b8892
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

update-dependencies: llama-cpp
	# update llama.cpp to the latest tagged version
	@git -C $(LLAMA_DIR) fetch --tags
	@git -C $(LLAMA_DIR) tag --sort=committerdate  | tail -1 | xargs git -C $(LLAMA_DIR) -c advice.detachedHead=false checkout
	# update our borrowed script and requirements
	@cp -f $(LLAMA_DIR)/convert_hf_to_gguf.py ./
	@rm -rf requirements/
	@cp -r $(LLAMA_DIR)/requirements .
	@git clean -fxd requirements/ > /dev/null
	@rm -rf gguf/
	@cp -r $(LLAMA_DIR)/gguf-py/gguf .
	# update Makefile with the new version tag.
	@( \
		VTAG=$$(git -C $(LLAMA_DIR) describe --tag) && \
		sed "s/^LLAMA_VERSION := $(LLAMA_VERSION)/LLAMA_VERSION := $$VTAG/" Makefile > Makefile.tmp && \
		mv -f Makefile.tmp Makefile && \
		echo "LLAMA_VERSION: $(LLAMA_VERSION) -> $$VTAG" \
	)
