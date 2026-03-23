GGML_DIR    := ../third_party/ggml
GGML_BUILD  := $(GGML_DIR)/build
GGML_INC    := $(GGML_DIR)/include
GGML_LIB    := $(GGML_BUILD)/src/libggml.a

CC          ?= clang
CFLAGS      := -std=c17 -O2 -I$(GGML_INC)
AR          ?= ar

SRCDIR      := src
BUILDDIR    := build
LIBDIR      := $(BUILDDIR)/lib

SRCS        := $(wildcard $(SRCDIR)/*.c)
OBJS        := $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.o,$(SRCS))
LIB         := $(LIBDIR)/libggml_lib.a

.PHONY: all ggml clean

all: ggml $(LIB)

ggml: $(GGML_LIB)

$(GGML_LIB):
	mkdir -p $(GGML_BUILD)
	cd $(GGML_BUILD) && cmake .. \
		-DCMAKE_BUILD_TYPE=Release \
		-DBUILD_SHARED_LIBS=OFF \
		-DGGML_METAL=ON \
		-DGGML_METAL_EMBED_LIBRARY=ON \
		-DGGML_NATIVE=ON \
		-DGGML_BUILD_TESTS=OFF \
		-DGGML_BUILD_EXAMPLES=OFF
	cd $(GGML_BUILD) && cmake --build . --parallel

$(BUILDDIR)/%.o: $(SRCDIR)/%.c $(SRCDIR)/ggml_ops.h
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB): $(OBJS)
	@mkdir -p $(LIBDIR)
	$(AR) rcs $@ $^

clean:
	rm -rf $(GGML_BUILD) $(BUILDDIR)
