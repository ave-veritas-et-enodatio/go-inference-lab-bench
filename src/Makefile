MAKE         := /usr/bin/make
GGML_LIB_DIR := ggml_lib
BINDIR       := ../bin

.PHONY: all ggml_lib bench test clean

all: ggml_lib bench

ggml_lib:
	$(MAKE) -C $(GGML_LIB_DIR)

bench: ggml_lib
	@mkdir -p $(BINDIR)
	go build -o $(BINDIR)/ ./bench

update-dependencies:
	go mod tidy
	go get -u ./...
	go mod tidy

test: ggml_lib
	go test ./... --count=1

clean:
	$(MAKE) -C $(GGML_LIB_DIR) clean
	rm -f $(BINDIR)/bench
