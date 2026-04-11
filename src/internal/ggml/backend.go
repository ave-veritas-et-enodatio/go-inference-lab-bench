package ggml

/*
#cgo CFLAGS: -std=c17 -I${SRCDIR}/../../ggml_lib/src
#include "ggml_ops.h"
#include <stdlib.h>
*/
import "C"
import "unsafe"

// GraphContext wraps a ggml_context used for building computation graphs.
type GraphContext struct {
	ptr     unsafe.Pointer
	memSize int
}

func NewGraphContext(memSize int) *GraphContext {
	p := C.ggml_go_init(C.size_t(memSize))
	if p == nil {
		return nil
	}
	return &GraphContext{ptr: unsafe.Pointer(p), memSize: memSize}
}

func (gc *GraphContext) Free() {
	if gc.ptr != nil {
		C.ggml_go_free(cCtx(gc.ptr))
		gc.ptr = nil
	}
}

func (gc *GraphContext) Reset() {
	C.ggml_go_free(cCtx(gc.ptr))
	gc.ptr = unsafe.Pointer(C.ggml_go_init(C.size_t(gc.memSize)))
}

func (gc *GraphContext) c() C.ggml_go_context { return cCtx(gc.ptr) }

// SetPtr sets the raw pointer (used when constructing from C-allocated contexts).
func (gc *GraphContext) SetPtr(p unsafe.Pointer) { gc.ptr = p }

// Graph wraps a ggml_cgraph computation graph.
type Graph struct{ ptr unsafe.Pointer }

func NewGraph(ctx *GraphContext, maxNodes int) *Graph {
	p := C.ggml_go_new_graph(ctx.c(), C.int(maxNodes))
	if p == nil {
		return nil
	}
	return &Graph{ptr: unsafe.Pointer(p)}
}

func (g *Graph) BuildForwardExpand(t Tensor) {
	C.ggml_go_build_forward_expand(cGraph(g.ptr), t.c())
}

// Backend wraps a ggml backend (GPU or CPU).
type Backend struct{ ptr unsafe.Pointer }

func MetalInit() *Backend {
	p := C.ggml_go_metal_init()
	if p == nil {
		return nil
	}
	return &Backend{ptr: unsafe.Pointer(p)}
}

func CPUInit() *Backend {
	p := C.ggml_go_cpu_init()
	if p == nil {
		return nil
	}
	return &Backend{ptr: unsafe.Pointer(p)}
}

func (b *Backend) Free() {
	if b.ptr != nil {
		C.ggml_go_backend_free(cBackend(b.ptr))
		b.ptr = nil
	}
}

func (b *Backend) c() C.ggml_go_backend     { return cBackend(b.ptr) }
func (b *Backend) bufType() C.ggml_go_buf_type { return C.ggml_go_backend_buf_type(b.c()) }

// Buffer wraps a ggml backend buffer.
type Buffer struct{ ptr unsafe.Pointer }

func AllocCtxTensors(ctx *GraphContext, backend *Backend) *Buffer {
	p := C.ggml_go_alloc_ctx_tensors(ctx.c(), backend.c())
	if p == nil {
		return nil
	}
	return &Buffer{ptr: unsafe.Pointer(p)}
}

func (buf *Buffer) Free() {
	if buf.ptr != nil {
		C.ggml_go_buffer_free(cBuffer(buf.ptr))
		buf.ptr = nil
	}
}

func (buf *Buffer) Size() int {
	return int(C.ggml_go_buffer_size(cBuffer(buf.ptr)))
}

// Clear writes value to every byte in the buffer (including alignment padding).
func (buf *Buffer) Clear(value byte) {
	C.ggml_go_buffer_clear(cBuffer(buf.ptr), C.uint8_t(value))
}

// Sched wraps a ggml multi-backend scheduler.
type Sched struct{ ptr unsafe.Pointer }

func NewSched(metal, cpu *Backend, graphSize int) *Sched {
	p := C.ggml_go_sched_new(metal.c(), metal.bufType(), cpu.c(), cpu.bufType(), 2, C.int(graphSize))
	if p == nil {
		return nil
	}
	return &Sched{ptr: unsafe.Pointer(p)}
}

func (s *Sched) AllocGraph(g *Graph) bool {
	return C.ggml_go_sched_alloc_graph(cSched(s.ptr), cGraph(g.ptr)) != 0
}

func (s *Sched) Compute(g *Graph) int {
	return int(C.ggml_go_sched_compute(cSched(s.ptr), cGraph(g.ptr)))
}

func (s *Sched) c() C.ggml_go_sched { return C.ggml_go_sched(s.ptr) }

func (s *Sched) Free() {
	if s.ptr != nil {
		C.ggml_go_sched_free(cSched(s.ptr))
		s.ptr = nil
	}
}

func (s *Sched) Reset() {
	C.ggml_go_sched_reset(cSched(s.ptr))
}

// --- Tensor I/O ---

func TensorSet(t Tensor, data unsafe.Pointer, offset, size int) {
	C.ggml_go_tensor_set(t.c(), data, C.size_t(offset), C.size_t(size))
}

func TensorGet(t Tensor, data unsafe.Pointer, offset, size int) {
	C.ggml_go_tensor_get(t.c(), data, C.size_t(offset), C.size_t(size))
}

func TensorSetBytes(t Tensor, data []byte, offset int) {
	C.ggml_go_tensor_set(t.c(), unsafe.Pointer(&data[0]), C.size_t(offset), C.size_t(len(data)))
}

func TensorGetBytes(t Tensor, offset, size int) []byte {
	buf := make([]byte, size)
	C.ggml_go_tensor_get(t.c(), unsafe.Pointer(&buf[0]), C.size_t(offset), C.size_t(size))
	return buf
}

// --- Context iteration ---

func GetFirstTensor(ctx *GraphContext) Tensor {
	return Tensor{ptr: unsafe.Pointer(C.ggml_go_get_first_tensor(ctx.c()))}
}

func GetNextTensor(ctx *GraphContext, t Tensor) Tensor {
	return Tensor{ptr: unsafe.Pointer(C.ggml_go_get_next_tensor(ctx.c(), t.c()))}
}

// TensorName returns the name of a tensor.
func TensorName(t Tensor) string {
	return C.GoString(C.ggml_go_tensor_name(t.c()))
}
