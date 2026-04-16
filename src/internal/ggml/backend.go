package ggml

/*
#cgo CFLAGS: -std=c17 -I${SRCDIR}/../../ggml_lib/src
#include "ggml_ops.h"
#include <stdlib.h>
*/
import "C"
import (
	"unsafe"

	log "inference-lab-bench/internal/log"
)

// AllocPerm controls whether a ggml context allows its backing buffer to
// hold tensor *data* (AllocPermAllow) or only tensor *descriptors*
// (AllocPermDisallow — the normal graph-building case). Named type rather
// than a plain bool to catch callers that swap it with memSize.
type AllocPerm uint8

const AllocPermAllow = AllocPerm(0)
const AllocPermDisallow = AllocPerm(1)

// GraphContext wraps a ggml_context used for building computation graphs.
type GraphContext struct {
	ptr       unsafe.Pointer
	memSize   int
	allocPerm AllocPerm
}

// NewGraphContext creates a ggml context of the given arena size. allocPerm
// is required — callers must pick AllocPermDisallow (graph-build contexts
// that only hold tensor descriptors; the normal case) or AllocPermAllow
// (scratch contexts that hold tensor data, e.g. load-time type conversion).
func NewGraphContext(memSize int, allocPerm AllocPerm) *GraphContext {
	p := C.ggml_go_init(C.size_t(memSize), C.int(allocPerm))
	if p == nil {
		return nil
	}
	return &GraphContext{ptr: unsafe.Pointer(p), memSize: memSize, allocPerm: allocPerm}
}

func (gc *GraphContext) Free() {
	if gc.ptr != nil {
		C.ggml_go_free(cCtx(gc.ptr))
		gc.ptr = nil
	}
}

func (gc *GraphContext) Reset() {
	if gc == nil || gc.ptr == nil {
		log.Error("GraphContext.Reset: nil receiver")
		return
	}
	C.ggml_go_free(cCtx(gc.ptr))
	gc.ptr = unsafe.Pointer(C.ggml_go_init(C.size_t(gc.memSize), C.int(gc.allocPerm)))
}

// Rewind resets the ggml arena's bump-pointer allocator without freeing the
// backing memory buffer. After Rewind, all tensor descriptors previously
// allocated from this context are invalid — subsequent NewTensor calls will
// reuse the arena from offset 0. Intended for reuse of an AllocPermAllow
// scratch context across many independent small graphs (e.g. per-tensor
// load-time type conversions), avoiding the malloc/free churn of Reset().
func (gc *GraphContext) Rewind() {
	if gc == nil || gc.ptr == nil {
		log.Error("GraphContext.Rewind: nil receiver")
		return
	}
	C.ggml_go_reset(cCtx(gc.ptr))
}

// UsedMem returns the number of bytes currently allocated from the ggml
// arena (sum of tensor descriptors + tensor data, in AllocPermAllow mode).
// Useful for sanity-checking scratch arena sizing.
func (gc *GraphContext) UsedMem() int {
	if gc == nil || gc.ptr == nil {
		return 0
	}
	return int(C.ggml_go_used_mem(cCtx(gc.ptr)))
}

func (gc *GraphContext) c() C.ggml_go_context { return cCtx(gc.ptr) }

// GraphOverheadCustom returns the exact number of bytes ggml requires for
// a cgraph structure (nodes array, leafs array, hash tables) of the given
// size. Thin wrapper over ggml_graph_overhead_custom.
func GraphOverheadCustom(size int, grads bool) int {
	g := C.int(0)
	if grads {
		g = 1
	}
	return int(C.ggml_go_graph_overhead_custom(C.int(size), g))
}

// GraphContextSize returns the minimum ggml context arena size needed to
// build a graph of up to maxNodes nodes, computed from ggml's own
// accounting: cgraph overhead + one tensor descriptor per potential node +
// a small alignment margin. Use this with NewGraphContext for graph
// contexts sized precisely to the declared maxNodes budget.
func GraphContextSize(maxNodes int) int {
	return GraphOverheadCustom(maxNodes, false) +
		TensorOverhead()*maxNodes +
		64 // alignment slop
}

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
	if g == nil || g.ptr == nil {
		log.Error("BuildForwardExpand: nil graph")
		return
	}
	if t.IsNil() {
		log.Error("BuildForwardExpand: nil tensor")
		return
	}
	C.ggml_go_build_forward_expand(cGraph(g.ptr), t.c())
}

// Backend wraps a ggml backend (GPU or CPU).
type Backend struct{ ptr unsafe.Pointer }

func GPUInit() *Backend {
	// TODO: select correct back end for platform.
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

func (b *Backend) c() C.ggml_go_backend        { return cBackend(b.ptr) }
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
	if buf == nil || buf.ptr == nil {
		log.Error("Buffer.Size: nil buffer")
		return 0
	}
	return int(C.ggml_go_buffer_size(cBuffer(buf.ptr)))
}

// Clear writes value to every byte in the buffer (including alignment padding).
func (buf *Buffer) Clear(value byte) {
	if buf == nil || buf.ptr == nil {
		log.Error("Buffer.Clear: nil buffer")
		return
	}
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
	if s == nil || s.ptr == nil {
		log.Error("Sched.AllocGraph: nil scheduler")
		return false
	}
	if g == nil || g.ptr == nil {
		log.Error("Sched.AllocGraph: nil graph")
		return false
	}
	return C.ggml_go_sched_alloc_graph(cSched(s.ptr), cGraph(g.ptr)) != 0
}

func (s *Sched) Compute(g *Graph) int {
	if s == nil || s.ptr == nil {
		log.Error("Sched.Compute: nil scheduler")
		return 0
	}
	if g == nil || g.ptr == nil {
		log.Error("Sched.Compute: nil graph")
		return 0
	}
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
	if s == nil || s.ptr == nil {
		log.Error("Sched.Reset: nil scheduler")
		return
	}
	C.ggml_go_sched_reset(cSched(s.ptr))
}

// --- Tensor I/O ---

func TensorSet(t Tensor, data unsafe.Pointer, offset, size int) {
	if t.IsNil() {
		log.Error("TensorSet: nil tensor")
		return
	}
	C.ggml_go_tensor_set(t.c(), data, C.size_t(offset), C.size_t(size))
}

// TensorHasBuffer reports whether the tensor currently has a backing
// backend buffer. A scheduler that builds a graph only allocates buffers
// for tensors referenced somewhere in that graph — an input tensor that
// ended up unreferenced (e.g. inpPos in a single recurrent_ssm layer
// graph) returns false here. Callers can use this to skip TensorSet
// calls on inputs the scheduler decided were dead.
func TensorHasBuffer(t Tensor) bool {
	if t.IsNil() {
		return false
	}
	return C.ggml_go_tensor_has_buffer(t.c()) != 0
}

func TensorGet(t Tensor, data unsafe.Pointer, offset, size int) {
	if t.IsNil() {
		log.Error("TensorGet: nil tensor")
		return
	}
	C.ggml_go_tensor_get(t.c(), data, C.size_t(offset), C.size_t(size))
}

func TensorSetBytes(t Tensor, data []byte, offset int) {
	if t.IsNil() {
		log.Error("TensorSetBytes: nil target tensor")
		return
	}
	if len(data) == 0 {
		log.Error("TensorSetBytes: empty data slice (offset=%d)", offset)
		return
	}
	C.ggml_go_tensor_set(t.c(), unsafe.Pointer(&data[0]), C.size_t(offset), C.size_t(len(data)))
}

func TensorGetBytes(t Tensor, offset, size int) []byte {
	if t.IsNil() {
		log.Error("TensorGetBytes: nil tensor")
		return nil
	}
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
