package ggml

/*
#cgo CFLAGS: -std=c17 -I${SRCDIR}/../../ggml_lib/src
#cgo LDFLAGS: ${SRCDIR}/../../ggml_lib/build/lib/libggml_lib.a
#cgo LDFLAGS: -L${SRCDIR}/../../third_party/ggml/build/src -L${SRCDIR}/../../third_party/ggml/build/src/ggml-metal
#cgo LDFLAGS: -lggml-metal -lggml-cpu -lggml-base -lggml
#cgo LDFLAGS: -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders -framework Accelerate -lstdc++
#include "ggml_ops.h"
*/
import "C"
import "unsafe"

// CGO cast helpers — convert between unsafe.Pointer and CGO's typed void* aliases.
func cTensor(p unsafe.Pointer) C.ggml_go_tensor   { return C.ggml_go_tensor(p) }
func cCtx(p unsafe.Pointer) C.ggml_go_context     { return C.ggml_go_context(p) }
func cGraph(p unsafe.Pointer) C.ggml_go_graph     { return C.ggml_go_graph(p) }
func cBackend(p unsafe.Pointer) C.ggml_go_backend  { return C.ggml_go_backend(p) }
func cBuffer(p unsafe.Pointer) C.ggml_go_buffer   { return C.ggml_go_buffer(p) }
func cBufType(p unsafe.Pointer) C.ggml_go_buf_type { return C.ggml_go_buf_type(p) }
func cSched(p unsafe.Pointer) C.ggml_go_sched     { return C.ggml_go_sched(p) }
func goPtr(p C.ggml_go_tensor) unsafe.Pointer      { return unsafe.Pointer(p) }

// Tensor is an opaque handle to a ggml_tensor in a ggml context.
type Tensor struct {
	ptr unsafe.Pointer
}

// NilTensor returns a zero-value Tensor (maps to NULL).
func NilTensor() Tensor { return Tensor{} }

// IsNil returns true if the tensor pointer is NULL.
func (t Tensor) IsNil() bool { return t.ptr == nil }

// c returns the CGO-typed pointer for passing to C functions.
func (t Tensor) c() C.ggml_go_tensor { return cTensor(t.ptr) }

// Ne returns the size of dimension dim (0-3).
func (t Tensor) Ne(dim int) int64 {
	return int64(C.ggml_go_ne(t.c(), C.int(dim)))
}

// Nb returns the stride in bytes of dimension dim (0-3).
func (t Tensor) Nb(dim int) int {
	return int(C.ggml_go_nb(t.c(), C.int(dim)))
}

// ElementSize returns the size of one element in bytes.
func (t Tensor) ElementSize() int {
	return int(C.ggml_go_element_size(t.c()))
}

// Nbytes returns the total size of the tensor data in bytes.
func (t Tensor) Nbytes() int {
	return int(C.ggml_go_nbytes(t.c()))
}

// Type constants matching ggml_type enum.
const (
	TypeF32  = int(C.GGML_GO_TYPE_F32)
	TypeF16  = int(C.GGML_GO_TYPE_F16)
	TypeI32  = int(C.GGML_GO_TYPE_I32)
	TypeQ4_0 = 2  // GGML_TYPE_Q4_0
	TypeQ4_K = 12 // GGML_TYPE_Q4_K
	TypeQ6_K = 14 // GGML_TYPE_Q6_K
)

// TensorType returns the ggml_type of a tensor.
func TensorType(t Tensor) int {
	return int(C.ggml_go_tensor_type(t.c()))
}

// RoPE mode constants.
const (
	RopeNeoX = int(C.GGML_GO_ROPE_NEOX)
)

// Status constants.
const (
	StatusSuccess = int(C.GGML_GO_STATUS_SUCCESS)
)

// RowSize returns the byte size of a row of ne elements of the given type.
func RowSize(typ int, ne int64) int {
	return int(C.ggml_go_row_size(C.int(typ), C.int64_t(ne)))
}

// TensorOverhead returns the byte overhead per tensor in a ggml context.
func TensorOverhead() int {
	return int(C.ggml_go_tensor_overhead())
}
