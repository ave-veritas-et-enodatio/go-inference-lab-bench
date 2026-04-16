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
import (
	"inference-lab-bench/internal/log"
	"unsafe"
)

// CGO cast helpers — convert between unsafe.Pointer and CGO's typed void* aliases.
func cTensor(p unsafe.Pointer) C.ggml_go_tensor    { return C.ggml_go_tensor(p) }
func cCtx(p unsafe.Pointer) C.ggml_go_context      { return C.ggml_go_context(p) }
func cGraph(p unsafe.Pointer) C.ggml_go_graph      { return C.ggml_go_graph(p) }
func cBackend(p unsafe.Pointer) C.ggml_go_backend  { return C.ggml_go_backend(p) }
func cBuffer(p unsafe.Pointer) C.ggml_go_buffer    { return C.ggml_go_buffer(p) }
func cBufType(p unsafe.Pointer) C.ggml_go_buf_type { return C.ggml_go_buf_type(p) }
func cSched(p unsafe.Pointer) C.ggml_go_sched      { return C.ggml_go_sched(p) }
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

// Ne returns the size of dimension dim (0-3). Returns 0 for a nil tensor.
func (t Tensor) Ne(dim int) int64 {
	if t.ptr == nil {
		log.Error("Tensor.Ne(%d): called on nil tensor — returning 0", dim)
		return 0
	}
	return int64(C.ggml_go_ne(t.c(), C.int(dim)))
}

// Nb returns the stride in bytes of dimension dim (0-3). Returns 0 for a nil tensor.
func (t Tensor) Nb(dim int) int {
	if t.ptr == nil {
		log.Error("Tensor.Nb(%d): called on nil tensor — returning 0", dim)
		return 0
	}
	return int(C.ggml_go_nb(t.c(), C.int(dim)))
}

// ElementSize returns the size of one element in bytes. Returns 0 for a nil tensor.
func (t Tensor) ElementSize() int {
	if t.ptr == nil {
		log.Error("Tensor.ElementSize: called on nil tensor — returning 0")
		return 0
	}
	return int(C.ggml_go_element_size(t.c()))
}

// Nbytes returns the total size of the tensor data in bytes. Returns 0 for a nil tensor.
func (t Tensor) Nbytes() int {
	if t.ptr == nil {
		log.Error("Tensor.Nbytes: called on nil tensor — returning 0")
		return 0
	}
	return int(C.ggml_go_nbytes(t.c()))
}

// GGMLType is the ggml_type enum. The underlying value is the C ggml_type
// integer; the named Go type exists so that tensor-type arguments cannot be
// accidentally swapped with unrelated ints (ne dimensions, mode flags, etc.).
type GGMLType int

// Type constants matching the ggml_type enum.
const (
	TypeF32  = GGMLType(C.GGML_GO_TYPE_F32)
	TypeF16  = GGMLType(C.GGML_GO_TYPE_F16)
	TypeI32  = GGMLType(C.GGML_GO_TYPE_I32)
	TypeBF16 = GGMLType(C.GGML_GO_TYPE_BF16)
	TypeQ4_0 = GGMLType(C.GGML_GO_TYPE_Q4_0)
	TypeQ4_K = GGMLType(C.GGML_GO_TYPE_Q4_K)
	TypeQ6_K = GGMLType(C.GGML_GO_TYPE_Q6_K)
)

// TensorType returns the ggml_type of a tensor. Returns 0 for a nil tensor.
func TensorType(t Tensor) GGMLType {
	if t.ptr == nil {
		log.Error("TensorType: called on nil tensor — returning 0")
		return 0
	}
	return GGMLType(C.ggml_go_tensor_type(t.c()))
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
func RowSize(typ GGMLType, ne int64) int {
	return int(C.ggml_go_row_size(C.int(typ), C.int64_t(ne)))
}

// ValidateRowData scans raw tensor bytes for NaN/Inf values, treating them as
// the given ggml type. For quantized types (Q4_K, Q6_K, …) only block
// scale/delta fields are inspected — near-zero cost. For float types every
// element is scanned.
//
// Intended as a one-shot sanity check at weight-load time: catches silently
// corrupted model files (a single NaN in a Q4_K scale propagates to every
// output row the block participates in) and turns them into a loud load-time
// error. The underlying ggml_validate_row_data prints a stderr line
// identifying the first offending block on failure.
//
// Returns true if all inspected values are finite. An empty slice is valid.
func ValidateRowData(typ GGMLType, data []byte) bool {
	if len(data) == 0 {
		return true
	}
	return C.ggml_go_validate_row_data(
		C.int(typ),
		unsafe.Pointer(&data[0]),
		C.size_t(len(data)),
	) != 0
}

// TensorOverhead returns the byte overhead per tensor in a ggml context.
func TensorOverhead() int {
	return int(C.ggml_go_tensor_overhead())
}
