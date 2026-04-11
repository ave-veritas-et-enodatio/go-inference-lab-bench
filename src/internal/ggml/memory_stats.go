package ggml

/*
#cgo CFLAGS: -std=c17 -I${SRCDIR}/../../ggml_lib/src
#include "ggml_ops.h"
#include <stdlib.h>
*/
import "C"

type MemoryStat struct {
	Allocated int64
	Total     int64
}

// MemoryStats represents VRAM/RAM usage from GGML backends.
type MemoryStats struct {
	VRAM        MemoryStat
	RAM         MemoryStat
	IsUMA       bool  // true if integrated GPU (UMA = shared RAM)
}

func DevMemory(backend *Backend) int64 {
	if backend == nil || backend.ptr == nil {
		return 0
	}

	var free C.size_t
	var total C.size_t

	C.ggml_go_dev_memory(backendPtr(backend), &free, &total)

	return int64(total)
}

func BackendBufferStats(sched *Sched, backend *Backend) int64 {
	if sched == nil || sched.ptr == nil || backend == nil || backend.ptr == nil {
		return 0
	}
	return int64(C.ggml_go_backend_sched_get_buffer_size(sched.c(), backendPtr(backend)))
}

func IsUMA(backend *Backend) bool {
	if backend == nil || backend.ptr == nil {
		return false
	}
	return C.ggml_go_backend_is_metal(backendPtr(backend)) != 0
}
