package ggml

/*
#cgo CFLAGS: -std=c17 -I${SRCDIR}/../../ggml_lib/src
#include "ggml_ops.h"
#include <stdlib.h>
*/
import "C"
import "inference-lab-bench/internal/log"

type MemoryStat struct {
	Allocated int64
	Total     int64
}

func (ms* MemoryStat) Available() int64 {
	return ms.Total - ms.Allocated
}

func (ms *MemoryStat) AllocatedPct() float32 {
	if ms.Total > 0 {
		return float32(ms.Allocated) / float32(ms.Total) * 100.0
	}
	return 0.0
}

func (ms *MemoryStat) AvailablePct() float32 {
	if ms.Total > 0 {
		return 100.0 - ms.AllocatedPct()
	}
	return 0.0
}

// MemoryStats represents VRAM/RAM usage from GGML backends.
type MemoryStats struct {
	VRAM        MemoryStat
	RAM         MemoryStat
	IsUMA       bool  // true if integrated GPU (UMA = shared RAM)
}

func DevMemory(gpu *Backend, cpu *Backend) MemoryStats {
	nonNull := gpu
	if nonNull == nil {
		nonNull = cpu
	}
	return MemoryStats {
		VRAM: BackendMemory(gpu),
		RAM: BackendMemory(cpu),
		IsUMA: IsUMA(nonNull),
	}
}

func BackendMemory(backend *Backend) MemoryStat {
	if backend == nil || backend.ptr == nil {
		return MemoryStat{}
	}

	var free C.size_t
	var total C.size_t

	C.ggml_go_dev_memory(backendPtr(backend), &free, &total)

	return MemoryStat{Allocated: int64(total - free), Total: int64(total)}
}

func BackendBufferStats(sched *Sched, backend *Backend) int64 {
	if sched == nil || sched.ptr == nil || backend == nil || backend.ptr == nil {
		log.Error("BackendBufferStats(): invalid Sched or Backend")
		return 0
	}
	return int64(C.ggml_go_backend_sched_get_buffer_size(sched.c(), backendPtr(backend)))
}

// IsUMA reports whether the backend uses unified memory (CPU and GPU share the same physical RAM).
// Currently implemented as a Metal check — on Apple Silicon, Metal implies UMA.
// TODO(portability): when non-Metal GPU backends (CUDA, ROCm, Vulkan) are added, replace with
// a backend-capability query rather than a Metal-specific probe.
func IsUMA(backend *Backend) bool {
	if backend == nil || backend.ptr == nil {
		log.Error("IsUMA(): invalid Backend")
		return false
	}
	return C.ggml_go_backend_is_metal(backendPtr(backend)) != 0
}
