package ggml

/*
#include "ggml_ops.h"
*/
import "C"
import (
	"strings"
	"unsafe"

	ilog "inference-lab-bench/internal/log"
)

// InitLogging registers the ggml log callback. Call once after log.InitLogger,
// before any model loading. ggml log output is routed to ilog.Debug or ilog.Warn/Error.
func InitLogging() {
	// this function is defined in ggml_lib (C wrapper library)
	C.ggml_go_register_log_callback()
}

// ggmlGoLogCallback is the C-callable callback registered via ggml_log_set.
// !HAZARD! the name must match the forward declared function in ggml_ops.c
// It is called by ggml for every log message it produces.
//
//export ggmlGoLogCallback
func ggmlGoLogCallback(level C.int, text *C.char, _ unsafe.Pointer) {
	msg := strings.TrimRight(C.GoString(text), "\n\r ")
	if msg == "" {
		return
	}
	// ggml_log_level: NONE=0, DEBUG=1, INFO=2, WARN=3, ERROR=4, CONT=5
	switch int(level) {
	case 3: // GGML_LOG_LEVEL_WARN
		ilog.Warn("[ggml] %s", msg)
	case 4: // GGML_LOG_LEVEL_ERROR
		ilog.Error("[ggml] %s", msg)
	default: // DEBUG=1, INFO=2, CONT=5, NONE=0
		ilog.Debug("[ggml] %s", msg)
	}
}
