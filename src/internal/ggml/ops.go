package ggml

/*
#cgo CFLAGS: -std=c17 -I${SRCDIR}/../../ggml_lib/src
#include "ggml_ops.h"
#include <stdlib.h>
*/
import "C"
import "unsafe"

import log "inference-lab-bench/internal/log"

// anyNil reports and returns true if any Tensor argument is nil.
// ctx must also be non-nil (this function checks it).
func anyNil(fn string, ctx *GraphContext, ts ...Tensor) bool {
	if ctx == nil {
		log.Error("%s: nil GraphContext", fn)
		return true
	}
	return anyNilTensor(fn, ts...)
}

// anyNilTensor reports and returns true if any Tensor argument is nil.
// Use for functions that do not take a GraphContext.
func anyNilTensor(fn string, ts ...Tensor) bool {
	for _, t := range ts {
		if t.IsNil() {
			log.Error("%s: nil tensor argument", fn)
			return true
		}
	}
	return false
}

// t wraps a C return into a Tensor.
func t(p C.ggml_go_tensor) Tensor { return Tensor{ptr: unsafe.Pointer(p)} }

// Nil tensor parameter contract for this file:
//
// Tensor parameters named with an "opt" prefix (e.g. optMask, optFreqFactors)
// are legally nil and correspond to ggml op arguments that the underlying C
// implementation explicitly guards with a null-pointer check. Pass NilTensor()
// to opt out of that feature (e.g. unmasked softmax, no RoPE frequency
// interpolation).
//
// Every other Tensor parameter is required. Required parameters are checked
// by the anyNil() / anyNilTensor() guards at the top of each wrapper; passing
// nil for a required parameter logs an error and returns NilTensor().
//
// When adding new ops, (a) name any legally-nullable tensor parameter with
// the opt* prefix, (b) exclude it from the anyNil guard, and (c) document
// which ggml argument it corresponds to.

// --- Tensor creation ---

func NewTensor1D(ctx *GraphContext, typ GGMLType, ne0 int64) Tensor {
	if ctx == nil {
		log.Error("NewTensor1D: nil GraphContext")
		return NilTensor()
	}
	return t(C.ggml_go_new_tensor_1d(ctx.c(), C.int(typ), C.int64_t(ne0)))
}
func NewTensor2D(ctx *GraphContext, typ GGMLType, ne0, ne1 int64) Tensor {
	if ctx == nil {
		log.Error("NewTensor2D: nil GraphContext")
		return NilTensor()
	}
	return t(C.ggml_go_new_tensor_2d(ctx.c(), C.int(typ), C.int64_t(ne0), C.int64_t(ne1)))
}
func NewTensor3D(ctx *GraphContext, typ GGMLType, ne0, ne1, ne2 int64) Tensor {
	if ctx == nil {
		log.Error("NewTensor3D: nil GraphContext")
		return NilTensor()
	}
	return t(C.ggml_go_new_tensor_3d(ctx.c(), C.int(typ), C.int64_t(ne0), C.int64_t(ne1), C.int64_t(ne2)))
}
func NewTensor4D(ctx *GraphContext, typ GGMLType, ne0, ne1, ne2, ne3 int64) Tensor {
	if ctx == nil {
		log.Error("NewTensor4D: nil GraphContext")
		return NilTensor()
	}
	return t(C.ggml_go_new_tensor_4d(ctx.c(), C.int(typ), C.int64_t(ne0), C.int64_t(ne1), C.int64_t(ne2), C.int64_t(ne3)))
}

// --- Views ---

func View2D(ctx *GraphContext, a Tensor, ne0, ne1 int64, nb1, offset int) Tensor {
	if anyNil("View2D", ctx, a) {
		return NilTensor()
	}
	return t(C.ggml_go_view_2d(ctx.c(), a.c(), C.int64_t(ne0), C.int64_t(ne1), C.size_t(nb1), C.size_t(offset)))
}
func View3D(ctx *GraphContext, a Tensor, ne0, ne1, ne2 int64, nb1, nb2, offset int) Tensor {
	if anyNil("View3D", ctx, a) {
		return NilTensor()
	}
	return t(C.ggml_go_view_3d(ctx.c(), a.c(), C.int64_t(ne0), C.int64_t(ne1), C.int64_t(ne2), C.size_t(nb1), C.size_t(nb2), C.size_t(offset)))
}
func View4D(ctx *GraphContext, a Tensor, ne0, ne1, ne2, ne3 int64, nb1, nb2, nb3, offset int) Tensor {
	if anyNil("View4D", ctx, a) {
		return NilTensor()
	}
	return t(C.ggml_go_view_4d(ctx.c(), a.c(), C.int64_t(ne0), C.int64_t(ne1), C.int64_t(ne2), C.int64_t(ne3), C.size_t(nb1), C.size_t(nb2), C.size_t(nb3), C.size_t(offset)))
}

// --- Reshape ---

func Reshape2D(ctx *GraphContext, a Tensor, ne0, ne1 int64) Tensor {
	if anyNil("Reshape2D", ctx, a) {
		return NilTensor()
	}
	return t(C.ggml_go_reshape_2d(ctx.c(), a.c(), C.int64_t(ne0), C.int64_t(ne1)))
}
func Reshape3D(ctx *GraphContext, a Tensor, ne0, ne1, ne2 int64) Tensor {
	if anyNil("Reshape3D", ctx, a) {
		return NilTensor()
	}
	return t(C.ggml_go_reshape_3d(ctx.c(), a.c(), C.int64_t(ne0), C.int64_t(ne1), C.int64_t(ne2)))
}
func Reshape4D(ctx *GraphContext, a Tensor, ne0, ne1, ne2, ne3 int64) Tensor {
	if anyNil("Reshape4D", ctx, a) {
		return NilTensor()
	}
	return t(C.ggml_go_reshape_4d(ctx.c(), a.c(), C.int64_t(ne0), C.int64_t(ne1), C.int64_t(ne2), C.int64_t(ne3)))
}

// --- Layout ops ---

func Cpy(ctx *GraphContext, a, b Tensor) Tensor {
	if anyNil("Cpy", ctx, a, b) {
		return NilTensor()
	}
	return t(C.ggml_go_cpy(ctx.c(), a.c(), b.c()))
}
func Permute(ctx *GraphContext, a Tensor, ax0, ax1, ax2, ax3 int) Tensor {
	if anyNil("Permute", ctx, a) {
		return NilTensor()
	}
	return t(C.ggml_go_permute(ctx.c(), a.c(), C.int(ax0), C.int(ax1), C.int(ax2), C.int(ax3)))
}
func Transpose(ctx *GraphContext, a Tensor) Tensor {
	if anyNil("Transpose", ctx, a) {
		return NilTensor()
	}
	return t(C.ggml_go_transpose(ctx.c(), a.c()))
}
func Cont(ctx *GraphContext, a Tensor) Tensor {
	if anyNil("Cont", ctx, a) {
		return NilTensor()
	}
	return t(C.ggml_go_cont(ctx.c(), a.c()))
}
func Cont2D(ctx *GraphContext, a Tensor, ne0, ne1 int64) Tensor {
	if anyNil("Cont2D", ctx, a) {
		return NilTensor()
	}
	return t(C.ggml_go_cont_2d(ctx.c(), a.c(), C.int64_t(ne0), C.int64_t(ne1)))
}
func Concat(ctx *GraphContext, a, b Tensor, dim int) Tensor {
	if anyNil("Concat", ctx, a, b) {
		return NilTensor()
	}
	return t(C.ggml_go_concat(ctx.c(), a.c(), b.c(), C.int(dim)))
}
func Repeat4D(ctx *GraphContext, a Tensor, ne0, ne1, ne2, ne3 int64) Tensor {
	if anyNil("Repeat4D", ctx, a) {
		return NilTensor()
	}
	return t(C.ggml_go_repeat_4d(ctx.c(), a.c(), C.int64_t(ne0), C.int64_t(ne1), C.int64_t(ne2), C.int64_t(ne3)))
}

// --- Arithmetic ---

func Add(ctx *GraphContext, a, b Tensor) Tensor {
	if anyNil("Add", ctx, a, b) {
		return NilTensor()
	}
	return t(C.ggml_go_add(ctx.c(), a.c(), b.c()))
}
func Mul(ctx *GraphContext, a, b Tensor) Tensor {
	if anyNil("Mul", ctx, a, b) {
		return NilTensor()
	}
	return t(C.ggml_go_mul(ctx.c(), a.c(), b.c()))
}
func Div(ctx *GraphContext, a, b Tensor) Tensor {
	if anyNil("Div", ctx, a, b) {
		return NilTensor()
	}
	return t(C.ggml_go_div(ctx.c(), a.c(), b.c()))
}
func Scale(ctx *GraphContext, a Tensor, s float32) Tensor {
	if anyNil("Scale", ctx, a) {
		return NilTensor()
	}
	return t(C.ggml_go_scale(ctx.c(), a.c(), C.float(s)))
}

// ScaleBias computes x = s*a + b element-wise (ggml_scale_bias). Useful for
// fused scale+shift transforms (e.g. add_scalar uses s=1.0).
func ScaleBias(ctx *GraphContext, a Tensor, s, b float32) Tensor {
	if anyNil("ScaleBias", ctx, a) {
		return NilTensor()
	}
	return t(C.ggml_go_scale_bias(ctx.c(), a.c(), C.float(s), C.float(b)))
}
func Clamp(ctx *GraphContext, a Tensor, minVal, maxVal float32) Tensor {
	if anyNil("Clamp", ctx, a) {
		return NilTensor()
	}
	return t(C.ggml_go_clamp(ctx.c(), a.c(), C.float(minVal), C.float(maxVal)))
}
func SumRows(ctx *GraphContext, a Tensor) Tensor {
	if anyNil("SumRows", ctx, a) {
		return NilTensor()
	}
	return t(C.ggml_go_sum_rows(ctx.c(), a.c()))
}
func Sum(ctx *GraphContext, a Tensor) Tensor {
	if anyNil("Sum", ctx, a) {
		return NilTensor()
	}
	return t(C.ggml_go_sum(ctx.c(), a.c()))
}
func Sqrt(ctx *GraphContext, a Tensor) Tensor {
	if anyNil("Sqrt", ctx, a) {
		return NilTensor()
	}
	return t(C.ggml_go_sqrt(ctx.c(), a.c()))
}
func Exp(ctx *GraphContext, a Tensor) Tensor {
	if anyNil("Exp", ctx, a) {
		return NilTensor()
	}
	return t(C.ggml_go_exp(ctx.c(), a.c()))
}
func Neg(ctx *GraphContext, a Tensor) Tensor {
	if anyNil("Neg", ctx, a) {
		return NilTensor()
	}
	return t(C.ggml_go_neg(ctx.c(), a.c()))
}
func MulMat(ctx *GraphContext, a, b Tensor) Tensor {
	if anyNil("MulMat", ctx, a, b) {
		return NilTensor()
	}
	return t(C.ggml_go_mul_mat(ctx.c(), a.c(), b.c()))
}
func MulMatId(ctx *GraphContext, as, b, ids Tensor) Tensor {
	if anyNil("MulMatId", ctx, as, b, ids) {
		return NilTensor()
	}
	return t(C.ggml_go_mul_mat_id(ctx.c(), as.c(), b.c(), ids.c()))
}

// --- Normalization ---

func RmsNorm(ctx *GraphContext, a Tensor, eps float32) Tensor {
	if anyNil("RmsNorm", ctx, a) {
		return NilTensor()
	}
	return t(C.ggml_go_rms_norm(ctx.c(), a.c(), C.float(eps)))
}
func L2Norm(ctx *GraphContext, a Tensor, eps float32) Tensor {
	if anyNil("L2Norm", ctx, a) {
		return NilTensor()
	}
	return t(C.ggml_go_l2_norm(ctx.c(), a.c(), C.float(eps)))
}

// --- Activations ---

func Silu(ctx *GraphContext, a Tensor) Tensor {
	if anyNil("Silu", ctx, a) {
		return NilTensor()
	}
	return t(C.ggml_go_silu(ctx.c(), a.c()))
}
func Sigmoid(ctx *GraphContext, a Tensor) Tensor {
	if anyNil("Sigmoid", ctx, a) {
		return NilTensor()
	}
	return t(C.ggml_go_sigmoid(ctx.c(), a.c()))
}
func Softplus(ctx *GraphContext, a Tensor) Tensor {
	if anyNil("Softplus", ctx, a) {
		return NilTensor()
	}
	return t(C.ggml_go_softplus(ctx.c(), a.c()))
}
func Gelu(ctx *GraphContext, a Tensor) Tensor {
	if anyNil("Gelu", ctx, a) {
		return NilTensor()
	}
	return t(C.ggml_go_gelu(ctx.c(), a.c()))
}
func Tanh(ctx *GraphContext, a Tensor) Tensor {
	if anyNil("Tanh", ctx, a) {
		return NilTensor()
	}
	return t(C.ggml_go_tanh(ctx.c(), a.c()))
}
// SoftMaxExt applies softmax with optional additive mask, scale, and ALiBi
// bias. optMask may legally be NilTensor (means no mask).
func SoftMaxExt(ctx *GraphContext, a, optMask Tensor, scale, maxBias float32) Tensor {
	if anyNil("SoftMaxExt", ctx, a) {
		return NilTensor()
	}
	return t(C.ggml_go_soft_max_ext(ctx.c(), a.c(), optMask.c(), C.float(scale), C.float(maxBias)))
}

// --- Embedding / indexing ---

func GetRows(ctx *GraphContext, a, b Tensor) Tensor {
	if anyNil("GetRows", ctx, a, b) {
		return NilTensor()
	}
	return t(C.ggml_go_get_rows(ctx.c(), a.c(), b.c()))
}

// --- Sorting / selection ---

func ArgsortTopK(ctx *GraphContext, a Tensor, k int) Tensor {
	if anyNil("ArgsortTopK", ctx, a) {
		return NilTensor()
	}
	return t(C.ggml_go_argsort_top_k(ctx.c(), a.c(), C.int(k)))
}

// --- RoPE ---

// RopeExt applies extended RoPE with optional per-frequency scaling factors.
// optFreqFactors may legally be NilTensor (means no frequency interpolation).
func RopeExt(ctx *GraphContext, a, pos, optFreqFactors Tensor,
	nDims, mode, nCtxOrig int,
	freqBase, freqScale, extFactor, attnFactor, betaFast, betaSlow float32) Tensor {
	if anyNil("RopeExt", ctx, a, pos) {
		return NilTensor()
	}
	return t(C.ggml_go_rope_ext(ctx.c(), a.c(), pos.c(), optFreqFactors.c(),
		C.int(nDims), C.int(mode), C.int(nCtxOrig),
		C.float(freqBase), C.float(freqScale), C.float(extFactor), C.float(attnFactor),
		C.float(betaFast), C.float(betaSlow)))
}

// RopeMulti applies multi-section RoPE with optional per-frequency scaling
// factors. optFreqFactors may legally be NilTensor (means no frequency
// interpolation).
func RopeMulti(ctx *GraphContext, a, pos, optFreqFactors Tensor,
	nDims int, sections [4]int, mode, nCtxOrig int,
	freqBase, freqScale, extFactor, attnFactor, betaFast, betaSlow float32) Tensor {
	if anyNil("RopeMulti", ctx, a, pos) {
		return NilTensor()
	}
	return t(C.ggml_go_rope_multi(ctx.c(), a.c(), pos.c(), optFreqFactors.c(),
		C.int(nDims), C.int(sections[0]), C.int(sections[1]), C.int(sections[2]), C.int(sections[3]),
		C.int(mode), C.int(nCtxOrig),
		C.float(freqBase), C.float(freqScale), C.float(extFactor), C.float(attnFactor),
		C.float(betaFast), C.float(betaSlow)))
}

// --- SSM / delta-net ---

func SSMConv(ctx *GraphContext, sx, c Tensor) Tensor {
	if anyNil("SSMConv", ctx, sx, c) {
		return NilTensor()
	}
	return t(C.ggml_go_ssm_conv(ctx.c(), sx.c(), c.c()))
}
func GatedDeltaNet(ctx *GraphContext, q, k, v, g, beta, state Tensor) Tensor {
	if anyNil("GatedDeltaNet", ctx, q, k, v, g, beta, state) {
		return NilTensor()
	}
	return t(C.ggml_go_gated_delta_net(ctx.c(), q.c(), k.c(), v.c(), g.c(), beta.c(), state.c()))
}

// --- Flash attention ---

// PrecF32 selects F32 accumulation precision for flash attention (GGML_PREC_F32).
const PrecF32 = int(C.GGML_GO_PREC_F32)

// FlashAttnExt runs fused scaled-dot-product attention with optional additive
// mask. optMask may legally be NilTensor (means unmasked attention, e.g. for
// diffusion models).
func FlashAttnExt(ctx *GraphContext, q, k, v, optMask Tensor, scale, maxBias, logitSoftcap float32) Tensor {
	if anyNil("FlashAttnExt", ctx, q, k, v) {
		return NilTensor()
	}
	return t(C.ggml_go_flash_attn_ext(ctx.c(), q.c(), k.c(), v.c(), optMask.c(),
		C.float(scale), C.float(maxBias), C.float(logitSoftcap)))
}

func FlashAttnExtSetPrec(tn Tensor, prec int) {
	if anyNilTensor("FlashAttnExtSetPrec", tn) {
		return
	}
	C.ggml_go_flash_attn_ext_set_prec(tn.c(), C.int(prec))
}

func Cast(ctx *GraphContext, a Tensor, typ GGMLType) Tensor {
	if anyNilTensor("Cast", a) {
		return NilTensor()
	}
	return t(C.ggml_go_cast(ctx.c(), a.c(), C.int(typ)))
}

// TensorData returns the raw data pointer of a tensor.
func TensorData(t Tensor) unsafe.Pointer {
	if t.ptr == nil {
		return nil
	}
	return C.ggml_go_tensor_data(t.c())
}

// TensorSetData sets the raw data pointer of a tensor (for no_alloc contexts
// where you want the tensor to point directly at external memory).
func TensorSetData(t Tensor, data unsafe.Pointer) {
	if t.ptr == nil {
		return
	}
	C.ggml_go_tensor_set_data(t.c(), data)
}

// GraphCompute runs a graph on CPU with nThreads threads (no scheduler/backend
// setup needed — suitable for small utility graphs like type conversion).
func GraphCompute(ctx *GraphContext, g *Graph, nThreads int) {
	if ctx == nil || ctx.ptr == nil {
		log.Error("GraphCompute: nil GraphContext")
		return
	}
	if g == nil || g.ptr == nil {
		log.Error("GraphCompute: nil Graph")
		return
	}
	C.ggml_go_graph_compute(ctx.c(), cGraph(g.ptr), C.int(nThreads))
}

// --- Precision ---

func MulMatSetPrecF32(t Tensor) {
	if anyNilTensor("MulMatSetPrecF32", t) {
		return
	}
	C.ggml_go_mul_mat_set_prec_f32(t.c())
}

// --- Tensor flags ---

func SetInput(t Tensor) {
	if anyNilTensor("SetInput", t) {
		return
	}
	C.ggml_go_set_input(t.c())
}
func SetOutput(t Tensor) {
	if anyNilTensor("SetOutput", t) {
		return
	}
	C.ggml_go_set_output(t.c())
}
func SetName(tn Tensor, name string) {
	if anyNilTensor("SetName", tn) {
		return
	}
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	C.ggml_go_set_name(tn.c(), cname)
}

func backendPtr(b *Backend) C.ggml_go_backend {
	if b == nil || b.ptr == nil {
		return nil
	}
	return C.ggml_go_backend(b.ptr)
}
