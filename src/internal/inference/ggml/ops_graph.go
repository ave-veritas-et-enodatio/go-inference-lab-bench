package ggml

/*
#cgo CFLAGS: -std=c17 -I${SRCDIR}/../../../ggml_lib/src
#include "ggml_ops.h"
#include <stdlib.h>
*/
import "C"
import "unsafe"

// t wraps a C return into a Tensor.
func t(p C.ggml_go_tensor) Tensor { return Tensor{ptr: unsafe.Pointer(p)} }

// --- Tensor creation ---

func NewTensor1D(ctx *GraphContext, typ int, ne0 int64) Tensor {
	return t(C.ggml_go_new_tensor_1d(ctx.c(), C.int(typ), C.int64_t(ne0)))
}
func NewTensor2D(ctx *GraphContext, typ int, ne0, ne1 int64) Tensor {
	return t(C.ggml_go_new_tensor_2d(ctx.c(), C.int(typ), C.int64_t(ne0), C.int64_t(ne1)))
}
func NewTensor3D(ctx *GraphContext, typ int, ne0, ne1, ne2 int64) Tensor {
	return t(C.ggml_go_new_tensor_3d(ctx.c(), C.int(typ), C.int64_t(ne0), C.int64_t(ne1), C.int64_t(ne2)))
}
func NewTensor4D(ctx *GraphContext, typ int, ne0, ne1, ne2, ne3 int64) Tensor {
	return t(C.ggml_go_new_tensor_4d(ctx.c(), C.int(typ), C.int64_t(ne0), C.int64_t(ne1), C.int64_t(ne2), C.int64_t(ne3)))
}

// --- Views ---

func View2D(ctx *GraphContext, a Tensor, ne0, ne1 int64, nb1, offset int) Tensor {
	return t(C.ggml_go_view_2d(ctx.c(), a.c(), C.int64_t(ne0), C.int64_t(ne1), C.size_t(nb1), C.size_t(offset)))
}
func View3D(ctx *GraphContext, a Tensor, ne0, ne1, ne2 int64, nb1, nb2, offset int) Tensor {
	return t(C.ggml_go_view_3d(ctx.c(), a.c(), C.int64_t(ne0), C.int64_t(ne1), C.int64_t(ne2), C.size_t(nb1), C.size_t(nb2), C.size_t(offset)))
}
func View4D(ctx *GraphContext, a Tensor, ne0, ne1, ne2, ne3 int64, nb1, nb2, nb3, offset int) Tensor {
	return t(C.ggml_go_view_4d(ctx.c(), a.c(), C.int64_t(ne0), C.int64_t(ne1), C.int64_t(ne2), C.int64_t(ne3), C.size_t(nb1), C.size_t(nb2), C.size_t(nb3), C.size_t(offset)))
}

// --- Reshape ---

func Reshape2D(ctx *GraphContext, a Tensor, ne0, ne1 int64) Tensor {
	return t(C.ggml_go_reshape_2d(ctx.c(), a.c(), C.int64_t(ne0), C.int64_t(ne1)))
}
func Reshape3D(ctx *GraphContext, a Tensor, ne0, ne1, ne2 int64) Tensor {
	return t(C.ggml_go_reshape_3d(ctx.c(), a.c(), C.int64_t(ne0), C.int64_t(ne1), C.int64_t(ne2)))
}
func Reshape4D(ctx *GraphContext, a Tensor, ne0, ne1, ne2, ne3 int64) Tensor {
	return t(C.ggml_go_reshape_4d(ctx.c(), a.c(), C.int64_t(ne0), C.int64_t(ne1), C.int64_t(ne2), C.int64_t(ne3)))
}

// --- Layout ops ---

func Permute(ctx *GraphContext, a Tensor, ax0, ax1, ax2, ax3 int) Tensor {
	return t(C.ggml_go_permute(ctx.c(), a.c(), C.int(ax0), C.int(ax1), C.int(ax2), C.int(ax3)))
}
func Transpose(ctx *GraphContext, a Tensor) Tensor {
	return t(C.ggml_go_transpose(ctx.c(), a.c()))
}
func Cont(ctx *GraphContext, a Tensor) Tensor {
	return t(C.ggml_go_cont(ctx.c(), a.c()))
}
func Cont2D(ctx *GraphContext, a Tensor, ne0, ne1 int64) Tensor {
	return t(C.ggml_go_cont_2d(ctx.c(), a.c(), C.int64_t(ne0), C.int64_t(ne1)))
}
func Concat(ctx *GraphContext, a, b Tensor, dim int) Tensor {
	return t(C.ggml_go_concat(ctx.c(), a.c(), b.c(), C.int(dim)))
}
func Repeat4D(ctx *GraphContext, a Tensor, ne0, ne1, ne2, ne3 int64) Tensor {
	return t(C.ggml_go_repeat_4d(ctx.c(), a.c(), C.int64_t(ne0), C.int64_t(ne1), C.int64_t(ne2), C.int64_t(ne3)))
}

// --- Arithmetic ---

func Add(ctx *GraphContext, a, b Tensor) Tensor    { return t(C.ggml_go_add(ctx.c(), a.c(), b.c())) }
func Mul(ctx *GraphContext, a, b Tensor) Tensor    { return t(C.ggml_go_mul(ctx.c(), a.c(), b.c())) }
func Div(ctx *GraphContext, a, b Tensor) Tensor    { return t(C.ggml_go_div(ctx.c(), a.c(), b.c())) }
func Scale(ctx *GraphContext, a Tensor, s float32) Tensor {
	return t(C.ggml_go_scale(ctx.c(), a.c(), C.float(s)))
}
func Clamp(ctx *GraphContext, a Tensor, minVal, maxVal float32) Tensor {
	return t(C.ggml_go_clamp(ctx.c(), a.c(), C.float(minVal), C.float(maxVal)))
}
func SumRows(ctx *GraphContext, a Tensor) Tensor   { return t(C.ggml_go_sum_rows(ctx.c(), a.c())) }
func MulMat(ctx *GraphContext, a, b Tensor) Tensor { return t(C.ggml_go_mul_mat(ctx.c(), a.c(), b.c())) }
func MulMatId(ctx *GraphContext, as, b, ids Tensor) Tensor {
	return t(C.ggml_go_mul_mat_id(ctx.c(), as.c(), b.c(), ids.c()))
}

// --- Normalization ---

func RmsNorm(ctx *GraphContext, a Tensor, eps float32) Tensor {
	return t(C.ggml_go_rms_norm(ctx.c(), a.c(), C.float(eps)))
}
func L2Norm(ctx *GraphContext, a Tensor, eps float32) Tensor {
	return t(C.ggml_go_l2_norm(ctx.c(), a.c(), C.float(eps)))
}

// --- Activations ---

func Silu(ctx *GraphContext, a Tensor) Tensor     { return t(C.ggml_go_silu(ctx.c(), a.c())) }
func Sigmoid(ctx *GraphContext, a Tensor) Tensor  { return t(C.ggml_go_sigmoid(ctx.c(), a.c())) }
func Softplus(ctx *GraphContext, a Tensor) Tensor { return t(C.ggml_go_softplus(ctx.c(), a.c())) }
func SoftMaxExt(ctx *GraphContext, a, mask Tensor, scale, maxBias float32) Tensor {
	return t(C.ggml_go_soft_max_ext(ctx.c(), a.c(), mask.c(), C.float(scale), C.float(maxBias)))
}

// --- Embedding / indexing ---

func GetRows(ctx *GraphContext, a, b Tensor) Tensor {
	return t(C.ggml_go_get_rows(ctx.c(), a.c(), b.c()))
}

// --- Sorting / selection ---

func ArgsortTopK(ctx *GraphContext, a Tensor, k int) Tensor {
	return t(C.ggml_go_argsort_top_k(ctx.c(), a.c(), C.int(k)))
}

// --- RoPE ---

func RopeExt(ctx *GraphContext, a, pos, freqFactors Tensor,
	nDims, mode, nCtxOrig int,
	freqBase, freqScale, extFactor, attnFactor, betaFast, betaSlow float32) Tensor {
	return t(C.ggml_go_rope_ext(ctx.c(), a.c(), pos.c(), freqFactors.c(),
		C.int(nDims), C.int(mode), C.int(nCtxOrig),
		C.float(freqBase), C.float(freqScale), C.float(extFactor), C.float(attnFactor),
		C.float(betaFast), C.float(betaSlow)))
}

func RopeMulti(ctx *GraphContext, a, pos, freqFactors Tensor,
	nDims int, sections [4]int, mode, nCtxOrig int,
	freqBase, freqScale, extFactor, attnFactor, betaFast, betaSlow float32) Tensor {
	return t(C.ggml_go_rope_multi(ctx.c(), a.c(), pos.c(), freqFactors.c(),
		C.int(nDims), C.int(sections[0]), C.int(sections[1]), C.int(sections[2]), C.int(sections[3]),
		C.int(mode), C.int(nCtxOrig),
		C.float(freqBase), C.float(freqScale), C.float(extFactor), C.float(attnFactor),
		C.float(betaFast), C.float(betaSlow)))
}

// --- SSM / delta-net ---

func SSMConv(ctx *GraphContext, sx, c Tensor) Tensor {
	return t(C.ggml_go_ssm_conv(ctx.c(), sx.c(), c.c()))
}
func GatedDeltaNet(ctx *GraphContext, q, k, v, g, beta, state Tensor) Tensor {
	return t(C.ggml_go_gated_delta_net(ctx.c(), q.c(), k.c(), v.c(), g.c(), beta.c(), state.c()))
}

// --- Tensor flags ---

func SetInput(t Tensor)  { C.ggml_go_set_input(t.c()) }
func SetOutput(t Tensor) { C.ggml_go_set_output(t.c()) }
func SetName(tn Tensor, name string) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	C.ggml_go_set_name(tn.c(), cname)
}
