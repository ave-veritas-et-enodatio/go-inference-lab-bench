/* ggml_ops.c — Pure C wrappers dispatching to ggml API.
 * Each function is a thin cast+forward. No model knowledge. */

#include "ggml_ops.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include "ggml-cpu.h"

#include <string.h>
#include <assert.h>

// enforce sync of relayed constants
static_assert(GGML_GO_STATUS_SUCCESS == GGML_STATUS_SUCCESS, "GGML_GO_STATUS_SUCCESS must match GGML_STATUS_SUCCESS");
static_assert(GGML_GO_TYPE_F32 == GGML_TYPE_F32, "GGML_GO_TYPE_F32 must match GGML_TYPE_F32");
static_assert(GGML_GO_TYPE_F16 == GGML_TYPE_F16, "GGML_GO_TYPE_F16 must match GGML_TYPE_F16");
static_assert(GGML_GO_TYPE_I32 == GGML_TYPE_I32, "GGML_GO_TYPE_I32 must match GGML_TYPE_I32");
static_assert(GGML_GO_ROPE_NEOX == GGML_GO_ROPE_NEOX, "GGML_GO_ROPE_NEOX must match GGML_ROPE_NEOX");

static_assert(GGML_GO_GGUF_TYPE_UINT32  == GGUF_TYPE_UINT32, "GGML_GO_GGUF_TYPE_UINT32 must match GGUF_TYPE_UINT32");
static_assert(GGML_GO_GGUF_TYPE_INT32   == GGUF_TYPE_INT32, "GGML_GO_GGUF_TYPE_INT32 must match GGUF_TYPE_INT32");
static_assert(GGML_GO_GGUF_TYPE_FLOAT32 == GGUF_TYPE_FLOAT32, "GGML_GO_GGUF_TYPE_FLOAT32 must match GGUF_TYPE_FLOAT32");
static_assert(GGML_GO_GGUF_TYPE_BOOL    == GGUF_TYPE_BOOL, "GGML_GO_GGUF_TYPE_BOOL must match GGUF_TYPE_BOOL");
static_assert(GGML_GO_GGUF_TYPE_ARRAY   == GGUF_TYPE_ARRAY, "GGML_GO_GGUF_TYPE_ARRAY must match GGUF_TYPE_ARRAY");


/* Casting helpers */
#define CTX(x)  ((struct ggml_context*)(x))
#define T(x)    ((struct ggml_tensor*)(x))
#define G(x)    ((struct ggml_cgraph*)(x))
#define GF(x)   ((struct gguf_context*)(x))
#define BE(x)   ((ggml_backend_t)(x))
#define BUF(x)  ((ggml_backend_buffer_t)(x))
#define BT(x)   ((ggml_backend_buffer_type_t)(x))
#define SC(x)   ((ggml_backend_sched_t)(x))

/* --- Tensor creation --- */
ggml_go_tensor ggml_go_new_tensor_1d(ggml_go_context ctx, int type, int64_t ne0)                                             { return ggml_new_tensor_1d(CTX(ctx), (enum ggml_type)type, ne0); }
ggml_go_tensor ggml_go_new_tensor_2d(ggml_go_context ctx, int type, int64_t ne0, int64_t ne1)                                { return ggml_new_tensor_2d(CTX(ctx), (enum ggml_type)type, ne0, ne1); }
ggml_go_tensor ggml_go_new_tensor_3d(ggml_go_context ctx, int type, int64_t ne0, int64_t ne1, int64_t ne2)                   { return ggml_new_tensor_3d(CTX(ctx), (enum ggml_type)type, ne0, ne1, ne2); }
ggml_go_tensor ggml_go_new_tensor_4d(ggml_go_context ctx, int type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3)      { return ggml_new_tensor_4d(CTX(ctx), (enum ggml_type)type, ne0, ne1, ne2, ne3); }

/* --- Views --- */
ggml_go_tensor ggml_go_view_2d(ggml_go_context ctx, ggml_go_tensor a, int64_t ne0, int64_t ne1, size_t nb1, size_t offset)                                    { return ggml_view_2d(CTX(ctx), T(a), ne0, ne1, nb1, offset); }
ggml_go_tensor ggml_go_view_3d(ggml_go_context ctx, ggml_go_tensor a, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, size_t nb2, size_t offset)           { return ggml_view_3d(CTX(ctx), T(a), ne0, ne1, ne2, nb1, nb2, offset); }
ggml_go_tensor ggml_go_view_4d(ggml_go_context ctx, ggml_go_tensor a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1, size_t nb2, size_t nb3, size_t offset) { return ggml_view_4d(CTX(ctx), T(a), ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset); }

/* --- Reshape --- */
ggml_go_tensor ggml_go_reshape_2d(ggml_go_context ctx, ggml_go_tensor a, int64_t ne0, int64_t ne1)                           { return ggml_reshape_2d(CTX(ctx), T(a), ne0, ne1); }
ggml_go_tensor ggml_go_reshape_3d(ggml_go_context ctx, ggml_go_tensor a, int64_t ne0, int64_t ne1, int64_t ne2)              { return ggml_reshape_3d(CTX(ctx), T(a), ne0, ne1, ne2); }
ggml_go_tensor ggml_go_reshape_4d(ggml_go_context ctx, ggml_go_tensor a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) { return ggml_reshape_4d(CTX(ctx), T(a), ne0, ne1, ne2, ne3); }

/* --- Layout ops --- */
ggml_go_tensor ggml_go_permute(ggml_go_context ctx, ggml_go_tensor a, int ax0, int ax1, int ax2, int ax3) { return ggml_permute(CTX(ctx), T(a), ax0, ax1, ax2, ax3); }
ggml_go_tensor ggml_go_transpose(ggml_go_context ctx, ggml_go_tensor a)                                   { return ggml_transpose(CTX(ctx), T(a)); }
ggml_go_tensor ggml_go_cont(ggml_go_context ctx, ggml_go_tensor a)                                        { return ggml_cont(CTX(ctx), T(a)); }
ggml_go_tensor ggml_go_cont_2d(ggml_go_context ctx, ggml_go_tensor a, int64_t ne0, int64_t ne1)           { return ggml_cont_2d(CTX(ctx), T(a), ne0, ne1); }
ggml_go_tensor ggml_go_concat(ggml_go_context ctx, ggml_go_tensor a, ggml_go_tensor b, int dim)            { return ggml_concat(CTX(ctx), T(a), T(b), dim); }
ggml_go_tensor ggml_go_repeat_4d(ggml_go_context ctx, ggml_go_tensor a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) { return ggml_repeat_4d(CTX(ctx), T(a), ne0, ne1, ne2, ne3); }

/* --- Arithmetic --- */
ggml_go_tensor ggml_go_add(ggml_go_context ctx, ggml_go_tensor a, ggml_go_tensor b)        { return ggml_add(CTX(ctx), T(a), T(b)); }
ggml_go_tensor ggml_go_mul(ggml_go_context ctx, ggml_go_tensor a, ggml_go_tensor b)        { return ggml_mul(CTX(ctx), T(a), T(b)); }
ggml_go_tensor ggml_go_div(ggml_go_context ctx, ggml_go_tensor a, ggml_go_tensor b)        { return ggml_div(CTX(ctx), T(a), T(b)); }
ggml_go_tensor ggml_go_scale(ggml_go_context ctx, ggml_go_tensor a, float s)                { return ggml_scale(CTX(ctx), T(a), s); }
ggml_go_tensor ggml_go_clamp(ggml_go_context ctx, ggml_go_tensor a, float min_val, float max_val) { return ggml_clamp(CTX(ctx), T(a), min_val, max_val); }
ggml_go_tensor ggml_go_sum_rows(ggml_go_context ctx, ggml_go_tensor a)                     { return ggml_sum_rows(CTX(ctx), T(a)); }
ggml_go_tensor ggml_go_mul_mat(ggml_go_context ctx, ggml_go_tensor a, ggml_go_tensor b)    { return ggml_mul_mat(CTX(ctx), T(a), T(b)); }
ggml_go_tensor ggml_go_mul_mat_id(ggml_go_context ctx, ggml_go_tensor as, ggml_go_tensor b, ggml_go_tensor ids) { return ggml_mul_mat_id(CTX(ctx), T(as), T(b), T(ids)); }

/* --- Normalization --- */
ggml_go_tensor ggml_go_rms_norm(ggml_go_context ctx, ggml_go_tensor a, float eps) { return ggml_rms_norm(CTX(ctx), T(a), eps); }
ggml_go_tensor ggml_go_l2_norm(ggml_go_context ctx, ggml_go_tensor a, float eps)  { return ggml_l2_norm(CTX(ctx), T(a), eps); }

/* --- Activations --- */
ggml_go_tensor ggml_go_silu(ggml_go_context ctx, ggml_go_tensor a)                                                         { return ggml_silu(CTX(ctx), T(a)); }
ggml_go_tensor ggml_go_sigmoid(ggml_go_context ctx, ggml_go_tensor a)                                                      { return ggml_sigmoid(CTX(ctx), T(a)); }
ggml_go_tensor ggml_go_softplus(ggml_go_context ctx, ggml_go_tensor a)                                                     { return ggml_softplus(CTX(ctx), T(a)); }
ggml_go_tensor ggml_go_gelu(ggml_go_context ctx, ggml_go_tensor a)                                                        { return ggml_gelu(CTX(ctx), T(a)); }
ggml_go_tensor ggml_go_tanh(ggml_go_context ctx, ggml_go_tensor a)                                                        { return ggml_tanh(CTX(ctx), T(a)); }
ggml_go_tensor ggml_go_soft_max_ext(ggml_go_context ctx, ggml_go_tensor a, ggml_go_tensor mask, float scale, float max_bias) { return ggml_soft_max_ext(CTX(ctx), T(a), T(mask), scale, max_bias); }

/* --- Embedding / indexing --- */
ggml_go_tensor ggml_go_get_rows(ggml_go_context ctx, ggml_go_tensor a, ggml_go_tensor b) { return ggml_get_rows(CTX(ctx), T(a), T(b)); }

/* --- Sorting / selection --- */
ggml_go_tensor ggml_go_argsort_top_k(ggml_go_context ctx, ggml_go_tensor a, int k) { return ggml_argsort_top_k(CTX(ctx), T(a), k); }

/* --- RoPE --- */
ggml_go_tensor ggml_go_rope_ext(ggml_go_context ctx, ggml_go_tensor a, ggml_go_tensor pos, ggml_go_tensor freq_factors,
    int n_dims, int mode, int n_ctx_orig,
    float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow)
{
    return ggml_rope_ext(CTX(ctx), T(a), T(pos), T(freq_factors), n_dims, mode, n_ctx_orig,
                         freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
}

/* --- RoPE multi (sections passed as 4 ints to avoid pointer lifetime issues) --- */
ggml_go_tensor ggml_go_rope_multi(ggml_go_context ctx, ggml_go_tensor a, ggml_go_tensor pos, ggml_go_tensor freq_factors,
    int n_dims, int s0, int s1, int s2, int s3, int mode, int n_ctx_orig,
    float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow)
{
    int sections[4] = {s0, s1, s2, s3};
    return ggml_rope_multi(CTX(ctx), T(a), T(pos), T(freq_factors), n_dims, sections, mode, n_ctx_orig,
                           freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
}

/* --- SSM / delta-net --- */
ggml_go_tensor ggml_go_ssm_conv(ggml_go_context ctx, ggml_go_tensor sx, ggml_go_tensor c)                                                           { return ggml_ssm_conv(CTX(ctx), T(sx), T(c)); }
ggml_go_tensor ggml_go_gated_delta_net(ggml_go_context ctx, ggml_go_tensor q, ggml_go_tensor k, ggml_go_tensor v, ggml_go_tensor g, ggml_go_tensor beta, ggml_go_tensor state) { return ggml_gated_delta_net(CTX(ctx), T(q), T(k), T(v), T(g), T(beta), T(state)); }

/* --- Precision --- */
void ggml_go_mul_mat_set_prec_f32(ggml_go_tensor t) { ggml_mul_mat_set_prec(T(t), GGML_PREC_F32); }

/* --- Tensor flags --- */
void ggml_go_set_input(ggml_go_tensor t)                   { ggml_set_input(T(t)); }
void ggml_go_set_output(ggml_go_tensor t)                  { ggml_set_output(T(t)); }
void ggml_go_set_name(ggml_go_tensor t, const char* name)  { ggml_set_name(T(t), name); }

/* --- Tensor accessors --- */
int64_t ggml_go_ne(ggml_go_tensor t, int dim)     { return T(t)->ne[dim]; }
size_t  ggml_go_nb(ggml_go_tensor t, int dim)     { return T(t)->nb[dim]; }
size_t  ggml_go_element_size(ggml_go_tensor t)     { return ggml_element_size(T(t)); }
size_t  ggml_go_nbytes(ggml_go_tensor t)           { return ggml_nbytes(T(t)); }
size_t  ggml_go_row_size(int type, int64_t ne)     { return ggml_row_size((enum ggml_type)type, ne); }
size_t  ggml_go_tensor_overhead(void)              { return ggml_tensor_overhead(); }
int     ggml_go_tensor_type(ggml_go_tensor t)      { return (int)T(t)->type; }

/* --- Context lifecycle --- */
ggml_go_context ggml_go_init(size_t mem_size) {
    struct ggml_init_params p = { .mem_size = mem_size, .mem_buffer = NULL, .no_alloc = true };
    return ggml_init(p);
}
void ggml_go_free(ggml_go_context ctx) { ggml_free(CTX(ctx)); }

/* --- Graph --- */
ggml_go_graph ggml_go_new_graph(ggml_go_context ctx, int size)          { return ggml_new_graph_custom(CTX(ctx), size, false); }
void          ggml_go_build_forward_expand(ggml_go_graph g, ggml_go_tensor t) { ggml_build_forward_expand(G(g), T(t)); }

/* --- Backend --- */
ggml_go_backend  ggml_go_metal_init(void)                              { return ggml_backend_metal_init(); }
ggml_go_backend  ggml_go_cpu_init(void)                                { return ggml_backend_cpu_init(); }
void             ggml_go_backend_free(ggml_go_backend backend)         { ggml_backend_free(BE(backend)); }
ggml_go_buf_type ggml_go_backend_buf_type(ggml_go_backend backend)     { return ggml_backend_get_default_buffer_type(BE(backend)); }
ggml_go_buffer   ggml_go_alloc_ctx_tensors(ggml_go_context ctx, ggml_go_backend backend) { return ggml_backend_alloc_ctx_tensors(CTX(ctx), BE(backend)); }
void             ggml_go_buffer_free(ggml_go_buffer buf)               { ggml_backend_buffer_free(BUF(buf)); }
size_t           ggml_go_buffer_size(ggml_go_buffer buf)               { return ggml_backend_buffer_get_size(BUF(buf)); }

/* --- Backend tensor I/O --- */
void ggml_go_tensor_set(ggml_go_tensor t, const void* data, size_t offset, size_t size) { ggml_backend_tensor_set(T(t), data, offset, size); }
void ggml_go_tensor_get(ggml_go_tensor t, void* data, size_t offset, size_t size)       { ggml_backend_tensor_get(T(t), data, offset, size); }

/* --- Scheduler --- */
ggml_go_sched ggml_go_sched_new(ggml_go_backend b0, ggml_go_buf_type bt0,
                                 ggml_go_backend b1, ggml_go_buf_type bt1,
                                 int n_backends, int graph_size)
{
    ggml_backend_t backends[2] = { BE(b0), BE(b1) };
    ggml_backend_buffer_type_t buf_types[2] = { BT(bt0), BT(bt1) };
    return ggml_backend_sched_new(backends, buf_types, n_backends, graph_size, false, true);
}
int  ggml_go_sched_alloc_graph(ggml_go_sched sched, ggml_go_graph g) { return ggml_backend_sched_alloc_graph(SC(sched), G(g)); }
int  ggml_go_sched_compute(ggml_go_sched sched, ggml_go_graph g)    { return (int)ggml_backend_sched_graph_compute(SC(sched), G(g)); }
void ggml_go_sched_free(ggml_go_sched sched)                         { ggml_backend_sched_free(SC(sched)); }

/* --- GGUF --- */
ggml_go_gguf ggml_go_gguf_init(const char* path, ggml_go_context* out_ctx) {
    struct ggml_context* shape_ctx = NULL;
    struct gguf_init_params p = { .no_alloc = true, .ctx = out_ctx ? &shape_ctx : NULL };
    struct gguf_context* gf = gguf_init_from_file(path, p);
    if (out_ctx) *out_ctx = shape_ctx;
    return gf;
}
void            ggml_go_gguf_free(ggml_go_gguf gf)                               { gguf_free(GF(gf)); }
int64_t         ggml_go_gguf_find_key(ggml_go_gguf gf, const char* key)          { return gguf_find_key(GF(gf), key); }
int             ggml_go_gguf_get_kv_type(ggml_go_gguf gf, int64_t idx)            { return (int)gguf_get_kv_type(GF(gf), idx); }
uint32_t        ggml_go_gguf_get_u32(ggml_go_gguf gf, int64_t idx)               { return gguf_get_val_u32(GF(gf), idx); }
float           ggml_go_gguf_get_f32(ggml_go_gguf gf, int64_t idx)               { return gguf_get_val_f32(GF(gf), idx); }
size_t          ggml_go_gguf_get_arr_n(ggml_go_gguf gf, int64_t idx)             { return gguf_get_arr_n(GF(gf), idx); }
const void*     ggml_go_gguf_get_arr_data(ggml_go_gguf gf, int64_t idx)          { return gguf_get_arr_data(GF(gf), idx); }
int             ggml_go_gguf_get_arr_type(ggml_go_gguf gf, int64_t idx)         { return (int)gguf_get_arr_type(GF(gf), idx); }
size_t          ggml_go_gguf_data_offset(ggml_go_gguf gf)                         { return gguf_get_data_offset(GF(gf)); }
int64_t         ggml_go_gguf_n_tensors(ggml_go_gguf gf)                           { return gguf_get_n_tensors(GF(gf)); }
const char*     ggml_go_gguf_tensor_name(ggml_go_gguf gf, int64_t idx)            { return gguf_get_tensor_name(GF(gf), idx); }
size_t          ggml_go_gguf_tensor_offset(ggml_go_gguf gf, int64_t idx)          { return gguf_get_tensor_offset(GF(gf), idx); }
size_t          ggml_go_gguf_tensor_size(ggml_go_gguf gf, int64_t idx)            { return gguf_get_tensor_size(GF(gf), idx); }

/* --- Context iteration --- */
ggml_go_tensor ggml_go_get_first_tensor(ggml_go_context ctx)                     { return ggml_get_first_tensor(CTX(ctx)); }
ggml_go_tensor ggml_go_get_next_tensor(ggml_go_context ctx, ggml_go_tensor t)    { return ggml_get_next_tensor(CTX(ctx), T(t)); }
const char*    ggml_go_tensor_name(ggml_go_tensor t)                             { return T(t)->name; }

/* --- Log callback bridge --- */
// Forward declaration for the CGo-exported Go callback (defined in logging.go via //export).
// CGo generates its own declaration in _cgo_export.h; we declare it locally here so
// this translation unit can reference it without including _cgo_export.h.
extern void ggmlGoLogCallback(int level, char* text, void* user_data);

void ggml_go_register_log_callback(void) {
    ggml_log_set((ggml_log_callback)ggmlGoLogCallback, NULL);
}
