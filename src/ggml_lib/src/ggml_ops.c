/* ggml_ops.c — Pure C wrappers dispatching to ggml API.
 * Each function is a thin cast+forward. No model knowledge. */

#include "ggml_ops.h"
#include "ggml.h"
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
static_assert(GGML_GO_TYPE_BF16 == GGML_TYPE_BF16, "GGML_GO_TYPE_BF16 must match GGML_TYPE_BF16");
static_assert(GGML_GO_TYPE_Q4_0 == GGML_TYPE_Q4_0, "GGML_GO_TYPEQ4_0 must match GGML_TYPEQ4_0");
static_assert(GGML_GO_TYPE_Q4_K == GGML_TYPE_Q4_K, "GGML_GO_TYPEQ4_K must match GGML_TYPEQ4_K");
static_assert(GGML_GO_TYPE_Q6_K == GGML_TYPE_Q6_K, "GGML_GO_TYPEQ6_K must match GGML_TYPEQ6_K");

static_assert(GGML_GO_ROPE_TYPE_NEOX == GGML_ROPE_TYPE_NEOX, "GGML_GO_ROPE_TYPE_NEOX must match GGML_ROPE_TYPE_NEOX");

static_assert(GGML_GO_BACKEND_DEVICE_TYPE_CPU == GGML_BACKEND_DEVICE_TYPE_CPU, "GGML_GO_BACKEND_DEVICE_TYPE_CPU must match GGML_BACKEND_DEVICE_TYPE_CPU");
static_assert(GGML_GO_BACKEND_DEVICE_TYPE_GPU == GGML_BACKEND_DEVICE_TYPE_GPU, "GGML_GO_BACKEND_DEVICE_TYPE_GPU must match GGML_BACKEND_DEVICE_TYPE_GPU");
static_assert(GGML_GO_BACKEND_DEVICE_TYPE_IGPU == GGML_BACKEND_DEVICE_TYPE_IGPU, "GGML_GO_BACKEND_DEVICE_TYPE_IGPU must match GGML_BACKEND_DEVICE_TYPE_IGPU");
static_assert(GGML_GO_BACKEND_DEVICE_TYPE_ACCEL == GGML_BACKEND_DEVICE_TYPE_ACCEL, "GGML_GO_BACKEND_DEVICE_TYPE_ACCEL must match GGML_BACKEND_DEVICE_TYPE_ACCEL");

/* Casting helpers */
#define CTX(x)  ((struct ggml_context*)(x))
#define T(x)    ((struct ggml_tensor*)(x))
#define G(x)    ((struct ggml_cgraph*)(x))
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
ggml_go_tensor ggml_go_cpy(ggml_go_context ctx, ggml_go_tensor a, ggml_go_tensor b)                       { return ggml_cpy(CTX(ctx), T(a), T(b)); }
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
ggml_go_tensor ggml_go_scale_bias(ggml_go_context ctx, ggml_go_tensor a, float s, float b)  { return ggml_scale_bias(CTX(ctx), T(a), s, b); }
ggml_go_tensor ggml_go_clamp(ggml_go_context ctx, ggml_go_tensor a, float min_val, float max_val) { return ggml_clamp(CTX(ctx), T(a), min_val, max_val); }
ggml_go_tensor ggml_go_sum_rows(ggml_go_context ctx, ggml_go_tensor a)                     { return ggml_sum_rows(CTX(ctx), T(a)); }
ggml_go_tensor ggml_go_sum(ggml_go_context ctx, ggml_go_tensor a)                          { return ggml_sum(CTX(ctx), T(a)); }
ggml_go_tensor ggml_go_sqrt(ggml_go_context ctx, ggml_go_tensor a)                         { return ggml_sqrt(CTX(ctx), T(a)); }
ggml_go_tensor ggml_go_exp(ggml_go_context ctx, ggml_go_tensor a)                          { return ggml_exp(CTX(ctx), T(a)); }
ggml_go_tensor ggml_go_neg(ggml_go_context ctx, ggml_go_tensor a)                          { return ggml_neg(CTX(ctx), T(a)); }

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

/* --- Flash attention --- */
ggml_go_tensor ggml_go_flash_attn_ext(ggml_go_context ctx, ggml_go_tensor q, ggml_go_tensor k, ggml_go_tensor v, ggml_go_tensor mask, float scale, float max_bias, float logit_softcap) { return ggml_flash_attn_ext(CTX(ctx), T(q), T(k), T(v), T(mask), scale, max_bias, logit_softcap); }
void ggml_go_flash_attn_ext_set_prec(ggml_go_tensor t, int prec) { ggml_flash_attn_ext_set_prec(T(t), (enum ggml_prec)prec); }
ggml_go_tensor ggml_go_cast(ggml_go_context ctx, ggml_go_tensor a, int type) { return ggml_cast(CTX(ctx), T(a), (enum ggml_type)type); }

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

int     ggml_go_validate_row_data(int type, const void* data, size_t nbytes) {
    return ggml_validate_row_data((enum ggml_type)type, data, nbytes) ? 1 : 0;
}

/* --- Graph compute (CPU-only, no backend/scheduler needed) --- */
void ggml_go_graph_compute(ggml_go_context ctx, ggml_go_graph g, int n_threads) {
    ggml_graph_compute_with_ctx(CTX(ctx), G(g), n_threads);
}

/* --- Tensor data pointer access (sets direct data pointer, no backend copy) --- */
void* ggml_go_tensor_data(ggml_go_tensor t)          { return T(t)->data; }
void  ggml_go_tensor_set_data(ggml_go_tensor t, void* data) { T(t)->data = data; }

/* --- Graph overhead accounting --- */
size_t ggml_go_graph_overhead_custom(int size, int grads) {
    return ggml_graph_overhead_custom((size_t)size, grads != 0);
}

/* --- Context lifecycle --- */
ggml_go_context ggml_go_init(size_t mem_size, int no_alloc) {
    struct ggml_init_params p = { .mem_size = mem_size, .mem_buffer = NULL, .no_alloc = no_alloc };
    return ggml_init(p);
}
void   ggml_go_free(ggml_go_context ctx)     { ggml_free(CTX(ctx)); }
void   ggml_go_reset(ggml_go_context ctx)    { ggml_reset(CTX(ctx)); }
size_t ggml_go_used_mem(ggml_go_context ctx) { return ggml_used_mem(CTX(ctx)); }

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
void             ggml_go_buffer_clear(ggml_go_buffer buf, uint8_t v)  { ggml_backend_buffer_clear(BUF(buf), v); }

/* --- Backend tensor I/O --- */
void ggml_go_tensor_set(ggml_go_tensor t, const void* data, size_t offset, size_t size) { ggml_backend_tensor_set(T(t), data, offset, size); }
void ggml_go_tensor_get(ggml_go_tensor t, void* data, size_t offset, size_t size)       { ggml_backend_tensor_get(T(t), data, offset, size); }

/* Mirrors the ggml_backend_tensor_set/get null check, including the view_src
 * indirection, so callers can skip I/O on tensors a scheduler decided not
 * to allocate (e.g. inputs referenced by no op in a per-layer graph). */
int ggml_go_tensor_has_buffer(ggml_go_tensor t) {
    if (t == NULL) return 0;
    struct ggml_tensor* tt = T(t);
    struct ggml_backend_buffer* buf = tt->view_src ? tt->view_src->buffer : tt->buffer;
    return buf != NULL;
}

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
void ggml_go_sched_reset(ggml_go_sched sched)                        { ggml_backend_sched_reset(SC(sched)); }

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

/* --- Memory tracking --- */
size_t ggml_go_backend_sched_get_buffer_size(ggml_go_sched sched, ggml_go_backend backend) {
    return ggml_backend_sched_get_buffer_size(SC(sched), BE(backend));
}

void ggml_go_dev_memory(ggml_go_backend backend, size_t *free, size_t *total) {
    ggml_backend_dev_t dev = ggml_backend_get_device(BE(backend));
    ggml_backend_dev_memory(dev, free, total);
}

/* Device type */
int ggml_go_backend_dev_type(ggml_go_backend backend) {
    return (int)ggml_backend_dev_type(ggml_backend_get_device(BE(backend)));
}

int ggml_go_backend_is_metal(ggml_go_backend backend) {
    return (int)ggml_backend_is_metal(BE(backend));
}
