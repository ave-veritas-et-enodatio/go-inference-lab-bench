/* ggml_ops.h — Pure C wrappers for ggml operations.
 * Model-agnostic: no architecture knowledge.
 * Used by Go via CGO. */

#ifndef GGML_OPS_H
#define GGML_OPS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque typed handles — CGO maps these to distinct Go pointer types.
 * Go side casts via unsafe.Pointer at the boundary. */
typedef void* ggml_go_tensor;
typedef void* ggml_go_context;
typedef void* ggml_go_graph;
typedef void* ggml_go_backend;
typedef void* ggml_go_buffer;
typedef void* ggml_go_buf_type;
typedef void* ggml_go_sched;
/* --- Type constants (must match ggml.h enum ggml_type) --- */
enum {
    GGML_GO_TYPE_F32  = 0,
    GGML_GO_TYPE_F16  = 1,
    GGML_GO_TYPE_I32  = 26,
};
enum {
    GGML_GO_ROPE_NEOX = 2,
};
enum {
    GGML_GO_PREC_F32 = 1, /* GGML_PREC_F32 */
};
enum {
    GGML_GO_STATUS_SUCCESS = 0,
};

enum {
    GGML_GO_BACKEND_DEVICE_TYPE_CPU = 0,
    GGML_GO_BACKEND_DEVICE_TYPE_GPU = 1,
    GGML_GO_BACKEND_DEVICE_TYPE_IGPU = 2,
    GGML_GO_BACKEND_DEVICE_TYPE_ACCEL = 3,
};
/* --- Tensor creation --- */
ggml_go_tensor ggml_go_new_tensor_1d(ggml_go_context ctx, int type, int64_t ne0);
ggml_go_tensor ggml_go_new_tensor_2d(ggml_go_context ctx, int type, int64_t ne0, int64_t ne1);
ggml_go_tensor ggml_go_new_tensor_3d(ggml_go_context ctx, int type, int64_t ne0, int64_t ne1, int64_t ne2);
ggml_go_tensor ggml_go_new_tensor_4d(ggml_go_context ctx, int type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

/* --- Views (no data copy, strided references) --- */
ggml_go_tensor ggml_go_view_2d(ggml_go_context ctx, ggml_go_tensor a, int64_t ne0, int64_t ne1, size_t nb1, size_t offset);
ggml_go_tensor ggml_go_view_3d(ggml_go_context ctx, ggml_go_tensor a, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, size_t nb2, size_t offset);
ggml_go_tensor ggml_go_view_4d(ggml_go_context ctx, ggml_go_tensor a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1, size_t nb2, size_t nb3, size_t offset);

/* --- Reshape --- */
ggml_go_tensor ggml_go_reshape_2d(ggml_go_context ctx, ggml_go_tensor a, int64_t ne0, int64_t ne1);
ggml_go_tensor ggml_go_reshape_3d(ggml_go_context ctx, ggml_go_tensor a, int64_t ne0, int64_t ne1, int64_t ne2);
ggml_go_tensor ggml_go_reshape_4d(ggml_go_context ctx, ggml_go_tensor a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

/* --- Layout ops --- */
ggml_go_tensor ggml_go_cpy(ggml_go_context ctx, ggml_go_tensor a, ggml_go_tensor b);
ggml_go_tensor ggml_go_permute(ggml_go_context ctx, ggml_go_tensor a, int ax0, int ax1, int ax2, int ax3);
ggml_go_tensor ggml_go_transpose(ggml_go_context ctx, ggml_go_tensor a);
ggml_go_tensor ggml_go_cont(ggml_go_context ctx, ggml_go_tensor a);
ggml_go_tensor ggml_go_cont_2d(ggml_go_context ctx, ggml_go_tensor a, int64_t ne0, int64_t ne1);
ggml_go_tensor ggml_go_concat(ggml_go_context ctx, ggml_go_tensor a, ggml_go_tensor b, int dim);
ggml_go_tensor ggml_go_repeat_4d(ggml_go_context ctx, ggml_go_tensor a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

/* --- Arithmetic --- */
ggml_go_tensor ggml_go_add(ggml_go_context ctx, ggml_go_tensor a, ggml_go_tensor b);
ggml_go_tensor ggml_go_mul(ggml_go_context ctx, ggml_go_tensor a, ggml_go_tensor b);
ggml_go_tensor ggml_go_div(ggml_go_context ctx, ggml_go_tensor a, ggml_go_tensor b);
ggml_go_tensor ggml_go_scale(ggml_go_context ctx, ggml_go_tensor a, float s);
ggml_go_tensor ggml_go_clamp(ggml_go_context ctx, ggml_go_tensor a, float min_val, float max_val);
ggml_go_tensor ggml_go_sum_rows(ggml_go_context ctx, ggml_go_tensor a);
ggml_go_tensor ggml_go_sum(ggml_go_context ctx, ggml_go_tensor a);
ggml_go_tensor ggml_go_sqrt(ggml_go_context ctx, ggml_go_tensor a);
ggml_go_tensor ggml_go_mul_mat(ggml_go_context ctx, ggml_go_tensor a, ggml_go_tensor b);
ggml_go_tensor ggml_go_mul_mat_id(ggml_go_context ctx, ggml_go_tensor as, ggml_go_tensor b, ggml_go_tensor ids);

/* --- Normalization --- */
ggml_go_tensor ggml_go_rms_norm(ggml_go_context ctx, ggml_go_tensor a, float eps);
ggml_go_tensor ggml_go_l2_norm(ggml_go_context ctx, ggml_go_tensor a, float eps);

/* --- Activations --- */
ggml_go_tensor ggml_go_silu(ggml_go_context ctx, ggml_go_tensor a);
ggml_go_tensor ggml_go_sigmoid(ggml_go_context ctx, ggml_go_tensor a);
ggml_go_tensor ggml_go_softplus(ggml_go_context ctx, ggml_go_tensor a);
ggml_go_tensor ggml_go_gelu(ggml_go_context ctx, ggml_go_tensor a);
ggml_go_tensor ggml_go_tanh(ggml_go_context ctx, ggml_go_tensor a);
ggml_go_tensor ggml_go_soft_max_ext(ggml_go_context ctx, ggml_go_tensor a, ggml_go_tensor mask, float scale, float max_bias);

/* --- Embedding / indexing --- */
ggml_go_tensor ggml_go_get_rows(ggml_go_context ctx, ggml_go_tensor a, ggml_go_tensor b);

/* --- Sorting / selection --- */
ggml_go_tensor ggml_go_argsort_top_k(ggml_go_context ctx, ggml_go_tensor a, int k);

/* --- RoPE --- */
ggml_go_tensor ggml_go_rope_ext(ggml_go_context ctx, ggml_go_tensor a, ggml_go_tensor pos, ggml_go_tensor freq_factors,
    int n_dims, int mode, int n_ctx_orig,
    float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow);
ggml_go_tensor ggml_go_rope_multi(ggml_go_context ctx, ggml_go_tensor a, ggml_go_tensor pos, ggml_go_tensor freq_factors,
    int n_dims, int s0, int s1, int s2, int s3, int mode, int n_ctx_orig,
    float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow);

/* --- SSM / delta-net --- */
ggml_go_tensor ggml_go_ssm_conv(ggml_go_context ctx, ggml_go_tensor sx, ggml_go_tensor c);
ggml_go_tensor ggml_go_gated_delta_net(ggml_go_context ctx, ggml_go_tensor q, ggml_go_tensor k, ggml_go_tensor v,
    ggml_go_tensor g, ggml_go_tensor beta, ggml_go_tensor state);

/* --- Flash attention --- */
ggml_go_tensor ggml_go_flash_attn_ext(ggml_go_context ctx,
    ggml_go_tensor q, ggml_go_tensor k, ggml_go_tensor v, ggml_go_tensor mask,
    float scale, float max_bias, float logit_softcap);
void ggml_go_flash_attn_ext_set_prec(ggml_go_tensor t, int prec);
ggml_go_tensor ggml_go_cast(ggml_go_context ctx, ggml_go_tensor a, int type);

/* --- Precision --- */
void ggml_go_mul_mat_set_prec_f32(ggml_go_tensor t);

/* --- Tensor flags --- */
void ggml_go_set_input(ggml_go_tensor t);
void ggml_go_set_output(ggml_go_tensor t);
void ggml_go_set_name(ggml_go_tensor t, const char* name);

/* --- Tensor accessors --- */
int64_t ggml_go_ne(ggml_go_tensor t, int dim);
size_t  ggml_go_nb(ggml_go_tensor t, int dim);
size_t  ggml_go_element_size(ggml_go_tensor t);
size_t  ggml_go_nbytes(ggml_go_tensor t);
size_t  ggml_go_row_size(int type, int64_t ne);
size_t  ggml_go_tensor_overhead(void);
int     ggml_go_tensor_type(ggml_go_tensor t);

/* --- Context lifecycle --- */
ggml_go_context ggml_go_init(size_t mem_size);
void            ggml_go_free(ggml_go_context ctx);

/* --- Graph --- */
ggml_go_graph   ggml_go_new_graph(ggml_go_context ctx, int size);
void            ggml_go_build_forward_expand(ggml_go_graph g, ggml_go_tensor t);

/* --- Backend --- */
ggml_go_backend  ggml_go_metal_init(void);
ggml_go_backend  ggml_go_cpu_init(void);
void             ggml_go_backend_free(ggml_go_backend backend);
ggml_go_buf_type ggml_go_backend_buf_type(ggml_go_backend backend);
ggml_go_buffer   ggml_go_alloc_ctx_tensors(ggml_go_context ctx, ggml_go_backend backend);
void             ggml_go_buffer_free(ggml_go_buffer buf);
size_t           ggml_go_buffer_size(ggml_go_buffer buf);
void             ggml_go_buffer_clear(ggml_go_buffer buf, uint8_t value);

/* --- Backend tensor I/O --- */
void ggml_go_tensor_set(ggml_go_tensor t, const void* data, size_t offset, size_t size);
void ggml_go_tensor_get(ggml_go_tensor t, void* data, size_t offset, size_t size);

/* --- Scheduler --- */
ggml_go_sched ggml_go_sched_new(ggml_go_backend b0, ggml_go_buf_type bt0,
                                 ggml_go_backend b1, ggml_go_buf_type bt1,
                                 int n_backends, int graph_size);
int  ggml_go_sched_alloc_graph(ggml_go_sched sched, ggml_go_graph g);
int  ggml_go_sched_compute(ggml_go_sched sched, ggml_go_graph g);
void ggml_go_sched_free(ggml_go_sched sched);
void ggml_go_sched_reset(ggml_go_sched sched);

/* --- Context iteration --- */
ggml_go_tensor ggml_go_get_first_tensor(ggml_go_context ctx);
ggml_go_tensor ggml_go_get_next_tensor(ggml_go_context ctx, ggml_go_tensor t);
const char*    ggml_go_tensor_name(ggml_go_tensor t);

/* --- Memory tracking --- */
size_t          ggml_go_backend_sched_get_buffer_size(ggml_go_sched sched, ggml_go_backend backend);
void            ggml_go_dev_memory(ggml_go_backend backend, size_t *free, size_t *total);

int             ggml_go_backend_dev_type(ggml_go_backend backend);
int             ggml_go_backend_is_metal(ggml_go_backend backend);


// ggml_go_register_log_callback installs the Go log callback as ggml's log handler.
// The callback itself (ggmlGoLogCallback) is defined in logging.go via //export —
// CGo auto-generates its declaration; do not redeclare it here.
void ggml_go_register_log_callback(void);

#ifdef __cplusplus
}
#endif

#endif /* GGML_OPS_H */
