#ifndef __C_TENSOR_H__
#define __C_TENSOR_H__

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

/* ============================================================================
 * Tensor Data Structure
 * Supports: dynamic shape, strides, quantization, multiple layouts
 * ============================================================================ */

typedef enum {
    TENSOR_LAYOUT_NCHW,  // PyTorch: [N, C, H, W]
    TENSOR_LAYOUT_NHWC,  // TensorFlow: [N, H, W, C]
    TENSOR_LAYOUT_CHWN,  // Channel-first for convolutions
    TENSOR_LAYOUT_OWI,   // [Out, Win, In] - flattened weights
} TensorLayout;

typedef enum {
    TENSOR_DTYPE_F32,
    TENSOR_DTYPE_F16,
    TENSOR_DTYPE_F64,
    TENSOR_DTYPE_INT8,
    TENSOR_DTYPE_INT16,
    TENSOR_DTYPE_INT32,
    TENSOR_DTYPE_INT64,
    TENSOR_DTYPE_UINT8,
    TENSOR_DTYPE_UINT16,
} TensorDType;

typedef enum {
    QUANT_NONE,
    QUANT_AFFINE,       // zero_point + scale
    QUANT_SYMMETRIC,    // zero_point = 0
} QuantType;

typedef struct {
    float scale;
    int32_t zero_point;
} QuantParams;

typedef struct {
    TensorDType dtype;
    TensorLayout layout;
    uint8_t ndim;
    size_t* shape;       // [d0, d1, ..., dn-1]
    size_t* strides;     // [s0, s1, ..., sn-1], in elements not bytes
    void* data;
    size_t size;         // total elements
    size_t capacity;     // allocated capacity

    // Quantization
    QuantType quant_type;
    QuantParams quant;
} Tensor;

static inline size_t tensor_dtype_size(TensorDType dtype) {
    switch (dtype) {
        case TENSOR_DTYPE_F32:
        case TENSOR_DTYPE_INT32:  return 4;
        case TENSOR_DTYPE_F16:
        case TENSOR_DTYPE_INT16:  return 2;
        case TENSOR_DTYPE_F64:
        case TENSOR_DTYPE_INT64: return 8;
        case TENSOR_DTYPE_INT8:
        case TENSOR_DTYPE_UINT8:  return 1;
        default: return 4;
    }
}

static inline size_t tensor_elem_count(const Tensor* t) {
    size_t count = 1;
    for (uint8_t i = 0; i < t->ndim; i++) count *= t->shape[i];
    return count;
}

static inline size_t tensor_nbytes(const Tensor* t) {
    return tensor_elem_count(t) * tensor_dtype_size(t->dtype);
}

static void tensor_compute_strides(Tensor* t) {
    if (t->layout == TENSOR_LAYOUT_NHWC) {
        // NHWC: [N, H, W, C] -> strides = [H*W*C, W*C, C, 1]
        t->strides[3] = 1;
        t->strides[2] = t->shape[3];
        t->strides[1] = t->shape[2] * t->shape[3];
        t->strides[0] = t->shape[1] * t->shape[2] * t->shape[3];
    } else {
        // Default NCHW / row-major: [N, C, H, W] -> strides = [C*H*W, H*W, W, 1]
        t->strides[t->ndim - 1] = 1;
        for (int i = (int)t->ndim - 2; i >= 0; i--) {
            t->strides[i] = t->strides[i + 1] * t->shape[i + 1];
        }
    }
}

static Tensor* tensor_create(TensorDType dtype, TensorLayout layout,
                               const size_t* shape, uint8_t ndim) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->dtype = dtype;
    t->layout = layout;
    t->ndim = ndim;
    t->shape = (size_t*)malloc(sizeof(size_t) * ndim);
    t->strides = (size_t*)malloc(sizeof(size_t) * ndim);
    t->size = 1;
    for (uint8_t i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->size *= shape[i];
    }
    t->capacity = t->size;
    tensor_compute_strides(t);

    size_t bytes = t->size * tensor_dtype_size(dtype);
    t->data = malloc(bytes);
    memset(t->data, 0, bytes);

    t->quant_type = QUANT_NONE;
    t->quant.scale = 1.0f;
    t->quant.zero_point = 0;
    return t;
}

static void tensor_free(Tensor* t) {
    if (!t) return;
    free(t->shape);
    free(t->strides);
    free(t->data);
    free(t);
}

static inline size_t tensor_flat_index(const Tensor* t, const size_t* indices) {
    size_t idx = 0;
    for (uint8_t i = 0; i < t->ndim; i++) idx += indices[i] * t->strides[i];
    return idx;
}

static inline void* tensor_at(const Tensor* t, const size_t* indices) {
    return (uint8_t*)t->data + tensor_flat_index(t, indices) * tensor_dtype_size(t->dtype);
}

static inline float tensor_get_f32(const Tensor* t, const size_t* indices) {
    size_t idx = tensor_flat_index(t, indices);
    switch (t->dtype) {
        case TENSOR_DTYPE_F32: return ((float*)t->data)[idx];
        case TENSOR_DTYPE_INT32: return (float)((int32_t*)t->data)[idx];
        case TENSOR_DTYPE_INT8: return (float)((int8_t*)t->data)[idx];
        case TENSOR_DTYPE_UINT8: return (float)((uint8_t*)t->data)[idx];
        default: return 0.0f;
    }
}

static inline void tensor_set_f32(Tensor* t, const size_t* indices, float val) {
    size_t idx = tensor_flat_index(t, indices);
    switch (t->dtype) {
        case TENSOR_DTYPE_F32: ((float*)t->data)[idx] = val; break;
        case TENSOR_DTYPE_INT32: ((int32_t*)t->data)[idx] = (int32_t)val; break;
        case TENSOR_DTYPE_INT8: ((int8_t*)t->data)[idx] = (int8_t)val; break;
        case TENSOR_DTYPE_UINT8: ((uint8_t*)t->data)[idx] = (uint8_t)val; break;
        default: break;
    }
}

static Tensor* tensor_reshape(const Tensor* t, const size_t* new_shape, uint8_t ndim) {
    size_t new_size = 1;
    for (uint8_t i = 0; i < ndim; i++) new_size *= new_shape[i];
    if (new_size != t->size) return NULL;

    Tensor* r = (Tensor*)malloc(sizeof(Tensor));
    r->dtype = t->dtype;
    r->layout = t->layout;
    r->ndim = ndim;
    r->shape = (size_t*)malloc(sizeof(size_t) * ndim);
    r->strides = (size_t*)malloc(sizeof(size_t) * ndim);
    memcpy(r->shape, new_shape, sizeof(size_t) * ndim);
    r->size = t->size;
    r->capacity = t->capacity;
    r->data = t->data;  // share data pointer
    r->quant_type = t->quant_type;
    r->quant = t->quant;
    tensor_compute_strides(r);
    return r;
}

static Tensor* tensor_clone(const Tensor* t) {
    Tensor* c = tensor_create(t->dtype, t->layout, t->shape, t->ndim);
    memcpy(c->data, t->data, tensor_nbytes(t));
    c->quant_type = t->quant_type;
    c->quant = t->quant;
    return c;
}

static Tensor* tensor_transpose(const Tensor* t, uint8_t axis0, uint8_t axis1) {
    if (axis0 >= t->ndim || axis1 >= t->ndim) return NULL;

    Tensor* r = tensor_create(t->dtype, t->layout, t->shape, t->ndim);

    // Compute flat index mapping and actually reorder the data
    size_t elem_size = tensor_dtype_size(t->dtype);
    uint8_t* src = (uint8_t*)t->data;
    uint8_t* dst = (uint8_t*)r->data;

    // For each element in output, compute source index
    size_t idx[8]; // ndim is typically small (max 255 but practical use << 8)
    for (size_t i = 0; i < r->size; i++) {
        // Convert flat output index to multi-dimensional indices
        size_t remaining = i;
        for (uint8_t d = 0; d < t->ndim; d++) {
            idx[d] = remaining % r->shape[d];
            remaining /= r->shape[d];
        }
        // Swap axes for source index
        size_t src_idx0 = idx[axis0]; idx[axis0] = idx[axis1]; idx[axis1] = src_idx0;
        // Convert back to flat source index
        size_t src_flat = 0;
        for (uint8_t d = 0; d < t->ndim; d++) {
            src_flat += idx[d] * t->strides[d];
        }
        memcpy(dst + i * elem_size, src + src_flat * elem_size, elem_size);
    }

    // Swap shape and strides
    size_t tmp = r->shape[axis0]; r->shape[axis0] = r->shape[axis1]; r->shape[axis1] = tmp;
    tmp = r->strides[axis0]; r->strides[axis0] = r->strides[axis1]; r->strides[axis1] = tmp;
    return r;
}

static Tensor* tensor_slice(const Tensor* t, size_t dim, size_t start, size_t end) {
    if (dim >= t->ndim || start >= t->shape[dim] || end > t->shape[dim] || start >= end) return NULL;

    Tensor* r = tensor_create(t->dtype, t->layout, t->shape, t->ndim);
    r->shape[dim] = end - start;
    r->size = tensor_elem_count(r);
    r->capacity = t->capacity;

    size_t elem_size = tensor_dtype_size(t->dtype);
    size_t offset = start * t->strides[dim];
    r->data = (uint8_t*)t->data + offset * elem_size;

    r->quant_type = t->quant_type;
    r->quant = t->quant;
    return r;
}

/* ============================================================================
 * Batch Normalization
 * Input: [N, C, H, W] (for 2D) or [N, C] (for 1D)
 * Computes: y = gamma * (x - mean) / sqrt(var + eps) + beta
 * ============================================================================ */

static void tensor_batch_norm_train(const Tensor* x, Tensor* y,
                                    const Tensor* gamma, const Tensor* beta,
                                    Tensor* mean, Tensor* var,
                                    Tensor* running_mean, Tensor* running_var,
                                    float momentum, float eps) {
    // x: [N, C, H, W] or [N, C]
    size_t N = x->shape[0];
    size_t C = x->shape[1];
    size_t H = (x->ndim == 4) ? x->shape[2] : 1;
    size_t W = (x->ndim == 4) ? x->shape[3] : 1;
    size_t HW = H * W;

    float* x_data = (float*)x->data;
    float* y_data = (float*)y->data;
    float* mean_data = (float*)mean->data;
    float* var_data = (float*)var->data;
    float* gamma_data = (float*)gamma->data;
    float* beta_data = (float*)beta->data;

    // Compute mean: mean_c = (1/NHW) * sum_{n,h,w} x_{n,c,h,w}
    for (size_t c = 0; c < C; c++) {
        float sum = 0.0f;
        for (size_t n = 0; n < N; n++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    size_t idx = n * C * HW + c * HW + h * W + w;
                    sum += x_data[idx];
                }
            }
        }
        mean_data[c] = sum / (N * HW);
    }

    // Compute variance: var_c = (1/NHW) * sum (x - mean)^2
    for (size_t c = 0; c < C; c++) {
        float sum = 0.0f;
        for (size_t n = 0; n < N; n++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    size_t idx = n * C * HW + c * HW + h * W + w;
                    float diff = x_data[idx] - mean_data[c];
                    sum += diff * diff;
                }
            }
        }
        var_data[c] = sum / (N * HW);
    }

    // Update running stats
    float m = momentum;
    for (size_t c = 0; c < C; c++) {
        float* rm_data = (float*)running_mean->data;
        float* rv_data = (float*)running_var->data;
        rm_data[c] = m * rm_data[c] + (1 - m) * mean_data[c];
        rv_data[c] = m * rv_data[c] + (1 - m) * var_data[c];
    }

    // Normalize and scale
    for (size_t c = 0; c < C; c++) {
        float std = sqrtf(var_data[c] + eps);
        float inv_std = 1.0f / std;
        float gamma_c = gamma_data[c];
        float beta_c = beta_data[c];
        float mean_c = mean_data[c];

        for (size_t n = 0; n < N; n++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    size_t idx = n * C * HW + c * HW + h * W + w;
                    y_data[idx] = gamma_c * (x_data[idx] - mean_c) * inv_std + beta_c;
                }
            }
        }
    }
}

static void tensor_batch_norm_inference(const Tensor* x, Tensor* y,
                                        const Tensor* gamma, const Tensor* beta,
                                        const Tensor* mean, const Tensor* var,
                                        float eps) {
    size_t N = x->shape[0];
    size_t C = x->shape[1];
    size_t H = (x->ndim == 4) ? x->shape[2] : 1;
    size_t W = (x->ndim == 4) ? x->shape[3] : 1;
    size_t HW = H * W;

    float* x_data = (float*)x->data;
    float* y_data = (float*)y->data;
    float* mean_data = (float*)mean->data;
    float* var_data = (float*)var->data;
    float* gamma_data = (float*)gamma->data;
    float* beta_data = (float*)beta->data;

    for (size_t c = 0; c < C; c++) {
        float std = sqrtf(var_data[c] + eps);
        float inv_std = 1.0f / std;
        float gamma_c = gamma_data[c];
        float beta_c = beta_data[c];
        float mean_c = mean_data[c];

        for (size_t n = 0; n < N; n++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    size_t idx = n * C * HW + c * HW + h * W + w;
                    y_data[idx] = gamma_c * (x_data[idx] - mean_c) * inv_std + beta_c;
                }
            }
        }
    }
}

/* ============================================================================
 * Dropout
 * During training: y = x * mask / (1-p), where mask_i ~ Bernoulli(1-p)
 * During inference: y = x
 * ============================================================================ */

static void tensor_dropout_forward(const Tensor* x, Tensor* y, Tensor* mask,
                                   float p, bool training) {
    float* x_data = (float*)x->data;
    float* y_data = (float*)y->data;
    uint8_t* mask_data = (uint8_t*)mask->data;

    if (!training) {
        // Inference: just copy
        for (size_t i = 0; i < x->size; i++) y_data[i] = x_data[i];
        return;
    }

    // Training: apply dropout
    float scale = 1.0f / (1.0f - p);
    for (size_t i = 0; i < x->size; i++) {
        mask_data[i] = (rand() < (float)RAND_MAX * (1.0f - p)) ? 1 : 0;
        y_data[i] = x_data[i] * mask_data[i] * scale;
    }
}

static void tensor_dropout_backward(const Tensor* grad_output, Tensor* grad_input,
                                    const Tensor* mask, float p) {
    float* go_data = (float*)grad_output->data;
    float* gi_data = (float*)grad_input->data;
    uint8_t* mask_data = (uint8_t*)mask->data;

    float scale = 1.0f / (1.0f - p);
    for (size_t i = 0; i < grad_output->size; i++) {
        gi_data[i] = go_data[i] * mask_data[i] * scale;
    }
}

/* ============================================================================
 * Layer Normalization
 * Input: [N, D] or [N, H, W, D]
 * Normalizes over last dimension
 * ============================================================================ */

static void tensor_layer_norm_forward(const Tensor* x, Tensor* y,
                                      const Tensor* gamma, const Tensor* beta,
                                      float eps) {
    float* x_data = (float*)x->data;
    float* y_data = (float*)y->data;
    float* gamma_data = (float*)gamma->data;
    float* beta_data = (float*)beta->data;

    size_t last_dim = x->shape[x->ndim - 1];
    size_t outer_size = x->size / last_dim;

    for (size_t i = 0; i < outer_size; i++) {
        // Compute mean
        float sum = 0.0f;
        for (size_t j = 0; j < last_dim; j++) {
            sum += x_data[i * last_dim + j];
        }
        float mean = sum / last_dim;

        // Compute variance
        float var_sum = 0.0f;
        for (size_t j = 0; j < last_dim; j++) {
            float diff = x_data[i * last_dim + j] - mean;
            var_sum += diff * diff;
        }
        float var = var_sum / last_dim;
        float std = sqrtf(var + eps);
        float inv_std = 1.0f / std;

        // Normalize and scale
        for (size_t j = 0; j < last_dim; j++) {
            float norm = (x_data[i * last_dim + j] - mean) * inv_std;
            y_data[i * last_dim + j] = gamma_data[j] * norm + beta_data[j];
        }
    }
}

/* ============================================================================
 * Upsample (Nearest Neighbor)
 * Input: [N, C, H, W], Output: [N, C, H*scale_h, W*scale_w]
 * ============================================================================ */

static Tensor* tensor_upsample_nearest_2d(const Tensor* x, size_t scale_h, size_t scale_w) {
    if (x->ndim != 4) return NULL;

    size_t N = x->shape[0];
    size_t C = x->shape[1];
    size_t H = x->shape[2];
    size_t W = x->shape[3];

    size_t out_shape[] = {N, C, H * scale_h, W * scale_w};
    Tensor* y = tensor_create(TENSOR_DTYPE_F32, x->layout, out_shape, 4);

    float* x_data = (float*)x->data;
    float* y_data = (float*)y->data;

    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    float val = x_data[n * C * H * W + c * H * W + h * W + w];
                    // Fill the upsampled block
                    for (size_t sh = 0; sh < scale_h; sh++) {
                        for (size_t sw = 0; sw < scale_w; sw++) {
                            size_t out_h = h * scale_h + sh;
                            size_t out_w = w * scale_w + sw;
                            y_data[n * C * (H * scale_h) * (W * scale_w) +
                                   c * (H * scale_h) * (W * scale_w) +
                                   out_h * (W * scale_w) + out_w] = val;
                        }
                    }
                }
            }
        }
    }
    return y;
}

/* ============================================================================
 * Quantization
 * ============================================================================ */

static void tensor_quantize(const Tensor* t, float* min_val, float* max_val) {
    *min_val = FLT_MAX;
    *max_val = -FLT_MAX;
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->size; i++) {
        if (data[i] < *min_val) *min_val = data[i];
        if (data[i] > *max_val) *max_val = data[i];
    }
}

static Tensor* tensor_quantize_affine(Tensor* t, TensorDType dtype) {
    float min_val, max_val;
    tensor_quantize(t, &min_val, &max_val);

    float scale = (max_val - min_val) / 255.0f;
    int32_t zero_point = (int32_t)(-min_val / scale);

    Tensor* q = tensor_create(dtype, t->layout, t->shape, t->ndim);
    q->quant_type = QUANT_AFFINE;
    q->quant.scale = scale;
    q->quant.zero_point = zero_point;

    float* src = (float*)t->data;
    switch (dtype) {
        case TENSOR_DTYPE_UINT8: {
            uint8_t* dst = (uint8_t*)q->data;
            for (size_t i = 0; i < t->size; i++) {
                int32_t val = (int32_t)(src[i] / scale + zero_point);
                if (val < 0) val = 0; if (val > 255) val = 255;
                dst[i] = (uint8_t)val;
            }
            break;
        }
        case TENSOR_DTYPE_INT8: {
            int8_t* dst = (int8_t*)q->data;
            for (size_t i = 0; i < t->size; i++) {
                int32_t val = (int32_t)(src[i] / scale + zero_point);
                if (val < -128) val = -128; if (val > 127) val = 127;
                dst[i] = (int8_t)val;
            }
            break;
        }
        default: break;
    }
    return q;
}

static Tensor* tensor_dequantize(const Tensor* t) {
    if (t->quant_type == QUANT_NONE) return tensor_clone(t);

    Tensor* f = tensor_create(TENSOR_DTYPE_F32, t->layout, t->shape, t->ndim);
    float* dst = (float*)f->data;
    float scale = t->quant.scale;
    int32_t zp = t->quant.zero_point;

    switch (t->dtype) {
        case TENSOR_DTYPE_INT8: {
            int8_t* src = (int8_t*)t->data;
            for (size_t i = 0; i < t->size; i++) dst[i] = (src[i] - zp) * scale;
            break;
        }
        case TENSOR_DTYPE_UINT8: {
            uint8_t* src = (uint8_t*)t->data;
            for (size_t i = 0; i < t->size; i++) dst[i] = (src[i] - zp) * scale;
            break;
        }
        default: break;
    }
    return f;
}

/* ============================================================================
 * Fill & Initialization
 * ============================================================================ */

static void tensor_fill_f32(Tensor* t, float val) {
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->size; i++) data[i] = val;
}

static void tensor_fill_randn(Tensor* t, float mean, float std) {
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->size; i++) {
        float u1 = (float)rand() / (float)(RAND_MAX);
        float u2 = (float)rand() / (float)(RAND_MAX);
        float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * acosf(-1.0f) * u2);
        data[i] = mean + std * z;
    }
}

static void tensor_fill_xavier(Tensor* t) {
    size_t fan_in = t->ndim > 0 ? t->shape[t->ndim - 1] : 1;
    size_t fan_out = t->ndim > 1 ? t->shape[t->ndim - 2] : 1;
    float std = sqrtf(2.0f / (fan_in + fan_out));
    tensor_fill_randn(t, 0.0f, std);
}

/* ============================================================================
 * Debug
 * ============================================================================ */

static void tensor_print(const Tensor* t, const char* name) {
    printf("%s: shape=[", name ? name : "tensor");
    for (uint8_t i = 0; i < t->ndim; i++) printf("%zu%s", t->shape[i], i < t->ndim-1 ? "," : "");
    printf("], layout=%d, dtype=%d, size=%zu", t->layout, t->dtype, t->size);
    if (t->quant_type != QUANT_NONE) {
        printf(", quant(scale=%.6f, zp=%d)", t->quant.scale, t->quant.zero_point);
    }
    printf("\n");
}

/* ============================================================================
 * Tensor Operations
 * ============================================================================ */

/* --------------------------------------------------------------------------
 * Element-wise Operations
 * C = A op B  (broadcasting supported)
 * -------------------------------------------------------------------------- */

static bool tensor_broadcast_shape(const Tensor* a, const Tensor* b,
                                   size_t* out_shape, uint8_t* out_ndim) {
    // Broadcasting: align dimensions from right, expand size-1 dimensions
    // Rule: each dimension must match or be 1
    uint8_t max_ndim = (a->ndim > b->ndim) ? a->ndim : b->ndim;
    uint8_t offset_a = max_ndim - a->ndim;
    uint8_t offset_b = max_ndim - b->ndim;

    for (uint8_t i = 0; i < max_ndim; i++) {
        size_t dim_a = (i < offset_a) ? 1 : a->shape[i - offset_a];
        size_t dim_b = (i < offset_b) ? 1 : b->shape[i - offset_b];
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return false;  // Cannot broadcast
        }
        out_shape[i] = (dim_a > dim_b) ? dim_a : dim_b;
    }
    *out_ndim = max_ndim;
    return true;
}

static size_t tensor_get_broadcast_index(const Tensor* t, const size_t* indices,
                                         uint8_t offset, uint8_t t_ndim) {
    // Compute linear index accounting for broadcasting (size-1 dims)
    size_t idx = 0;
    for (uint8_t i = 0; i < t_ndim; i++) {
        // If this tensor has size 1 at this dim, use index 0 regardless of actual index
        size_t effective_idx = (t->shape[i] == 1) ? 0 : indices[i + offset];
        idx += effective_idx * t->strides[i];
    }
    return idx;
}

static Tensor* tensor_add(const Tensor* a, const Tensor* b) {
    // Broadcasting: expand dimensions of size 1
    // Rule: compare from rightmost dimension
    size_t out_shape[8];
    uint8_t out_ndim;
    if (!tensor_broadcast_shape(a, b, out_shape, &out_ndim)) return NULL;

    Tensor* c = tensor_create(a->dtype, a->layout, out_shape, out_ndim);
    float* ca = (float*)c->data;
    uint8_t offset_a = out_ndim - a->ndim;
    uint8_t offset_b = out_ndim - b->ndim;

    // Iterate over output indices
    size_t total = c->size;
    size_t idx_buf[8] = {0};

    for (size_t i = 0; i < total; i++) {
        // Decode linear index to multi-dimensional indices
        size_t temp = i;
        for (uint8_t j = out_ndim; j > 0; j--) {
            idx_buf[j-1] = temp % out_shape[j-1];
            temp /= out_shape[j-1];
        }

        size_t idx_a = tensor_get_broadcast_index(a, idx_buf, offset_a, a->ndim);
        size_t idx_b = tensor_get_broadcast_index(b, idx_buf, offset_b, b->ndim);
        ca[i] = ((float*)a->data)[idx_a] + ((float*)b->data)[idx_b];
    }
    return c;
}

static Tensor* tensor_sub(const Tensor* a, const Tensor* b) {
    size_t out_shape[8];
    uint8_t out_ndim;
    if (!tensor_broadcast_shape(a, b, out_shape, &out_ndim)) return NULL;

    Tensor* c = tensor_create(a->dtype, a->layout, out_shape, out_ndim);
    float* ca = (float*)c->data;
    uint8_t offset_a = out_ndim - a->ndim;
    uint8_t offset_b = out_ndim - b->ndim;

    size_t total = c->size;
    size_t idx_buf[8] = {0};

    for (size_t i = 0; i < total; i++) {
        size_t temp = i;
        for (uint8_t j = out_ndim; j > 0; j--) {
            idx_buf[j-1] = temp % out_shape[j-1];
            temp /= out_shape[j-1];
        }

        size_t idx_a = tensor_get_broadcast_index(a, idx_buf, offset_a, a->ndim);
        size_t idx_b = tensor_get_broadcast_index(b, idx_buf, offset_b, b->ndim);
        ca[i] = ((float*)a->data)[idx_a] - ((float*)b->data)[idx_b];
    }
    return c;
}

static Tensor* tensor_mul(const Tensor* a, const Tensor* b) {
    size_t out_shape[8];
    uint8_t out_ndim;
    if (!tensor_broadcast_shape(a, b, out_shape, &out_ndim)) return NULL;

    Tensor* c = tensor_create(a->dtype, a->layout, out_shape, out_ndim);
    float* ca = (float*)c->data;
    uint8_t offset_a = out_ndim - a->ndim;
    uint8_t offset_b = out_ndim - b->ndim;

    size_t total = c->size;
    size_t idx_buf[8] = {0};

    for (size_t i = 0; i < total; i++) {
        size_t temp = i;
        for (uint8_t j = out_ndim; j > 0; j--) {
            idx_buf[j-1] = temp % out_shape[j-1];
            temp /= out_shape[j-1];
        }

        size_t idx_a = tensor_get_broadcast_index(a, idx_buf, offset_a, a->ndim);
        size_t idx_b = tensor_get_broadcast_index(b, idx_buf, offset_b, b->ndim);
        ca[i] = ((float*)a->data)[idx_a] * ((float*)b->data)[idx_b];
    }
    return c;
}

static Tensor* tensor_div(const Tensor* a, const Tensor* b) {
    size_t out_shape[8];
    uint8_t out_ndim;
    if (!tensor_broadcast_shape(a, b, out_shape, &out_ndim)) return NULL;

    Tensor* c = tensor_create(a->dtype, a->layout, out_shape, out_ndim);
    float* ca = (float*)c->data;
    uint8_t offset_a = out_ndim - a->ndim;
    uint8_t offset_b = out_ndim - b->ndim;

    size_t total = c->size;
    size_t idx_buf[8] = {0};

    for (size_t i = 0; i < total; i++) {
        size_t temp = i;
        for (uint8_t j = out_ndim; j > 0; j--) {
            idx_buf[j-1] = temp % out_shape[j-1];
            temp /= out_shape[j-1];
        }

        size_t idx_a = tensor_get_broadcast_index(a, idx_buf, offset_a, a->ndim);
        size_t idx_b = tensor_get_broadcast_index(b, idx_buf, offset_b, b->ndim);
        ca[i] = ((float*)a->data)[idx_a] / (((float*)b->data)[idx_b] + 1e-8f);
    }
    return c;
}

static void tensor_add_inplace(Tensor* a, const Tensor* b) {
    size_t out_shape[8];
    uint8_t out_ndim;
    if (!tensor_broadcast_shape(a, b, out_shape, &out_ndim)) return;

    // Check if shapes are compatible (a must be broadcastable to output, and output == a shape)
    uint8_t offset_b = out_ndim - b->ndim;
    size_t idx_buf[8] = {0};

    for (size_t i = 0; i < a->size; i++) {
        size_t temp = i;
        for (uint8_t j = a->ndim; j > 0; j--) {
            idx_buf[j-1] = temp % a->shape[j-1];
            temp /= a->shape[j-1];
        }

        // Compute corresponding output index (prepend 0s for broadcast)
        size_t out_idx_buf[8] = {0};
        for (uint8_t j = 0; j < a->ndim; j++) {
            out_idx_buf[offset_b + j] = idx_buf[j];
        }

        size_t idx_b = tensor_get_broadcast_index(b, out_idx_buf, offset_b, b->ndim);
        ((float*)a->data)[i] += ((float*)b->data)[idx_b];
    }
}

static void tensor_scale(Tensor* t, float scalar) {
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->size; i++) data[i] *= scalar;
}

/* --------------------------------------------------------------------------
 * Matrix Multiplication (GEMM)
 * C = alpha * A @ B + beta * C
 * A[MxK], B[KxN], C[MxN]
 *
 * Cache-optimal blocking: tile for L1 cache
 * L1_BLOCK = 16 -> 16x16 floats = 1KB per block (fits in 32KB L1)
 * -------------------------------------------------------------------------- */

#define TENSOR_GEMM_BLOCK 16

static void tensor_gemm(float* C, const float* A, const float* B,
                        size_t M, size_t N, size_t K,
                        float alpha, float beta) {
    // Initialize C with beta * C
    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++)
            C[i*N + j] *= beta;

    // Blocked matrix multiplication for cache efficiency
    for (size_t i = 0; i < M; i += TENSOR_GEMM_BLOCK) {
        for (size_t k = 0; k < K; k += TENSOR_GEMM_BLOCK) {
            // Compute A[i:i+ib, k:k+kb] @ B[k:k+kb, :]
            size_t ib = (i + TENSOR_GEMM_BLOCK < M) ? TENSOR_GEMM_BLOCK : M - i;
            size_t kb = (k + TENSOR_GEMM_BLOCK < K) ? TENSOR_GEMM_BLOCK : K - k;

            for (size_t ii = 0; ii < ib; ii++) {
                for (size_t kk = 0; kk < kb; kk++) {
                    float a_ik = A[(i + ii) * K + (k + kk)];
                    const float* B_row = B + (k + kk) * N;
                    float* C_row = C + (i + ii) * N;
                    for (size_t j = 0; j < N; j++) {
                        C_row[j] += alpha * a_ik * B_row[j];
                    }
                }
            }
        }
    }
}

static Tensor* tensor_matmul(const Tensor* a, const Tensor* b) {
    // Matrix multiply: [M,K] @ [K,N] -> [M,N]
    // Or batched: [B,M,K] @ [B,K,N] -> [B,M,N]
    // Also supports broadcasting: [B,M,K] @ [K,N] or [M,K] @ [B,K,N]
    if (a->ndim < 2 || b->ndim < 2) return NULL;

    // Determine if we have batch dimensions
    bool a_has_batch = (a->ndim > 2);
    bool b_has_batch = (b->ndim > 2);

    size_t M, K, N, B;

    if (!a_has_batch && !b_has_batch) {
        // Simple matmul: [M,K] @ [K,N] -> [M,N]
        M = a->shape[0]; K = a->shape[1];
        N = b->shape[1];
        if (b->shape[0] != K) return NULL;
        B = 1;
        size_t shape[] = {M, N};
        Tensor* c = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 2);
        tensor_gemm((float*)c->data, (float*)a->data, (float*)b->data,
                    M, N, K, 1.0f, 0.0f);
        return c;
    } else if (a_has_batch && !b_has_batch) {
        // [B,M,K] @ [K,N] -> [B,M,N]
        B = a->shape[0]; M = a->shape[1]; K = a->shape[2];
        N = b->shape[1];
        if (b->shape[0] != K) return NULL;
        size_t out_shape[] = {B, M, N};
        Tensor* c = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, out_shape, 3);
        float* c_data = (float*)c->data;
        float* a_data = (float*)a->data;
        float* b_data = (float*)b->data;
        for (size_t batch = 0; batch < B; batch++) {
            tensor_gemm(c_data + batch * M * N,
                        a_data + batch * M * K,
                        b_data, M, N, K, 1.0f, 0.0f);
        }
        return c;
    } else if (!a_has_batch && b_has_batch) {
        // [M,K] @ [B,K,N] -> [B,M,N]
        M = a->shape[0]; K = a->shape[1];
        B = b->shape[0]; N = b->shape[2];
        if (b->shape[1] != K) return NULL;
        size_t out_shape[] = {B, M, N};
        Tensor* c = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, out_shape, 3);
        float* c_data = (float*)c->data;
        float* a_data = (float*)a->data;
        float* b_data = (float*)b->data;
        for (size_t batch = 0; batch < B; batch++) {
            tensor_gemm(c_data + batch * M * N,
                        a_data,
                        b_data + batch * K * N, M, N, K, 1.0f, 0.0f);
        }
        return c;
    } else {
        // Both have batch: [B,M,K] @ [B,K,N] -> [B,M,N]
        if (a->ndim != 3 || b->ndim != 3) return NULL;
        B = a->shape[0]; M = a->shape[1]; K = a->shape[2];
        if (b->shape[0] != B || b->shape[1] != K) return NULL;
        N = b->shape[2];
        size_t out_shape[] = {B, M, N};
        Tensor* c = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, out_shape, 3);
        float* c_data = (float*)c->data;
        float* a_data = (float*)a->data;
        float* b_data = (float*)b->data;
        for (size_t batch = 0; batch < B; batch++) {
            tensor_gemm(c_data + batch * M * N,
                        a_data + batch * M * K,
                        b_data + batch * K * N, M, N, K, 1.0f, 0.0f);
        }
        return c;
    }
}

/* --------------------------------------------------------------------------
 * Reduction Operations
 * C = reduce(op, A, axis)
 * -------------------------------------------------------------------------- */

static Tensor* tensor_sum(const Tensor* t, size_t axis) {
    if (axis >= t->ndim) return NULL;

    size_t* new_shape = (size_t*)malloc(sizeof(size_t) * (t->ndim - 1));
    size_t new_ndim = 0;
    for (size_t i = 0; i < t->ndim; i++) {
        if (i != axis) new_shape[new_ndim++] = t->shape[i];
    }

    Tensor* result = tensor_create(t->dtype, t->layout, new_shape, new_ndim);
    float* dst = (float*)result->data;
    float* src = (float*)t->data;

    // outer_size = product of all dims except 'axis'
    size_t outer_size = 1;
    for (size_t i = 0; i < t->ndim; i++) {
        if (i != axis) outer_size *= t->shape[i];
    }

    // For each output position, iterate over axis and accumulate
    size_t stride = t->strides[axis];
    size_t block = t->shape[axis];

    for (size_t outer = 0; outer < outer_size; outer++) {
        float sum = 0.0f;
        // Compute base index for this output position
        // outer encodes indices for dims < axis and > axis combined
        size_t base = 0;
        size_t temp = outer;
        for (size_t d = 0; d < t->ndim; d++) {
            if (d == axis) continue;
            size_t dim_size = (d == 0) ? t->strides[0] : t->strides[d-1] / t->strides[d];
            size_t idx = temp % dim_size;
            temp /= dim_size;
            base += idx * t->strides[d];
        }
        for (size_t i = 0; i < block; i++) {
            sum += src[base + i * stride];
        }
        dst[outer] = sum;
    }

    free(new_shape);
    return result;
}

static Tensor* tensor_mean(const Tensor* t, size_t axis) {
    if (axis >= t->ndim) return NULL;
    Tensor* result = tensor_sum(t, axis);
    float* data = (float*)result->data;
    size_t n = t->shape[axis];
    for (size_t i = 0; i < result->size; i++) data[i] /= n;
    return result;
}

static Tensor* tensor_max(const Tensor* t, size_t axis) {
    if (axis >= t->ndim) return NULL;

    size_t* new_shape = (size_t*)malloc(sizeof(size_t) * (t->ndim - 1));
    size_t new_ndim = 0;
    for (size_t i = 0; i < t->ndim; i++) {
        if (i != axis) new_shape[new_ndim++] = t->shape[i];
    }

    Tensor* result = tensor_create(t->dtype, t->layout, new_shape, new_ndim);
    float* dst = (float*)result->data;
    float* src = (float*)t->data;

    size_t outer_size = 1;
    for (size_t i = 0; i < t->ndim; i++) {
        if (i != axis) outer_size *= t->shape[i];
    }

    size_t stride = t->strides[axis];
    size_t block = t->shape[axis];

    for (size_t outer = 0; outer < outer_size; outer++) {
        float max_val = -FLT_MAX;
        size_t base = 0;
        size_t temp = outer;
        for (size_t d = 0; d < t->ndim; d++) {
            if (d == axis) continue;
            size_t dim_size = (d == 0) ? t->strides[0] : t->strides[d-1] / t->strides[d];
            size_t idx = temp % dim_size;
            temp /= dim_size;
            base += idx * t->strides[d];
        }
        for (size_t i = 0; i < block; i++) {
            float v = src[base + i * stride];
            if (v > max_val) max_val = v;
        }
        dst[outer] = max_val;
    }
    free(new_shape);
    return result;
}

/* --------------------------------------------------------------------------
 * Activation Functions
 * In-place operations
 * -------------------------------------------------------------------------- */

static void tensor_relu(Tensor* t) {
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->size; i++) if (data[i] < 0) data[i] = 0;
}

static void tensor_sigmoid(Tensor* t) {
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->size; i++) data[i] = 1.0f / (1.0f + expf(-data[i]));
}

static void tensor_tanh(Tensor* t) {
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->size; i++) data[i] = tanhf(data[i]);
}

static void tensor_softmax(Tensor* t, size_t axis) {
    float* data = (float*)t->data;
    size_t stride = t->strides[axis];
    size_t block = t->shape[axis];
    size_t blocks = t->size / block;

    for (size_t b = 0; b < blocks; b++) {
        size_t base = b * block * stride;
        // Find max for numerical stability
        float max_val = data[base];
        for (size_t i = 1; i < block; i++) {
            float v = data[base + i * stride];
            if (v > max_val) max_val = v;
        }
        // Exp and sum
        float sum = 0.0f;
        for (size_t i = 0; i < block; i++) {
            data[base + i * stride] = expf(data[base + i * stride] - max_val);
            sum += data[base + i * stride];
        }
        // Normalize
        for (size_t i = 0; i < block; i++) {
            data[base + i * stride] /= sum;
        }
    }
}

/* --------------------------------------------------------------------------
 * 2D Convolution (Im2Col approach)
 * Input: [N, C, H, W]
 * Weight: [OutC, C, KH, KW]
 * Output: [N, OutC, H', W']
 * H' = (H - KH + 2*pad) / stride + 1
 * W' = (W - KW + 2*pad) / stride + 1
 * -------------------------------------------------------------------------- */

typedef struct {
    size_t stride_h;
    size_t stride_w;
    size_t pad_h;
    size_t pad_w;
    size_t dilation_h;
    size_t dilation_w;
} Conv2DParams;

static Tensor* tensor_conv2d(const Tensor* input, const Tensor* weight,
                             const Conv2DParams* params) {
    // input: [N, C, H, W]
    // weight: [OutC, C, KH, KW]
    if (input->ndim != 4 || weight->ndim != 4) return NULL;

    size_t N = input->shape[0];
    size_t C = input->shape[1];
    size_t H = input->shape[2];
    size_t W = input->shape[3];
    size_t OutC = weight->shape[0];
    size_t KH = weight->shape[2];
    size_t KW = weight->shape[3];

    size_t stride = params ? params->stride_h : 1;
    size_t pad = params ? params->pad_h : 0;

    size_t H_out = (H + 2 * pad - KH) / stride + 1;
    size_t W_out = (W + 2 * pad - KW) / stride + 1;

    size_t out_shape[] = {N, OutC, H_out, W_out};
    Tensor* output = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, out_shape, 4);

    float* out_data = (float*)output->data;
    float* in_data = (float*)input->data;
    float* w_data = (float*)weight->data;

    // Im2Col: flatten each patch into a column
    // Col shape: [KH * KW * C, H_out * W_out]
    size_t col_h = KH * KW * C;
    size_t col_w = H_out * W_out;
    float* col = (float*)malloc(sizeof(float) * col_h * col_w);

    for (size_t n = 0; n < N; n++) {
        // Im2Col for this sample
        for (size_t h = 0; h < H_out; h++) {
            for (size_t w = 0; w < W_out; w++) {
                size_t col_idx = h * W_out + w;
                size_t in_h_base = h * stride;
                size_t in_w_base = w * stride;
                size_t idx = 0;
                for (size_t kh = 0; kh < KH; kh++) {
                    for (size_t kw = 0; kw < KW; kw++) {
                        for (size_t c = 0; c < C; c++) {
                            size_t src_h = in_h_base + kh - pad;
                            size_t src_w = in_w_base + kw - pad;
                            if (src_h < H && src_w < W && src_h >= 0 && src_w >= 0) {
                                col[idx * col_w + col_idx] = in_data[n*C*H*W + c*H*W + src_h*W + src_w];
                            } else {
                                col[idx * col_w + col_idx] = 0.0f;
                            }
                            idx++;
                        }
                    }
                }
            }
        }

        // GEMM: [OutC, KH*KW*C] @ [KH*KW*C, H_out*W_out] -> [OutC, H_out*W_out]
        for (size_t oc = 0; oc < OutC; oc++) {
            for (size_t hw = 0; hw < H_out * W_out; hw++) {
                float sum = 0.0f;
                for (size_t c = 0; c < KH * KW * C; c++) {
                    sum += w_data[oc * KH * KW * C + c] * col[c * col_w + hw];
                }
                out_data[n * OutC * H_out * W_out + oc * H_out * W_out + hw] = sum;
            }
        }
    }

    free(col);
    return output;
}

/* --------------------------------------------------------------------------
 * 2D Pooling
 * MaxPool / AvgPool
 * Input: [N, C, H, W]
 * Output: [N, C, H', W']
 * -------------------------------------------------------------------------- */

static Tensor* tensor_maxpool2d(const Tensor* input, size_t pool_h, size_t pool_w,
                                 size_t stride_h, size_t stride_w) {
    if (input->ndim != 4) return NULL;

    size_t N = input->shape[0];
    size_t C = input->shape[1];
    size_t H = input->shape[2];
    size_t W = input->shape[3];

    size_t H_out = (H - pool_h) / stride_h + 1;
    size_t W_out = (W - pool_w) / stride_w + 1;

    size_t out_shape[] = {N, C, H_out, W_out};
    Tensor* output = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, out_shape, 4);

    float* out_data = (float*)output->data;
    float* in_data = (float*)input->data;

    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H_out; h++) {
                for (size_t w = 0; w < W_out; w++) {
                    float max_val = -FLT_MAX;
                    for (size_t ph = 0; ph < pool_h; ph++) {
                        for (size_t pw = 0; pw < pool_w; pw++) {
                            size_t src_h = h * stride_h + ph;
                            size_t src_w = w * stride_w + pw;
                            if (src_h < H && src_w < W) {
                                float val = in_data[n*C*H*W + c*H*W + src_h*W + src_w];
                                if (val > max_val) max_val = val;
                            }
                        }
                    }
                    out_data[n*C*H_out*W_out + c*H_out*W_out + h*W_out + w] = max_val;
                }
            }
        }
    }
    return output;
}

static Tensor* tensor_avgpool2d(const Tensor* input, size_t pool_h, size_t pool_w,
                                size_t stride_h, size_t stride_w) {
    if (input->ndim != 4) return NULL;

    size_t N = input->shape[0];
    size_t C = input->shape[1];
    size_t H = input->shape[2];
    size_t W = input->shape[3];

    size_t H_out = (H - pool_h) / stride_h + 1;
    size_t W_out = (W - pool_w) / stride_w + 1;

    size_t out_shape[] = {N, C, H_out, W_out};
    Tensor* output = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, out_shape, 4);

    float* out_data = (float*)output->data;
    float* in_data = (float*)input->data;
    float inv_n = 1.0f / (pool_h * pool_w);

    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H_out; h++) {
                for (size_t w = 0; w < W_out; w++) {
                    float sum = 0.0f;
                    for (size_t ph = 0; ph < pool_h; ph++) {
                        for (size_t pw = 0; pw < pool_w; pw++) {
                            size_t src_h = h * stride_h + ph;
                            size_t src_w = w * stride_w + pw;
                            if (src_h < H && src_w < W) {
                                sum += in_data[n*C*H*W + c*H*W + src_h*W + src_w];
                            }
                        }
                    }
                    out_data[n*C*H_out*W_out + c*H_out*W_out + h*W_out + w] = sum * inv_n;
                }
            }
        }
    }
    return output;
}

/* --------------------------------------------------------------------------
 * Utility Operations
 * -------------------------------------------------------------------------- */

static Tensor* tensor_concat(const Tensor** tensors, size_t n, size_t axis) {
    if (n == 0) return NULL;
    if (axis >= tensors[0]->ndim) return NULL;

    // Check all tensors have same shape except axis
    for (size_t i = 1; i < n; i++) {
        if (tensors[i]->ndim != tensors[0]->ndim) return NULL;
        for (size_t j = 0; j < tensors[0]->ndim; j++) {
            if (j != axis && tensors[i]->shape[j] != tensors[0]->shape[j]) return NULL;
        }
    }

    // Compute output shape
    size_t total_dim = 0;
    for (size_t i = 0; i < n; i++) total_dim += tensors[i]->shape[axis];

    size_t* out_shape = (size_t*)malloc(sizeof(size_t) * tensors[0]->ndim);
    for (size_t i = 0; i < tensors[0]->ndim; i++) {
        out_shape[i] = (i == axis) ? total_dim : tensors[0]->shape[i];
    }

    Tensor* output = tensor_create(tensors[0]->dtype, tensors[0]->layout, out_shape, tensors[0]->ndim);
    float* dst = (float*)output->data;

    size_t offset = 0;
    size_t stride = tensors[0]->strides[axis];
    for (size_t t_idx = 0; t_idx < n; t_idx++) {
        float* src = (float*)tensors[t_idx]->data;
        size_t dim_size = tensors[t_idx]->shape[axis];

        for (size_t i = 0; i < tensors[t_idx]->size; i++) {
            size_t src_idx = i;
            size_t dim_idx = (src_idx / stride) % dim_size;
            size_t flat_idx = (src_idx % stride) + dim_idx * stride + offset * stride;
            dst[flat_idx] = src[i];
        }
        offset += dim_size;
    }

    free(out_shape);
    return output;
}

static Tensor* tensor_pad(const Tensor* t, size_t pad_h, size_t pad_w, float value) {
    size_t new_shape[] = {t->shape[0] + 2*pad_h, t->shape[1] + 2*pad_w};
    Tensor* padded = tensor_create(t->dtype, t->layout, new_shape, 2);
    tensor_fill_f32(padded, value);

    float* dst = (float*)padded->data;
    float* src = (float*)t->data;

    for (size_t i = 0; i < t->shape[0]; i++) {
        for (size_t j = 0; j < t->shape[1]; j++) {
            dst[(i + pad_h) * new_shape[1] + (j + pad_w)] = src[i * t->shape[1] + j];
        }
    }
    return padded;
}

static void tensor_clip(Tensor* t, float min_val, float max_val) {
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->size; i++) {
        if (data[i] < min_val) data[i] = min_val;
        if (data[i] > max_val) data[i] = max_val;
    }
}

/* ============================================================================
 * Quantized Tensor Operations
 * ============================================================================ */

/**
 * Quantized GEMM for INT8 inference
 * C = A @ B with quantization
 * A[M x K], B[K x N], C[M x N]
 *
 * Uses symmetric quantization (zero_point = 0) for simplicity
 */
static void tensor_gemm_quantized(int32_t* C, const int8_t* A, const int8_t* B,
                                  float scale_A, float scale_B,
                                  size_t M, size_t N, size_t K) {
    // Clear output
    for (size_t i = 0; i < M * N; i++) C[i] = 0;

    // Blocked computation for cache efficiency
    for (size_t i = 0; i < M; i++) {
        for (size_t k = 0; k < K; k++) {
            int32_t a_val = (int32_t)A[i * K + k];
            // Process multiple B elements per iteration (SIMD-friendly)
            for (size_t j = 0; j < N; j++) {
                int32_t b_val = (int32_t)B[k * N + j];
                C[i * N + j] += a_val * b_val;
            }
        }
    }
    // Scale result
    float scale = scale_A * scale_B;
    for (size_t i = 0; i < M * N; i++) {
        // Convert from integer accumulation back to float and scale
        C[i] = (int32_t)(C[i] * scale);
    }
}

/**
 * Quantized matrix multiplication wrapper
 * Input and output are quantized tensors
 * Supports: [M,K] @ [K,N] -> [M,N]
 *           [B,M,K] @ [B,K,N] -> [B,M,N]
 *           [B,M,K] @ [K,N] -> [B,M,N]
 */
static Tensor* tensor_matmul_quantized(const Tensor* a, const Tensor* b,
                                       float scale_A, float scale_B,
                                       TensorDType out_dtype) {
    if (a->ndim < 2 || b->ndim < 2) return NULL;

    bool a_has_batch = (a->ndim > 2);
    bool b_has_batch = (b->ndim > 2);

    size_t M, K, N, B;

    if (!a_has_batch && !b_has_batch) {
        // Simple: [M,K] @ [K,N] -> [M,N]
        M = a->shape[0]; K = a->shape[1];
        N = b->shape[1];
        if (b->shape[0] != K) return NULL;
        size_t out_shape[] = {M, N};
        Tensor* c = tensor_create(out_dtype, a->layout, out_shape, 2);
        tensor_gemm_quantized((int32_t*)c->data,
                              (const int8_t*)a->data,
                              (const int8_t*)b->data,
                              scale_A, scale_B, M, N, K);
        return c;
    } else if (a_has_batch && !b_has_batch) {
        // [B,M,K] @ [K,N] -> [B,M,N]
        B = a->shape[0]; M = a->shape[1]; K = a->shape[2];
        N = b->shape[1];
        if (b->shape[0] != K) return NULL;
        size_t out_shape[] = {B, M, N};
        Tensor* c = tensor_create(out_dtype, a->layout, out_shape, 3);
        int32_t* c_data = (int32_t*)c->data;
        const int8_t* a_data = (const int8_t*)a->data;
        const int8_t* b_data = (const int8_t*)b->data;
        for (size_t batch = 0; batch < B; batch++) {
            tensor_gemm_quantized(c_data + batch * M * N,
                                  a_data + batch * M * K,
                                  b_data,
                                  scale_A, scale_B, M, N, K);
        }
        return c;
    } else if (!a_has_batch && b_has_batch) {
        // [M,K] @ [B,K,N] -> [B,M,N]
        M = a->shape[0]; K = a->shape[1];
        B = b->shape[0]; N = b->shape[2];
        if (b->shape[1] != K) return NULL;
        size_t out_shape[] = {B, M, N};
        Tensor* c = tensor_create(out_dtype, a->layout, out_shape, 3);
        int32_t* c_data = (int32_t*)c->data;
        const int8_t* a_data = (const int8_t*)a->data;
        const int8_t* b_data = (const int8_t*)b->data;
        for (size_t batch = 0; batch < B; batch++) {
            tensor_gemm_quantized(c_data + batch * M * N,
                                  a_data,
                                  b_data + batch * K * N,
                                  scale_A, scale_B, M, N, K);
        }
        return c;
    } else {
        // Both have batch: [B,M,K] @ [B,K,N] -> [B,M,N]
        if (a->ndim != 3 || b->ndim != 3) return NULL;
        B = a->shape[0]; M = a->shape[1]; K = a->shape[2];
        if (b->shape[0] != B || b->shape[1] != K) return NULL;
        N = b->shape[2];
        size_t out_shape[] = {B, M, N};
        Tensor* c = tensor_create(out_dtype, a->layout, out_shape, 3);
        int32_t* c_data = (int32_t*)c->data;
        const int8_t* a_data = (const int8_t*)a->data;
        const int8_t* b_data = (const int8_t*)b->data;
        for (size_t batch = 0; batch < B; batch++) {
            tensor_gemm_quantized(c_data + batch * M * N,
                                  a_data + batch * M * K,
                                  b_data + batch * K * N,
                                  scale_A, scale_B, M, N, K);
        }
        return c;
    }
}

/**
 * Quantized ReLU - threshold at zero
 * For INT8: values < 0 become 0
 */
static void tensor_relu_quantized(Tensor* t) {
    int8_t* data = (int8_t*)t->data;
    for (size_t i = 0; i < t->size; i++) {
        if (data[i] < 0) data[i] = 0;
    }
}

/**
 * Quantized ReLU6 - threshold at 0 and 6
 * For INT8: values < 0 become 0, values > 6 become 6
 * Requires zero_point adjustment
 */
static void tensor_relu6_quantized(Tensor* t, float scale, int8_t zero_point) {
    int8_t* data = (int8_t*)t->data;
    int8_t min_val = -zero_point;  // 0 in floating point maps to -zero_point in quantized
    int8_t max_val = (int8_t)((6.0f / scale) + zero_point);
    if (max_val > 127) max_val = 127;

    for (size_t i = 0; i < t->size; i++) {
        if (data[i] < min_val) data[i] = min_val;
        if (data[i] > max_val) data[i] = max_val;
    }
}

/**
 * Quantized activation - clamp to range
 */
static void tensor_clamp_quantized(Tensor* t, float min_val, float max_val,
                                    float scale, int8_t zero_point) {
    int8_t* data = (int8_t*)t->data;
    int8_t qmin = (int8_t)((min_val / scale) + zero_point);
    int8_t qmax = (int8_t)((max_val / scale) + zero_point);

    for (size_t i = 0; i < t->size; i++) {
        if (data[i] < qmin) data[i] = qmin;
        if (data[i] > qmax) data[i] = qmax;
    }
}

/**
 * Quantized element-wise add with broadcasting
 * C = A + B, requires same quantization parameters
 */
static void tensor_add_quantized(Tensor* c, const Tensor* a, const Tensor* b) {
    if (a->quant_type != b->quant_type) return;
    if (c->quant_type != a->quant_type) return;

    int8_t* ca = (int8_t*)c->data;
    int8_t* aa = (int8_t*)a->data;
    int8_t* ba = (int8_t*)b->data;

    // Add requires dequantization due to possible overflow
    // Simplified: assume no overflow (for small values)
    for (size_t i = 0; i < c->size; i++) {
        int32_t sum = (int32_t)aa[i] + (int32_t)ba[i];
        if (sum > 127) sum = 127;
        if (sum < -128) sum = -128;
        ca[i] = (int8_t)sum;
    }
}

/**
 * Quantized element-wise multiply
 * C = A * B (element-wise)
 */
static void tensor_mul_quantized(Tensor* c, const Tensor* a, const Tensor* b) {
    if (a->quant_type != b->quant_type) return;

    int8_t* ca = (int8_t*)c->data;
    int8_t* aa = (int8_t*)a->data;
    int8_t* ba = (int8_t*)b->data;

    for (size_t i = 0; i < c->size; i++) {
        int32_t prod = (int32_t)aa[i] * (int32_t)ba[i];
        // Quantize back to int8
        int32_t qprod = prod / 128;  // scale factor for symmetric quantization
        if (qprod > 127) qprod = 127;
        if (qprod < -128) qprod = -128;
        ca[i] = (int8_t)qprod;
    }
}

/**
 * Quantized Softmax
 * Requires dequantization, computation in float, requantization
 */
static void tensor_softmax_quantized(Tensor* t, size_t axis,
                                     float scale, int8_t zero_point) {
    // Dequantize
    float* temp = (float*)malloc(sizeof(float) * t->size);
    int8_t* data = (int8_t*)t->data;
    for (size_t i = 0; i < t->size; i++) {
        temp[i] = ((float)data[i] - zero_point) * scale;
    }

    // Compute softmax in float
    size_t stride = t->strides[axis];
    size_t block = t->shape[axis];
    size_t blocks = t->size / block;

    for (size_t b = 0; b < blocks; b++) {
        size_t base = b * block * stride;
        // Find max
        float max_val = temp[base];
        for (size_t i = 1; i < block; i++) {
            if (temp[base + i * stride] > max_val)
                max_val = temp[base + i * stride];
        }
        // Exp and sum
        float sum = 0.0f;
        for (size_t i = 0; i < block; i++) {
            temp[base + i * stride] = expf(temp[base + i * stride] - max_val);
            sum += temp[base + i * stride];
        }
        // Normalize and requantize
        for (size_t i = 0; i < block; i++) {
            float val = temp[base + i * stride] / sum;
            int32_t qval = (int32_t)(val / scale + zero_point);
            if (qval < -128) qval = -128;
            if (qval > 127) qval = 127;
            data[base + i * stride] = (int8_t)qval;
        }
    }
    free(temp);
}

/**
 * Quantized Average Pooling
 * Input: INT8 quantized, Output: INT8 quantized
 */
static Tensor* tensor_avgpool2d_quantized(const Tensor* input, size_t pool_h, size_t pool_w,
                                          size_t stride_h, size_t stride_w) {
    if (input->ndim != 4) return NULL;

    size_t N = input->shape[0];
    size_t C = input->shape[1];
    size_t H = input->shape[2];
    size_t W = input->shape[3];

    size_t H_out = (H - pool_h) / stride_h + 1;
    size_t W_out = (W - pool_w) / stride_w + 1;

    size_t out_shape[] = {N, C, H_out, W_out};
    Tensor* output = tensor_create(TENSOR_DTYPE_INT8, input->layout, out_shape, 4);
    output->quant_type = input->quant_type;
    output->quant = input->quant;

    int8_t* out_data = (int8_t*)output->data;
    int8_t* in_data = (int8_t*)input->data;
    int32_t pool_size = (int32_t)(pool_h * pool_w);
    float inv_pool = 1.0f / (float)pool_size;

    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H_out; h++) {
                for (size_t w = 0; w < W_out; w++) {
                    int32_t sum = 0;
                    for (size_t ph = 0; ph < pool_h; ph++) {
                        for (size_t pw = 0; pw < pool_w; pw++) {
                            size_t src_h = h * stride_h + ph;
                            size_t src_w = w * stride_w + pw;
                            if (src_h < H && src_w < W) {
                                sum += in_data[n*C*H*W + c*H*W + src_h*W + src_w];
                            }
                        }
                    }
                    // Dequantize, compute mean, requantize
                    float deq_sum = (sum - output->quant.zero_point * pool_size) * output->quant.scale;
                    float mean = deq_sum * inv_pool;
                    int32_t qval = (int32_t)(mean / output->quant.scale + output->quant.zero_point);
                    if (qval < -128) qval = -128;
                    if (qval > 127) qval = 127;
                    out_data[n*C*H_out*W_out + c*H_out*W_out + h*W_out + w] = (int8_t)qval;
                }
            }
        }
    }
    return output;
}

/**
 * Quantized Max Pooling
 * Input: INT8 quantized, Output: INT8 quantized (same scale/zero_point)
 */
static Tensor* tensor_maxpool2d_quantized(const Tensor* input, size_t pool_h, size_t pool_w,
                                          size_t stride_h, size_t stride_w) {
    if (input->ndim != 4) return NULL;

    size_t N = input->shape[0];
    size_t C = input->shape[1];
    size_t H = input->shape[2];
    size_t W = input->shape[3];

    size_t H_out = (H - pool_h) / stride_h + 1;
    size_t W_out = (W - pool_w) / stride_w + 1;

    size_t out_shape[] = {N, C, H_out, W_out};
    Tensor* output = tensor_create(TENSOR_DTYPE_INT8, input->layout, out_shape, 4);
    output->quant_type = input->quant_type;
    output->quant = input->quant;

    int8_t* out_data = (int8_t*)output->data;
    int8_t* in_data = (int8_t*)input->data;

    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H_out; h++) {
                for (size_t w = 0; w < W_out; w++) {
                    int8_t max_val = -128;
                    for (size_t ph = 0; ph < pool_h; ph++) {
                        for (size_t pw = 0; pw < pool_w; pw++) {
                            size_t src_h = h * stride_h + ph;
                            size_t src_w = w * stride_w + pw;
                            if (src_h < H && src_w < W) {
                                int8_t val = in_data[n*C*H*W + c*H*W + src_h*W + src_w];
                                if (val > max_val) max_val = val;
                            }
                        }
                    }
                    out_data[n*C*H_out*W_out + c*H_out*W_out + h*W_out + w] = max_val;
                }
            }
        }
    }
    return output;
}

/**
 * Requantize: convert from one quantization to another
 */
static void tensor_requantize(const Tensor* src, Tensor* dst,
                               float new_scale, int8_t new_zero_point) {
    if (src->dtype != dst->dtype) return;

    int8_t* src_data = (int8_t*)src->data;
    int8_t* dst_data = (int8_t*)dst->data;

    for (size_t i = 0; i < src->size; i++) {
        // Dequantize using old params
        float val = ((float)src_data[i] - src->quant.zero_point) * src->quant.scale;
        // Requantize using new params
        int32_t qval = (int32_t)(val / new_scale + new_zero_point);
        if (qval < -128) qval = -128;
        if (qval > 127) qval = 127;
        dst_data[i] = (int8_t)qval;
    }

    dst->quant.scale = new_scale;
    dst->quant.zero_point = new_zero_point;
}

/**
 * Quantization-aware batch normalization
 * Fuse BN params into conv weights for inference
 */
static void tensor_fuse_batch_norm(int8_t* weights, float* bias,
                                    const float* mean, const float* var,
                                    const float* gamma, const float* beta,
                                    float eps, size_t channels) {
    for (size_t c = 0; c < channels; c++) {
        float std = sqrtf(var[c] + eps);
        float scale = gamma[c] / std;

        // Fuse scale into weights
        for (size_t i = 0; i < 3 * 3; i++) {  // Assuming 3x3 kernel
            weights[c * 9 + i] = (int8_t)(weights[c * 9 + i] * scale);
        }

        // Fuse scale and shift into bias
        if (bias) {
            bias[c] = scale * (bias[c] - mean[c]) + beta[c];
        }
    }
}

#endif /* __C_TENSOR_H__ */
