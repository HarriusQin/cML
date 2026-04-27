/**
 * @file dl_tensor.c
 * @brief Tensor operations implementation
 */

#define DL_TENSOR_IMPLEMENTATION
#include "dl_tensor.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * INTERNAL HELPER FUNCTIONS
 * ============================================================================ */

/**
 * @brief Calculate stride array from shape array (row-major order)
 */
static void calculate_stride(size_t *stride, const size_t *shape, size_t ndim) {
    stride[ndim - 1] = 1;
    for (size_t i = ndim - 1; i > 0; i--) {
        stride[i - 1] = stride[i] * shape[i];
    }
}

/**
 * @brief Calculate total size from shape
 */
static size_t calc_total_size(const size_t *shape, size_t ndim) {
    size_t total = 1;
    for (size_t i = 0; i < ndim; i++) {
        total *= shape[i];
    }
    return total;
}

/**
 * @brief Compare shapes
 */
static bool shapes_equal(const size_t *a, const size_t *b, size_t ndim) {
    for (size_t i = 0; i < ndim; i++) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

/**
 * @brief Xorshift random number generator for consistent random values
 */
static uint64_t xorshift_state = 123456789ULL;

static double xorshift_rand(void) {
    uint64_t x = xorshift_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    xorshift_state = x;
    return (double)x / (double)UINT64_MAX;
}

static double rand_normal(void) {
    double u1 = xorshift_rand();
    double u2 = xorshift_rand();
    return sqrt(-2.0 * log(u1 + 1e-10)) * cos(2.0 * M_PI * u2);
}

/* ============================================================================
 * TENSOR CREATION
 * ============================================================================ */

Tensor *tensor_create(size_t ndim, const size_t *shape, DL_Device_t device, bool requires_grad) {
    if (ndim == 0 || !shape) return NULL;

    Tensor *t = DL_CALLOC(Tensor, 1);
    if (!t) return NULL;

    t->ndim = ndim;
    t->shape = DL_MALLOC(size_t, ndim);
    t->stride = DL_MALLOC(size_t, ndim);
    if (!t->shape || !t->stride) {
        free(t->shape);
        free(t->stride);
        free(t);
        return NULL;
    }

    memcpy(t->shape, shape, ndim * sizeof(size_t));
    calculate_stride(t->stride, shape, ndim);
    t->size = calc_total_size(shape, ndim);
    t->device = device;
    t->requires_grad = requires_grad;
    t->grad = NULL;
    t->ctx = NULL;

    t->data = DL_CALLOC(double, t->size);
    if (!t->data) {
        free(t->shape);
        free(t->stride);
        free(t);
        return NULL;
    }

    return t;
}

Tensor *tensor_zeros(size_t ndim, const size_t *shape, DL_Device_t device, bool requires_grad) {
    Tensor *t = tensor_create(ndim, shape, device, requires_grad);
    return t;  /* calloc already zeros */
}

Tensor *tensor_ones(size_t ndim, const size_t *shape, DL_Device_t device, bool requires_grad) {
    Tensor *t = tensor_create(ndim, shape, device, requires_grad);
    if (t) {
        for (size_t i = 0; i < t->size; i++) {
            t->data[i] = 1.0;
        }
    }
    return t;
}

Tensor *tensor_full(size_t ndim, const size_t *shape, double fill_value,
                    DL_Device_t device, bool requires_grad) {
    Tensor *t = tensor_create(ndim, shape, device, requires_grad);
    if (t) {
        for (size_t i = 0; i < t->size; i++) {
            t->data[i] = fill_value;
        }
    }
    return t;
}

Tensor *tensor_from_array_1d(const double *data, size_t size, bool requires_grad) {
    Tensor *t = tensor_create(1, &size, DL_DEVICE_CPU, requires_grad);
    if (t && data) {
        memcpy(t->data, data, size * sizeof(double));
    }
    return t;
}

Tensor *tensor_from_array_2d(const double *data, size_t rows, size_t cols, bool requires_grad) {
    size_t shape[] = {rows, cols};
    Tensor *t = tensor_create(2, shape, DL_DEVICE_CPU, requires_grad);
    if (t && data) {
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                t->data[i * cols + j] = data[i * cols + j];
            }
        }
    }
    return t;
}

Tensor *tensor_randn(size_t ndim, const size_t *shape, double mean, double std, bool requires_grad) {
    Tensor *t = tensor_create(ndim, shape, DL_DEVICE_CPU, requires_grad);
    if (t) {
        for (size_t i = 0; i < t->size; i++) {
            t->data[i] = mean + std * rand_normal();
        }
    }
    return t;
}

Tensor *tensor_rand(size_t ndim, const size_t *shape, bool requires_grad) {
    Tensor *t = tensor_create(ndim, shape, DL_DEVICE_CPU, requires_grad);
    if (t) {
        for (size_t i = 0; i < t->size; i++) {
            t->data[i] = xorshift_rand();
        }
    }
    return t;
}

Tensor *tensor_eye(size_t n, bool requires_grad) {
    Tensor *t = tensor_zeros_2d(n, n, requires_grad);
    if (t) {
        for (size_t i = 0; i < n; i++) {
            t->data[i * n + i] = 1.0;
        }
    }
    return t;
}

Tensor *tensor_clone(const Tensor *t) {
    if (!t) return NULL;

    Tensor *out = tensor_create(t->ndim, t->shape, t->device, t->requires_grad);
    if (out && t->data) {
        memcpy(out->data, t->data, t->size * sizeof(double));
    }
    return out;
}

void tensor_free(Tensor *t) {
    if (!t) return;
    free(t->data);
    free(t->shape);
    free(t->stride);
    if (t->grad) {
        tensor_free((Tensor *)t->grad);
    }
    free(t);
}

/* ============================================================================
 * TENSOR INFO
 * ============================================================================ */

bool tensor_shape_equal(const Tensor *a, const Tensor *b) {
    if (!a || !b) return false;
    if (a->ndim != b->ndim) return false;
    return shapes_equal(a->shape, b->shape, a->ndim);
}

size_t tensor_flat_index(const Tensor *t, const size_t *indices) {
    size_t idx = 0;
    for (size_t i = 0; i < t->ndim; i++) {
        idx += indices[i] * t->stride[i];
    }
    return idx;
}

double tensor_get(const Tensor *t, const size_t *indices) {
    return t->data[tensor_flat_index(t, indices)];
}

void tensor_set(Tensor *t, const size_t *indices, double value) {
    t->data[tensor_flat_index(t, indices)] = value;
}

void tensor_print(const Tensor *t, const char *name, bool print_data) {
    if (!t) {
        printf("%s: NULL\n", name ? name : "tensor");
        return;
    }

    printf("%s: shape=[", name ? name : "tensor");
    for (size_t i = 0; i < t->ndim; i++) {
        printf("%zu%s", t->shape[i], i < t->ndim - 1 ? ", " : "");
    }
    printf("], size=%zu, requires_grad=%s\n", t->size, t->requires_grad ? "true" : "false");

    if (print_data && t->data && t->size <= 100) {
        printf("  data: ");
        if (t->ndim == 1) {
            for (size_t i = 0; i < t->size; i++) {
                printf("%.4f ", t->data[i]);
            }
        } else if (t->ndim == 2) {
            for (size_t i = 0; i < t->shape[0]; i++) {
                printf("\n  row%zu: ", i);
                for (size_t j = 0; j < t->shape[1]; j++) {
                    printf("%.4f ", t->data[i * t->shape[1] + j]);
                }
            }
        }
        printf("\n");
    }
}

/* ============================================================================
 * SHAPE OPERATIONS
 * ============================================================================ */

Tensor *tensor_reshape(const Tensor *t, size_t ndim, const size_t *shape) {
    if (!t || !shape) return NULL;

    size_t new_size = calc_total_size(shape, ndim);
    if (new_size != t->size) {
        DL_ERROR("Reshape size mismatch: %zu vs %zu", new_size, t->size);
        return NULL;
    }

    Tensor *out = tensor_create(ndim, shape, t->device, t->requires_grad);
    if (out) {
        memcpy(out->data, t->data, t->size * sizeof(double));
        out->stride[ndim - 1] = 1;
        for (size_t i = ndim - 1; i > 0; i--) {
            out->stride[i - 1] = out->stride[i] * shape[i];
        }
    }
    return out;
}

Tensor *tensor_transpose(const Tensor *t) {
    if (!t || t->ndim != 2) return NULL;

    size_t shape[] = {t->shape[1], t->shape[0]};
    Tensor *out = tensor_create(2, shape, t->device, t->requires_grad);
    if (!out) return NULL;

    for (size_t i = 0; i < t->shape[0]; i++) {
        for (size_t j = 0; j < t->shape[1]; j++) {
            out->data[j * t->shape[0] + i] = t->data[i * t->shape[1] + j];
        }
    }
    return out;
}

Tensor *tensor_expand_dims(const Tensor *t, size_t axis) {
    if (!t) return NULL;
    if (axis > t->ndim) axis = t->ndim;

    size_t *new_shape = DL_MALLOC(size_t, t->ndim + 1);
    if (!new_shape) return NULL;

    for (size_t i = 0; i < axis; i++) new_shape[i] = t->shape[i];
    new_shape[axis] = 1;
    for (size_t i = axis; i < t->ndim; i++) new_shape[i + 1] = t->shape[i];

    Tensor *out = tensor_create(t->ndim + 1, new_shape, t->device, t->requires_grad);
    free(new_shape);

    if (out) {
        memcpy(out->data, t->data, t->size * sizeof(double));
    }
    return out;
}

Tensor *tensor_squeeze(const Tensor *t, int axis) {
    if (!t) return NULL;

    if (axis >= 0) {
        /* Squeeze specific axis if it's size 1 */
        if (t->shape[axis] != 1) return tensor_clone(t);

        size_t *new_shape = DL_MALLOC(size_t, t->ndim - 1);
        if (!new_shape) return NULL;

        for (size_t i = 0; i < (size_t)axis; i++) new_shape[i] = t->shape[i];
        for (size_t i = axis + 1; i < t->ndim; i++) new_shape[i - 1] = t->shape[i];

        Tensor *out = tensor_create(t->ndim - 1, new_shape, t->device, t->requires_grad);
        free(new_shape);
        if (out) memcpy(out->data, t->data, t->size * sizeof(double));
        return out;
    } else {
        /* Squeeze all size-1 dimensions */
        size_t new_ndim = 0;
        for (size_t i = 0; i < t->ndim; i++) {
            if (t->shape[i] != 1) new_ndim++;
        }
        if (new_ndim == t->ndim) return tensor_clone(t);

        size_t *new_shape = DL_MALLOC(size_t, new_ndim);
        size_t j = 0;
        for (size_t i = 0; i < t->ndim; i++) {
            if (t->shape[i] != 1) new_shape[j++] = t->shape[i];
        }

        Tensor *out = tensor_create(new_ndim, new_shape, t->device, t->requires_grad);
        free(new_shape);
        if (out) memcpy(out->data, t->data, t->size * sizeof(double));
        return out;
    }
}

Tensor *tensor_concat(Tensor **tensors, size_t n, size_t axis) {
    if (!tensors || n == 0 || !tensors[0]) return NULL;

    Tensor *first = tensors[0];
    if (axis >= first->ndim) return NULL;

    /* Calculate result shape */
    size_t result_shape[8];  /* Max 8 dims */
    size_t ndim = first->ndim;
    for (size_t i = 0; i < ndim; i++) result_shape[i] = first->shape[i];
    for (size_t i = 1; i < n; i++) {
        if (!tensor_shape_equal(tensors[0], tensors[i])) {
            DL_ERROR("Cannot concatenate tensors with different shapes");
            return NULL;
        }
        result_shape[axis] += tensors[i]->shape[axis];
    }

    Tensor *result = tensor_create(ndim, result_shape, first->device, first->requires_grad);
    if (!result) return NULL;

    /* Copy data */
    size_t offset = 0;
    size_t stride = 1;
    for (int i = (int)axis - 1; i >= 0; i--) stride *= result_shape[i];

    for (size_t i = 0; i < n; i++) {
        size_t copy_size = tensors[i]->shape[axis] * stride;
        memcpy(result->data + offset, tensors[i]->data, copy_size * sizeof(double));
        offset += copy_size;
    }

    return result;
}

Tensor **tensor_split(const Tensor *t, size_t n_chunks, size_t axis) {
    if (!t || n_chunks == 0 || axis >= t->ndim) return NULL;
    if (t->shape[axis] % n_chunks != 0) {
        DL_ERROR("Split dimension must be divisible by n_chunks");
        return NULL;
    }

    Tensor **results = DL_CALLOC(Tensor *, n_chunks);
    if (!results) return NULL;

    size_t chunk_size = t->shape[axis] / n_chunks;
    size_t stride = 1;
    for (size_t i = 0; i < axis; i++) stride *= t->shape[i];

    for (size_t c = 0; c < n_chunks; c++) {
        size_t *new_shape = DL_MALLOC(size_t, t->ndim);
        for (size_t i = 0; i < t->ndim; i++) {
            new_shape[i] = (i == axis) ? chunk_size : t->shape[i];
        }
        results[c] = tensor_create(t->ndim, new_shape, t->device, t->requires_grad);
        free(new_shape);

        if (!results[c]) {
            for (size_t j = 0; j < c; j++) tensor_free(results[j]);
            free(results);
            return NULL;
        }

        /* Copy chunk */
        size_t src_offset = c * chunk_size * stride;
        memcpy(results[c]->data, t->data + src_offset, results[c]->size * sizeof(double));
    }

    return results;
}

Tensor *tensor_stack(Tensor **tensors, size_t n, size_t axis) {
    if (!tensors || n == 0 || !tensors[0]) return NULL;

    Tensor *first = tensors[0];
    size_t ndim = first->ndim + 1;

    if (axis > ndim) axis = ndim;

    /* Calculate result shape */
    size_t *result_shape = DL_CALLOC(size_t, ndim);
    for (size_t i = 0; i < axis; i++) result_shape[i] = first->shape[i];
    result_shape[axis] = n;
    for (size_t i = axis; i < first->ndim; i++) result_shape[i + 1] = first->shape[i];

    Tensor *result = tensor_create(ndim, result_shape, first->device, first->requires_grad);
    free(result_shape);
    if (!result) return NULL;

    /* Copy each tensor into the result */
    size_t elem_per_chunk = first->size;
    for (size_t i = 0; i < n; i++) {
        if (!tensor_shape_equal(first, tensors[i])) {
            DL_ERROR("Cannot stack tensors with different shapes");
            tensor_free(result);
            return NULL;
        }
        size_t offset = i * elem_per_chunk;
        for (size_t j = 0; j < elem_per_chunk; j++) {
            result->data[offset + j] = tensors[i]->data[j];
        }
    }

    return result;
}

/* ============================================================================
 * ELEMENT-WISE OPERATIONS
 * ============================================================================ */

Tensor *tensor_add(const Tensor *a, const Tensor *b) {
    if (!a || !b) return NULL;

    /* Broadcast if needed */
    size_t ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    size_t shape[8];
    for (size_t i = 0; i < ndim; i++) {
        size_t dim_a = (i < a->ndim) ? a->shape[a->ndim - 1 - i] : 1;
        size_t dim_b = (i < b->ndim) ? b->shape[b->ndim - 1 - i] : 1;
        shape[ndim - 1 - i] = dim_a > dim_b ? dim_a : dim_b;
    }

    Tensor *out = tensor_create(ndim, shape, a->device, a->requires_grad);
    if (!out) return NULL;

    /* Simple element-wise (assuming same shape for now) */
    for (size_t i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
    return out;
}

Tensor *tensor_sub(const Tensor *a, const Tensor *b) {
    if (!a || !b) return NULL;

    Tensor *out = tensor_clone(a);
    if (!out) return NULL;

    for (size_t i = 0; i < out->size; i++) {
        out->data[i] -= b->data[i];
    }
    return out;
}

Tensor *tensor_mul(const Tensor *a, const Tensor *b) {
    if (!a || !b) return NULL;

    Tensor *out = tensor_clone(a);
    if (!out) return NULL;

    for (size_t i = 0; i < out->size; i++) {
        out->data[i] *= b->data[i];
    }
    return out;
}

Tensor *tensor_div(const Tensor *a, const Tensor *b) {
    if (!a || !b) return NULL;

    Tensor *out = tensor_clone(a);
    if (!out) return NULL;

    for (size_t i = 0; i < out->size; i++) {
        out->data[i] /= (b->data[i] + DL_EPS);
    }
    return out;
}

Tensor *tensor_add_scalar(const Tensor *a, double scalar) {
    if (!a) return NULL;

    Tensor *out = tensor_clone(a);
    if (!out) return NULL;

    for (size_t i = 0; i < out->size; i++) {
        out->data[i] += scalar;
    }
    return out;
}

Tensor *tensor_mul_scalar(const Tensor *a, double scalar) {
    if (!a) return NULL;

    Tensor *out = tensor_clone(a);
    if (!out) return NULL;

    for (size_t i = 0; i < out->size; i++) {
        out->data[i] *= scalar;
    }
    return out;
}

Tensor *tensor_neg(const Tensor *a) {
    return tensor_mul_scalar(a, -1.0);
}

Tensor *tensor_pow(const Tensor *a, double exponent) {
    if (!a) return NULL;

    Tensor *out = tensor_clone(a);
    if (!out) return NULL;

    for (size_t i = 0; i < out->size; i++) {
        out->data[i] = pow(out->data[i], exponent);
    }
    return out;
}

void tensor_add_inplace(Tensor *a, const Tensor *b) {
    if (!a || !b) return;
    for (size_t i = 0; i < a->size; i++) {
        a->data[i] += b->data[i];
    }
}

void tensor_add_scalar_inplace(Tensor *a, double scalar) {
    if (!a) return;
    for (size_t i = 0; i < a->size; i++) {
        a->data[i] += scalar;
    }
}

void tensor_mul_scalar_inplace(Tensor *a, double scalar) {
    if (!a) return;
    for (size_t i = 0; i < a->size; i++) {
        a->data[i] *= scalar;
    }
}

/* ============================================================================
 * REDUCTION OPERATIONS
 * ============================================================================ */

double tensor_sum(const Tensor *t) {
    if (!t) return 0.0;
    double s = 0.0;
    for (size_t i = 0; i < t->size; i++) {
        s += t->data[i];
    }
    return s;
}

Tensor *tensor_sum_axis(const Tensor *t, size_t axis) {
    if (!t || axis >= t->ndim) return NULL;

    size_t *new_shape = DL_MALLOC(size_t, t->ndim - 1);
    for (size_t i = 0; i < axis; i++) new_shape[i] = t->shape[i];
    for (size_t i = axis + 1; i < t->ndim; i++) new_shape[i - 1] = t->shape[i];

    Tensor *out = tensor_create(t->ndim - 1, new_shape, t->device, t->requires_grad);
    free(new_shape);
    if (!out) return NULL;

    size_t stride = 1;
    for (size_t i = 0; i < axis; i++) stride *= t->shape[i];
    size_t block_size = stride * t->shape[axis];
    size_t n_blocks = t->size / block_size;

    for (size_t b = 0; b < n_blocks; b++) {
        size_t block_start = b * block_size;
        for (size_t i = 0; i < stride; i++) {
            double s = 0.0;
            for (size_t j = 0; j < t->shape[axis]; j++) {
                s += t->data[block_start + j * stride + i];
            }
            out->data[b * stride + i] = s;
        }
    }

    return out;
}

double tensor_mean(const Tensor *t) {
    if (!t || t->size == 0) return 0.0;
    return tensor_sum(t) / t->size;
}

Tensor *tensor_mean_axis(const Tensor *t, size_t axis) {
    Tensor *s = tensor_sum_axis(t, axis);
    if (!s) return NULL;

    double n = (double)t->shape[axis];
    for (size_t i = 0; i < s->size; i++) {
        s->data[i] /= n;
    }
    return s;
}

double tensor_max(const Tensor *t) {
    if (!t || t->size == 0) return 0.0;
    double m = t->data[0];
    for (size_t i = 1; i < t->size; i++) {
        if (t->data[i] > m) m = t->data[i];
    }
    return m;
}

double tensor_min(const Tensor *t) {
    if (!t || t->size == 0) return 0.0;
    double m = t->data[0];
    for (size_t i = 1; i < t->size; i++) {
        if (t->data[i] < m) m = t->data[i];
    }
    return m;
}

double tensor_std(const Tensor *t) {
    if (!t || t->size == 0) return 0.0;
    double m = tensor_mean(t);
    double var = 0.0;
    for (size_t i = 0; i < t->size; i++) {
        double d = t->data[i] - m;
        var += d * d;
    }
    return sqrt(var / t->size);
}

double tensor_norm(const Tensor *t) {
    if (!t) return 0.0;
    double s = 0.0;
    for (size_t i = 0; i < t->size; i++) {
        s += t->data[i] * t->data[i];
    }
    return sqrt(s);
}

/* ============================================================================
 * MATRIX OPERATIONS
 * ============================================================================ */

Tensor *tensor_matmul(const Tensor *a, const Tensor *b) {
    if (!a || !b) return NULL;
    if (a->ndim != 2 || b->ndim != 2) {
        DL_ERROR("Matmul requires 2D tensors");
        return NULL;
    }
    if (a->shape[1] != b->shape[0]) {
        DL_ERROR("Matmul dimension mismatch: %zu x %zu @ %zu x %zu",
                 a->shape[0], a->shape[1], b->shape[0], b->shape[1]);
        return NULL;
    }

    size_t m = a->shape[0], k = a->shape[1], n = b->shape[1];
    size_t shape[] = {m, n};
    Tensor *out = tensor_create(2, shape, a->device, a->requires_grad);
    if (!out) return NULL;

    /* C[i,j] = sum_k A[i,k] * B[k,j] */
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            double s = 0.0;
            for (size_t kk = 0; kk < k; kk++) {
                s += a->data[i * k + kk] * b->data[kk * n + j];
            }
            out->data[i * n + j] = s;
        }
    }
    return out;
}

Tensor *tensor_matmul_t(const Tensor *a, bool trans_a, const Tensor *b, bool trans_b) {
    if (!a || !b) return NULL;

    size_t a_rows = trans_a ? a->shape[1] : a->shape[0];
    size_t a_cols = trans_a ? a->shape[0] : a->shape[1];
    size_t b_rows = trans_b ? b->shape[1] : b->shape[0];
    size_t b_cols = trans_b ? b->shape[0] : b->shape[1];

    if (a_cols != b_rows) {
        DL_ERROR("Matmul dimension mismatch");
        return NULL;
    }

    size_t m = a_rows, k = a_cols, n = b_cols;
    size_t shape[] = {m, n};
    Tensor *out = tensor_create(2, shape, a->device, a->requires_grad);
    if (!out) return NULL;

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            double s = 0.0;
            for (size_t kk = 0; kk < k; kk++) {
                size_t a_i = trans_a ? kk * a->shape[0] + i : i * a->shape[1] + kk;
                size_t b_k = trans_b ? j * b->shape[0] + kk : kk * b->shape[1] + j;
                s += a->data[a_i] * b->data[b_k];
            }
            out->data[i * n + j] = s;
        }
    }
    return out;
}

Tensor *tensor_outer(const Tensor *a, const Tensor *b) {
    if (!a || !b || a->ndim != 1 || b->ndim != 1) return NULL;

    size_t m = a->shape[0], n = b->shape[0];
    size_t shape[] = {m, n};
    Tensor *out = tensor_create(2, shape, a->device, a->requires_grad);
    if (!out) return NULL;

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            out->data[i * n + j] = a->data[i] * b->data[j];
        }
    }
    return out;
}

double tensor_dot(const Tensor *a, const Tensor *b) {
    if (!a || !b) return 0.0;
    if (a->size != b->size) return 0.0;

    double s = 0.0;
    for (size_t i = 0; i < a->size; i++) {
        s += a->data[i] * b->data[i];
    }
    return s;
}

Tensor *tensor_bmm(const Tensor *a, const Tensor *b) {
    if (!a || !b) return NULL;
    if (a->ndim != 3 || b->ndim != 3) {
        DL_ERROR("BMM requires 3D tensors [batch, m, k] x [batch, k, n]");
        return NULL;
    }
    if (a->shape[0] != b->shape[0] || a->shape[2] != b->shape[1]) {
        DL_ERROR("BMM dimension mismatch");
        return NULL;
    }

    size_t batch = a->shape[0];
    size_t m = a->shape[1], k = a->shape[2], n = b->shape[2];

    size_t shape[] = {batch, m, n};
    Tensor *out = tensor_create(3, shape, a->device, a->requires_grad);
    if (!out) return NULL;

    for (size_t bt = 0; bt < batch; bt++) {
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                double s = 0.0;
                for (size_t kk = 0; kk < k; kk++) {
                    s += a->data[bt * m * k + i * k + kk] *
                         b->data[bt * k * n + kk * n + j];
                }
                out->data[bt * m * n + i * n + j] = s;
            }
        }
    }
    return out;
}

/* ============================================================================
 * BROADCASTING
 * ============================================================================ */

bool tensor_can_broadcast(const Tensor *a, const Tensor *b, size_t *out_ndim, size_t *out_shape) {
    if (!a || !b) return false;

    size_t ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    *out_ndim = ndim;

    for (size_t i = 0; i < ndim; i++) {
        size_t dim_a = (i < a->ndim) ? a->shape[a->ndim - 1 - i] : 1;
        size_t dim_b = (i < b->ndim) ? b->shape[b->ndim - 1 - i] : 1;
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return false;
        }
        out_shape[ndim - 1 - i] = (dim_a > dim_b) ? dim_a : dim_b;
    }
    return true;
}

Tensor *tensor_broadcast_to(const Tensor *t, size_t ndim, const size_t *target_shape) {
    if (!t || !target_shape) return NULL;

    /* For now, just verify shapes match where both are > 1 */
    if (ndim < t->ndim) {
        DL_ERROR("Cannot broadcast to smaller ndim");
        return NULL;
    }

    Tensor *out = tensor_create(ndim, target_shape, t->device, t->requires_grad);
    if (!out) return NULL;

    /* Copy data (broadcast by replication) */
    for (size_t i = 0; i < out->size; i++) {
        /* Calculate source index */
        size_t src_idx = 0;
        for (size_t d = 0; d < t->ndim; d++) {
            size_t dim_size = (d < ndim - t->ndim) ? 1 : t->shape[d - (ndim - t->ndim)];
            size_t target_dim_size = out->shape[d];
            size_t coord = (i / out->stride[d]) % target_dim_size;
            if (dim_size > 1) {
                src_idx += coord * t->stride[d - (ndim - t->ndim)];
            }
        }
        out->data[i] = t->data[src_idx];
    }

    return out;
}

/* ============================================================================
 * GRADIENT OPERATIONS
 * ============================================================================ */

void tensor_alloc_grad(Tensor *t) {
    if (!t || t->requires_grad) return;
    t->grad = tensor_zeros(t->ndim, t->shape, t->device, false);
}

void tensor_free_grad(Tensor *t) {
    if (!t) return;
    if (t->grad) {
        tensor_free((Tensor *)t->grad);
        t->grad = NULL;
    }
}

void tensor_zero_grad(Tensor *t) {
    if (!t || !t->grad) return;
    Tensor *g = (Tensor *)t->grad;
    for (size_t i = 0; i < g->size; i++) {
        g->data[i] = 0.0;
    }
}

void tensor_accum_grad(Tensor *t, const Tensor *grad) {
    if (!t || !t->grad || !grad) return;
    Tensor *g = (Tensor *)t->grad;
    for (size_t i = 0; i < g->size; i++) {
        g->data[i] += grad->data[i];
    }
}

/* ============================================================================
 * IN-PLACE OPERATIONS
 * ============================================================================ */

void tensor_copy_to(Tensor *dest, const Tensor *src) {
    if (!dest || !src) return;
    if (!tensor_shape_equal(dest, src)) {
        DL_ERROR("Tensor copy requires same shape");
        return;
    }
    memcpy(dest->data, src->data, dest->size * sizeof(double));
}

void tensor_fill(Tensor *t, double value) {
    if (!t) return;
    for (size_t i = 0; i < t->size; i++) {
        t->data[i] = value;
    }
}
