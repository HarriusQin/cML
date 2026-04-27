/**
 * @file dl_tensor.h
 * @brief Tensor operations for deep learning library
 *
 * Provides basic tensor manipulation functions including:
 * - Creation (zeros, ones, random, from data)
 * - Shape operations (reshape, transpose, slice)
 * - Element-wise operations
 * - Reduction operations (sum, mean, max)
 * - Matrix multiplication
 */

#ifndef __DL_TENSOR_H__
#define __DL_TENSOR_H__

#include "dl_base.h"

/* ============================================================================
 * TENSOR CREATION
 * ============================================================================ */

/**
 * @brief Create a tensor with uninitialized data
 *
 * @param ndim Number of dimensions
 * @param shape Array of dimension sizes
 * @param device Device type (CPU for now)
 * @param requires_grad Whether to track gradients
 * @return New tensor or NULL on failure
 */
Tensor *tensor_create(size_t ndim, const size_t *shape, DL_Device_t device, bool requires_grad);

/**
 * @brief Create a tensor filled with zeros
 */
Tensor *tensor_zeros(size_t ndim, const size_t *shape, DL_Device_t device, bool requires_grad);

/**
 * @brief Create a tensor filled with ones
 */
Tensor *tensor_ones(size_t ndim, const size_t *shape, DL_Device_t device, bool requires_grad);

/**
 * @brief Create a tensor filled with a constant value
 */
Tensor *tensor_full(size_t ndim, const size_t *shape, double fill_value,
                   DL_Device_t device, bool requires_grad);

/**
 * @brief Create a 1D tensor from an array
 */
Tensor *tensor_from_array_1d(const double *data, size_t size, bool requires_grad);

/**
 * @brief Create a 2D tensor (matrix) from an array
 */
Tensor *tensor_from_array_2d(const double *data, size_t rows, size_t cols, bool requires_grad);

/**
 * @brief Create a tensor with random normal values
 */
Tensor *tensor_randn(size_t ndim, const size_t *shape, double mean, double std, bool requires_grad);

/**
 * @brief Create a tensor with random uniform values in [0, 1)
 */
Tensor *tensor_rand(size_t ndim, const size_t *shape, bool requires_grad);

/**
 * @brief Create an identity matrix
 */
Tensor *tensor_eye(size_t n, bool requires_grad);

/**
 * @brief Clone a tensor (deep copy)
 */
Tensor *tensor_clone(const Tensor *t);

/**
 * @brief Free a tensor and its data
 */
void tensor_free(Tensor *t);

/* ============================================================================
 * TENSOR INFO
 * ============================================================================ */

/**
 * @brief Check if two tensors have the same shape
 */
bool tensor_shape_equal(const Tensor *a, const Tensor *b);

/**
 * @brief Get flat index from multi-dimensional indices
 */
size_t tensor_flat_index(const Tensor *t, const size_t *indices);

/**
 * @brief Get element at indices
 */
double tensor_get(const Tensor *t, const size_t *indices);

/**
 * @brief Set element at indices
 */
void tensor_set(Tensor *t, const size_t *indices, double value);

/**
 * @brief Print tensor info and optionally data
 */
void tensor_print(const Tensor *t, const char *name, bool print_data);

/* ============================================================================
 * SHAPE OPERATIONS
 * ============================================================================ */

/**
 * @brief Reshape tensor to new shape
 *
 * @param t Input tensor
 * @param ndim New number of dimensions
 * @param shape New shape
 * @return New tensor with new shape (shares data)
 */
Tensor *tensor_reshape(const Tensor *t, size_t ndim, const size_t *shape);

/**
 * @brief Transpose a 2D tensor (swap rows and cols)
 */
Tensor *tensor_transpose(const Tensor *t);

/**
 * @brief Expand dimensions (add size-1 dimensions)
 *
 * @param t Input tensor
 * @param axis Position to insert new dimension
 * @return New tensor with expanded shape
 */
Tensor *tensor_expand_dims(const Tensor *t, size_t axis);

/**
 * @brief Squeeze dimensions of size 1
 *
 * @param t Input tensor
 * @param axis Dimension to squeeze (-1 for all)
 * @return New tensor with squeezed shape
 */
Tensor *tensor_squeeze(const Tensor *t, int axis);

/**
 * @brief Concatenate tensors along an axis
 */
Tensor *tensor_concat(Tensor **tensors, size_t n, size_t axis);

/**
 * @brief Split tensor into chunks along axis
 */
Tensor **tensor_split(const Tensor *t, size_t n_chunks, size_t axis);

/**
 * @brief Stack tensors along new axis
 */
Tensor *tensor_stack(Tensor **tensors, size_t n, size_t axis);

/* ============================================================================
 * ELEMENT-WISE OPERATIONS
 * ============================================================================ */

/**
 * @brief Element-wise addition
 */
Tensor *tensor_add(const Tensor *a, const Tensor *b);

/**
 * @brief Element-wise subtraction
 */
Tensor *tensor_sub(const Tensor *a, const Tensor *b);

/**
 * @brief Element-wise multiplication
 */
Tensor *tensor_mul(const Tensor *a, const Tensor *b);

/**
 * @brief Element-wise division
 */
Tensor *tensor_div(const Tensor *a, const Tensor *b);

/**
 * @brief Scalar addition
 */
Tensor *tensor_add_scalar(const Tensor *a, double scalar);

/**
 * @brief Scalar multiplication
 */
Tensor *tensor_mul_scalar(const Tensor *a, double scalar);

/**
 * @brief Negation
 */
Tensor *tensor_neg(const Tensor *a);

/**
 * @brief Power operation
 */
Tensor *tensor_pow(const Tensor *a, double exponent);

/**
 * @brief In-place element-wise addition (no autograd support)
 */
void tensor_add_inplace(Tensor *a, const Tensor *b);

/**
 * @brief In-place scalar addition
 */
void tensor_add_scalar_inplace(Tensor *a, double scalar);

/**
 * @brief In-place scalar multiplication
 */
void tensor_mul_scalar_inplace(Tensor *a, double scalar);

/* ============================================================================
 * REDUCTION OPERATIONS
 * ============================================================================ */

/**
 * @brief Sum all elements
 */
double tensor_sum(const Tensor *t);

/**
 * @brief Sum along axis
 */
Tensor *tensor_sum_axis(const Tensor *t, size_t axis);

/**
 * @brief Mean of all elements
 */
double tensor_mean(const Tensor *t);

/**
 * @brief Mean along axis
 */
Tensor *tensor_mean_axis(const Tensor *t, size_t axis);

/**
 * @brief Max element
 */
double tensor_max(const Tensor *t);

/**
 * @brief Max along axis (returns indices too)
 */
void tensor_max_axis(const Tensor *t, size_t axis, Tensor **values, Tensor **indices);

/**
 * @brief Min element
 */
double tensor_min(const Tensor *t);

/**
 * @brief Standard deviation
 */
double tensor_std(const Tensor *t);

/**
 * @brief L2 norm (Euclidean norm)
 */
double tensor_norm(const Tensor *t);

/* ============================================================================
 * MATRIX OPERATIONS
 * ============================================================================ */

/**
 * @brief Matrix multiplication
 *
 * @param a First matrix [m, k]
 * @param b Second matrix [k, n]
 * @return Result matrix [m, n]
 */
Tensor *tensor_matmul(const Tensor *a, const Tensor *b);

/**
 * @brief Matrix multiplication with transpose flags
 *
 * @param a First matrix
 * @param trans_a Whether to transpose a
 * @param b Second matrix
 * @param trans_b Whether to transpose b
 */
Tensor *tensor_matmul_t(const Tensor *a, bool trans_a, const Tensor *b, bool trans_b);

/**
 * @brief Outer product of two vectors
 *
 * @param a Vector [m]
 * @param b Vector [n]
 * @return Matrix [m, n]
 */
Tensor *tensor_outer(const Tensor *a, const Tensor *b);

/**
 * @brief Dot product of two vectors
 */
double tensor_dot(const Tensor *a, const Tensor *b);

/**
 * @brief Batch matrix multiplication
 *
 * For batched matmul where each matrix in batch is multiplied.
 * a: [batch, m, k], b: [batch, k, n] -> [batch, m, n]
 */
Tensor *tensor_bmm(const Tensor *a, const Tensor *b);

/* ============================================================================
 * BROADCASTING
 * ============================================================================ */

/**
 * @brief Broadcast tensor to target shape
 *
 * Handles numpy-style broadcasting rules.
 */
Tensor *tensor_broadcast_to(const Tensor *t, size_t ndim, const size_t *target_shape);

/**
 * @brief Check if broadcasting is valid
 */
bool tensor_can_broadcast(const Tensor *a, const Tensor *b, size_t *out_ndim, size_t *out_shape);

/* ============================================================================
 * TENSOR DATA ACCESS (for internal use)
 * ============================================================================ */

/**
 * @brief Get raw data pointer at linear index
 */
static inline double *tensor_data_at(Tensor *t, size_t index) {
    return t->data + index;
}

/**
 * @brief Get const raw data pointer at linear index
 */
static inline const double *tensor_const_data_at(const Tensor *t, size_t index) {
    return t->data + index;
}

/**
 * @brief Get data pointer
 */
static inline double *tensor_data(Tensor *t) {
    return t->data;
}

/**
 * @brief Get const data pointer
 */
static inline const double *tensor_const_data(const Tensor *t) {
    return t->data;
}

/* ============================================================================
 * GRADIENT OPERATIONS
 * ============================================================================ */

/**
 * @brief Allocate gradient tensor if needed
 */
void tensor_alloc_grad(Tensor *t);

/**
 * @brief Free gradient tensor
 */
void tensor_free_grad(Tensor *t);

/**
 * @brief Zero the gradient
 */
void tensor_zero_grad(Tensor *t);

/**
 * @brief Accumulate gradient (add to existing)
 */
void tensor_accum_grad(Tensor *t, const Tensor *grad);

/* ============================================================================
 * IN-PLACE OPERATIONS (use carefully with autograd)
 * ============================================================================ */

/**
 * @brief Copy data from one tensor to another (same shape required)
 */
void tensor_copy_to(Tensor *dest, const Tensor *src);

/**
 * @brief Fill tensor with value
 */
void tensor_fill(Tensor *t, double value);

/* ============================================================================
 * CONVENIENCE FUNCTIONS
 * ============================================================================ */

/**
 * @brief Create 2D tensor [rows, cols] with zeros
 */
static inline Tensor *tensor_zeros_2d(size_t rows, size_t cols, bool requires_grad) {
    size_t shape[] = {rows, cols};
    return tensor_zeros(2, shape, DL_DEVICE_CPU, requires_grad);
}

/**
 * @brief Create 2D tensor [rows, cols] with random normal
 */
static inline Tensor *tensor_randn_2d(size_t rows, size_t cols, double mean, double std, bool requires_grad) {
    size_t shape[] = {rows, cols};
    return tensor_randn(2, shape, mean, std, requires_grad);
}

/**
 * @brief Create 1D tensor with zeros
 */
static inline Tensor *tensor_zeros_1d(size_t size, bool requires_grad) {
    return tensor_zeros(1, &size, DL_DEVICE_CPU, requires_grad);
}

/**
 * @brief Create 1D tensor with ones
 */
static inline Tensor *tensor_ones_1d(size_t size, bool requires_grad) {
    return tensor_ones(1, &size, DL_DEVICE_CPU, requires_grad);
}

#endif /* __DL_TENSOR_H__ */
