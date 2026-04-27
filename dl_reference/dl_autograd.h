/**
 * @file dl_autograd.h
 * @brief Automatic differentiation engine
 *
 * Provides dynamic computational graph construction and automatic
 * differentiation (backpropagation) for tensor operations.
 *
 * Design:
 * - Operations build graph nodes when requires_grad=true
 * - Backward pass traverses graph in reverse topological order
 * - Gradients accumulated via chain rule
 */

#ifndef __DL_AUTOGRAD_H__
#define __DL_AUTOGRAD_H__

#include "dl_base.h"
#include "dl_tensor.h"

/* ============================================================================
 * GRADIENT FUNCTION (Autograd Node)
 * ============================================================================ */

/**
 * @brief Gradient function for an operation
 *
 * Each operation that can require gradients implements this interface.
 */
typedef struct GradientFunction {
    OpType op_type;              /**< Operation type */

    /* Saved tensors from forward pass needed for backward */
    Tensor **saved_tensors;      /**< Tensors saved during forward */
    size_t n_saved;              /**< Number of saved tensors */

    /* Saved data (non-tensor) needed for backward */
    void *saved_data;            /**< E.g., convolution params, shape info */
    size_t saved_data_size;      /**< Size of saved data */

    /* Backward function */
    void (*backward)(struct GradientFunction *grad_fn, Tensor *grad_output);

    /* Cleanup */
    void (*free_fn)(struct GradientFunction *grad_fn);

    /* References to input tensors (for graph traversal) */
    Tensor **inputs;
    size_t n_inputs;
} GradientFunction;

/* Forward declarations for backward computation functions */
void compute_add_backward(GradientFunction *grad_fn, Tensor *grad_output);
void compute_sub_backward(GradientFunction *grad_fn, Tensor *grad_output);
void compute_mul_backward(GradientFunction *grad_fn, Tensor *grad_output);
void compute_div_backward(GradientFunction *grad_fn, Tensor *grad_output);
void compute_neg_backward(GradientFunction *grad_fn, Tensor *grad_output);
void compute_pow_backward(GradientFunction *grad_fn, Tensor *grad_output);
void compute_matmul_backward(GradientFunction *grad_fn, Tensor *grad_output);
void compute_sum_backward(GradientFunction *grad_fn, Tensor *grad_output);
void compute_mean_backward(GradientFunction *grad_fn, Tensor *grad_output);
void compute_reshape_backward(GradientFunction *grad_fn, Tensor *grad_output);
void compute_transpose_backward(GradientFunction *grad_fn, Tensor *grad_output);
void compute_relu_backward(GradientFunction *grad_fn, Tensor *grad_output);
void compute_sigmoid_backward(GradientFunction *grad_fn, Tensor *grad_output);
void compute_tanh_backward(GradientFunction *grad_fn, Tensor *grad_output);

/**
 * @brief Create a gradient function
 */
GradientFunction *create_grad_fn(OpType op_type, Tensor **inputs, size_t n_inputs,
                                 Tensor **saved_tensors, size_t n_saved,
                                 void *saved_data, size_t saved_data_size,
                                 void (*backward_fn)(GradientFunction *, Tensor *));

/* ============================================================================
 * TENSOR GRADIENT ACCESS
 * ============================================================================ */

/**
 * @brief Get gradient tensor
 */
static inline Tensor *tensor_grad(const Tensor *t) {
    return (Tensor *)t->grad;
}

/**
 * @brief Check if tensor requires gradient
 */
static inline bool tensor_requires_grad(const Tensor *t) {
    return t->requires_grad;
}

/**
 * @brief Check if gradient is computed
 */
static inline bool tensor_has_grad(const Tensor *t) {
    return t->grad != NULL;
}

/* ============================================================================
 * GRADIENT SCOPE
 * ============================================================================ */

/**
 * @brief Begin gradient computation scope
 *
 * Within this scope, operations build the computation graph.
 */
void dl_grad_enable(void);

/**
 * @brief End gradient computation scope
 */
void dl_grad_disable(void);

/**
 * @brief Check if inside gradient scope
 */
bool dl_in_grad_scope(void);

/* ============================================================================
 * GRAPH CONSTRUCTION
 * ============================================================================ */

/**
 * @brief Set gradient function for tensor output
 *
 * Called by tensor operations to register their gradient function.
 */
void tensor_set_grad_fn(Tensor *output, GradientFunction *grad_fn);

/**
 * @brief Get gradient function from tensor
 */
GradientFunction *tensor_get_grad_fn(const Tensor *t);

/* ============================================================================
 * GRADIENT COMPUTATION
 * ============================================================================ */

/**
 * @brief Compute gradient of tensor with respect to leaves
 *
 * @param t Tensor to compute gradient for
 * @param grad_output Gradient of loss w.r.t. output (NULL = ones)
 */
void dl_backward(Tensor *t, Tensor *grad_output);

/**
 * @brief Compute gradient w.r.t. specific tensors
 *
 * @param t Tensor to backpropagate from
 * @param targets Tensors to compute gradients for
 * @param n_targets Number of target tensors
 * @param grad_output Gradient of loss w.r.t. output
 */
void dl_backward_target(Tensor *t, Tensor **targets, size_t n_targets,
                       Tensor *grad_output);

/**
 * @brief Zero gradients of specified tensors
 */
void dl_zero_grad(Tensor **tensors, size_t n_tensors);

/**
 * @brief Free gradient function and cleanup
 */
void grad_fn_free(GradientFunction *grad_fn);

/* ============================================================================
 * LEAF TENSOR MANAGEMENT
 * ============================================================================ */

/**
 * @brief Register tensor as a leaf tensor
 *
 * Leaf tensors are created by user (e.g., input data) and
 * gradients are accumulated into their grad field.
 */
void tensor_register_leaf(Tensor *t, bool is_leaf);

/**
 * @brief Check if tensor is a leaf
 */
bool tensor_is_leaf(const Tensor *t);

/* ============================================================================
 * GRADIENT ACCUMULATION
 * ============================================================================ */

/**
 * @brief Accumulate gradient (add to existing)
 *
 * Handles cases where tensor is used in multiple operations.
 */
void tensor_accumulate_grad(Tensor *t, const Tensor *grad);

/**
 * @brief Check if tensor grad is views/alias of another
 */
bool tensor_grad_is_alias(const Tensor *t);

/* ============================================================================
 * IN-PLACE OPERATION HANDLING
 * ============================================================================ */

/**
 * @brief Increment version counter for in-place ops
 */
int dl_inc_version(void);

/**
 * @brief Get current version counter
 */
int dl_get_version(void);

/**
 * @brief Check version mismatch (for error detection)
 */
bool dl_check_version(const Tensor *t);

#endif /* __DL_AUTOGRAD_H__ */
