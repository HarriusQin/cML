/**
 * @file dl_autograd.c
 * @brief Automatic differentiation implementation
 */

#include "dl_autograd.h"
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * GLOBAL AUTOGRAD CONTEXT
 * ============================================================================ */

/* Definition of dl_ctx declared in dl_base.h */
DLContext dl_ctx = {
    .enabled = false,
    .in_grad_scope = false,
    .version_counter = 0
};

static struct {
    bool enabled;
    bool in_scope;
    int version_counter;
} grad_ctx = {
    .enabled = false,
    .in_scope = false,
    .version_counter = 0
};

void dl_grad_enable(void) {
    grad_ctx.enabled = true;
    grad_ctx.in_scope = true;
    dl_ctx.enabled = true;
    dl_ctx.in_grad_scope = true;
}

void dl_grad_disable(void) {
    grad_ctx.in_scope = false;
}

bool dl_in_grad_scope(void) {
    return grad_ctx.in_scope;
}

int dl_inc_version(void) {
    return ++grad_ctx.version_counter;
}

int dl_get_version(void) {
    return grad_ctx.version_counter;
}

bool dl_check_version(const Tensor *t) {
    return true;
}

/* ============================================================================
 * GRADIENT FUNCTION REGISTRY
 * ============================================================================ */

void tensor_set_grad_fn(Tensor *output, GradientFunction *grad_fn) {
    if (!output) return;
    /* Store grad_fn in ctx field */
    output->ctx = grad_fn;
}

GradientFunction *tensor_get_grad_fn(const Tensor *t) {
    if (!t) return NULL;
    return (GradientFunction *)t->ctx;
}

/* ============================================================================
 * GRADIENT COMPUTATION - FORWARD DECLARATIONS
 * ============================================================================ */

static void free_gradient_function(GradientFunction *grad_fn);
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

/* ============================================================================
 * GRADIENT FUNCTION FACTORY
 * ============================================================================ */

GradientFunction *create_grad_fn(OpType op_type, Tensor **inputs, size_t n_inputs,
                                 Tensor **saved_tensors, size_t n_saved,
                                 void *saved_data, size_t saved_data_size,
                                 void (*backward_fn)(GradientFunction *, Tensor *)) {
    GradientFunction *grad_fn = (GradientFunction *)calloc(1, sizeof(GradientFunction));
    if (!grad_fn) return NULL;

    grad_fn->op_type = op_type;
    grad_fn->inputs = inputs;
    grad_fn->n_inputs = n_inputs;
    grad_fn->saved_tensors = saved_tensors;
    grad_fn->n_saved = n_saved;
    grad_fn->saved_data = saved_data;
    grad_fn->saved_data_size = saved_data_size;
    grad_fn->backward = backward_fn;
    grad_fn->free_fn = free_gradient_function;

    return grad_fn;
}

/* ============================================================================
 * GRADIENT FUNCTION CLEANUP
 * ============================================================================ */

static void free_gradient_function(GradientFunction *grad_fn) {
    if (!grad_fn) return;

    /* Free saved tensors if they were cloned */
    if (grad_fn->saved_tensors) {
        for (size_t i = 0; i < grad_fn->n_saved; i++) {
            /* Don't free - they are views of original tensors */
        }
        free(grad_fn->saved_tensors);
    }

    if (grad_fn->saved_data) {
        free(grad_fn->saved_data);
    }

    free(grad_fn);
}

void grad_fn_free(GradientFunction *grad_fn) {
    if (grad_fn && grad_fn->free_fn) {
        grad_fn->free_fn(grad_fn);
    }
}

/* ============================================================================
 * LEAF TENSOR MANAGEMENT
 * ============================================================================ */

typedef struct {
    Tensor *tensor;
    bool is_leaf;
} LeafEntry;

static LeafEntry *leaf_registry = NULL;
static size_t leaf_count = 0;
static size_t leaf_capacity = 0;

void tensor_register_leaf(Tensor *t, bool is_leaf) {
    if (!t) return;

    if (leaf_capacity == 0) {
        leaf_capacity = 64;
        leaf_registry = (LeafEntry *)calloc(leaf_capacity, sizeof(LeafEntry));
    }

    if (leaf_count >= leaf_capacity) {
        leaf_capacity *= 2;
        leaf_registry = (LeafEntry *)realloc(leaf_registry, leaf_capacity * sizeof(LeafEntry));
    }

    leaf_registry[leaf_count].tensor = t;
    leaf_registry[leaf_count].is_leaf = is_leaf;
    leaf_count++;
}

bool tensor_is_leaf(const Tensor *t) {
    for (size_t i = 0; i < leaf_count; i++) {
        if (leaf_registry[i].tensor == t) {
            return leaf_registry[i].is_leaf;
        }
    }
    return false;
}

/* ============================================================================
 * GRADIENT ACCUMULATION
 * ============================================================================ */

void tensor_accumulate_grad(Tensor *t, const Tensor *grad) {
    if (!t || !grad) return;

    if (t->grad == NULL) {
        t->grad = tensor_clone(grad);
    } else {
        Tensor *existing = (Tensor *)t->grad;
        Tensor *sum = tensor_add(existing, grad);
        tensor_free(existing);
        t->grad = sum;
    }
}

bool tensor_grad_is_alias(const Tensor *t) {
    return false;
}

/* ============================================================================
 * GRADIENT ZEROING
 * ============================================================================ */

void dl_zero_grad(Tensor **tensors, size_t n_tensors) {
    for (size_t i = 0; i < n_tensors; i++) {
        if (tensors[i] && tensors[i]->grad) {
            tensor_free((Tensor *)tensors[i]->grad);
            tensors[i]->grad = NULL;
        }
    }
}

/* ============================================================================
 * BACKWARD PASS IMPLEMENTATIONS
 * ============================================================================ */

/* Element-wise Add: z = x + y, dz/dx = 1, dz/dy = 1 */
void compute_add_backward(GradientFunction *grad_fn, Tensor *grad_output) {
    if (!grad_fn || !grad_output) return;

    for (size_t i = 0; i < grad_fn->n_inputs; i++) {
        Tensor *input = grad_fn->inputs[i];
        if (input && input->requires_grad && input->grad == NULL) {
            tensor_alloc_grad((Tensor *)input);
            tensor_copy_to((Tensor *)input->grad, grad_output);
        }
    }
}

/* Element-wise Sub: z = x - y, dz/dx = 1, dz/dy = -1 */
void compute_sub_backward(GradientFunction *grad_fn, Tensor *grad_output) {
    if (!grad_fn || !grad_output) return;

    if (grad_fn->n_inputs >= 1 && grad_fn->inputs[0] && grad_fn->inputs[0]->requires_grad) {
        tensor_alloc_grad((Tensor *)grad_fn->inputs[0]);
        tensor_copy_to((Tensor *)grad_fn->inputs[0]->grad, grad_output);
    }

    if (grad_fn->n_inputs >= 2 && grad_fn->inputs[1] && grad_fn->inputs[1]->requires_grad) {
        tensor_alloc_grad((Tensor *)grad_fn->inputs[1]);
        Tensor *neg_grad = tensor_neg(grad_output);
        tensor_copy_to((Tensor *)grad_fn->inputs[1]->grad, neg_grad);
        tensor_free(neg_grad);
    }
}

/* Element-wise Mul: z = x * y, dz/dx = y, dz/dy = x */
void compute_mul_backward(GradientFunction *grad_fn, Tensor *grad_output) {
    if (!grad_fn || !grad_output || grad_fn->n_saved < 2) return;

    Tensor *x = grad_fn->saved_tensors[0];
    Tensor *y = grad_fn->saved_tensors[1];

    if (grad_fn->inputs[0] && grad_fn->inputs[0]->requires_grad) {
        tensor_alloc_grad((Tensor *)grad_fn->inputs[0]);
        Tensor *grad_x = tensor_mul(grad_output, y);
        tensor_copy_to((Tensor *)grad_fn->inputs[0]->grad, grad_x);
        tensor_free(grad_x);
    }

    if (grad_fn->inputs[1] && grad_fn->inputs[1]->requires_grad) {
        tensor_alloc_grad((Tensor *)grad_fn->inputs[1]);
        Tensor *grad_y = tensor_mul(grad_output, x);
        tensor_copy_to((Tensor *)grad_fn->inputs[1]->grad, grad_y);
        tensor_free(grad_y);
    }
}

/* Element-wise Div: z = x / y, dz/dx = 1/y, dz/dy = -x/y^2 */
void compute_div_backward(GradientFunction *grad_fn, Tensor *grad_output) {
    if (!grad_fn || !grad_output || grad_fn->n_saved < 2) return;

    Tensor *x = grad_fn->saved_tensors[0];
    Tensor *y = grad_fn->saved_tensors[1];

    if (grad_fn->inputs[0] && grad_fn->inputs[0]->requires_grad) {
        tensor_alloc_grad((Tensor *)grad_fn->inputs[0]);
        Tensor *inv_y = tensor_div(tensor_ones(1, (size_t[]){1}, DL_DEVICE_CPU, false), y);
        Tensor *grad_x = tensor_mul(grad_output, inv_y);
        tensor_copy_to((Tensor *)grad_fn->inputs[0]->grad, grad_x);
        tensor_free(grad_x);
        tensor_free(inv_y);
    }
}

/* Negation: z = -x, dz/dx = -1 */
void compute_neg_backward(GradientFunction *grad_fn, Tensor *grad_output) {
    if (!grad_fn || !grad_output) return;

    if (grad_fn->inputs[0] && grad_fn->inputs[0]->requires_grad) {
        tensor_alloc_grad((Tensor *)grad_fn->inputs[0]);
        Tensor *neg_grad = tensor_neg(grad_output);
        tensor_copy_to((Tensor *)grad_fn->inputs[0]->grad, neg_grad);
        tensor_free(neg_grad);
    }
}

/* Power: z = x^exponent, dz/dx = exponent * x^(exponent-1) */
void compute_pow_backward(GradientFunction *grad_fn, Tensor *grad_output) {
    if (!grad_fn || !grad_output || grad_fn->n_saved < 2) return;

    Tensor *x = grad_fn->saved_tensors[0];
    double exponent = *(double *)grad_fn->saved_data;

    if (grad_fn->inputs[0] && grad_fn->inputs[0]->requires_grad) {
        tensor_alloc_grad((Tensor *)grad_fn->inputs[0]);
        Tensor *x_pow = tensor_pow(x, exponent - 1);
        Tensor *grad_x = tensor_mul_scalar(tensor_mul(grad_output, x_pow), exponent);
        tensor_copy_to((Tensor *)grad_fn->inputs[0]->grad, grad_x);
        tensor_free(grad_x);
        tensor_free(x_pow);
    }
}

/* Matrix Multiply: z = X @ Y, dz/dX = dz/dz @ Y^T, dz/dY = X^T @ dz/dz */
void compute_matmul_backward(GradientFunction *grad_fn, Tensor *grad_output) {
    if (!grad_fn || !grad_output || grad_fn->n_inputs < 2) return;

    Tensor *X = grad_fn->inputs[0];
    Tensor *Y = grad_fn->inputs[1];

    if (X && X->requires_grad) {
        tensor_alloc_grad((Tensor *)X);
        Tensor *grad_X = tensor_matmul_t(grad_output, false, Y, true);
        tensor_copy_to((Tensor *)X->grad, grad_X);
        tensor_free(grad_X);
    }

    if (Y && Y->requires_grad) {
        tensor_alloc_grad((Tensor *)Y);
        Tensor *grad_Y = tensor_matmul_t(X, true, grad_output, false);
        tensor_copy_to((Tensor *)Y->grad, grad_Y);
        tensor_free(grad_Y);
    }
}

/* Sum: z = sum(x), dz/dx = ones */
void compute_sum_backward(GradientFunction *grad_fn, Tensor *grad_output) {
    if (!grad_fn || !grad_output || grad_fn->n_saved < 1) return;

    Tensor *x = grad_fn->saved_tensors[0];
    if (grad_fn->inputs[0] && grad_fn->inputs[0]->requires_grad) {
        tensor_alloc_grad((Tensor *)grad_fn->inputs[0]);
        Tensor *grad_x = tensor_broadcast_to(grad_output, x->ndim, x->shape);
        tensor_copy_to((Tensor *)grad_fn->inputs[0]->grad, grad_x);
        tensor_free(grad_x);
    }
}

/* Mean: z = mean(x), dz/dx = 1/size */
void compute_mean_backward(GradientFunction *grad_fn, Tensor *grad_output) {
    if (!grad_fn || !grad_output || grad_fn->n_saved < 1) return;

    Tensor *x = grad_fn->saved_tensors[0];
    size_t size = x->size;
    double scale = 1.0 / (double)size;

    if (grad_fn->inputs[0] && grad_fn->inputs[0]->requires_grad) {
        tensor_alloc_grad((Tensor *)grad_fn->inputs[0]);
        Tensor *grad_scale = tensor_full(1, (size_t[]){1}, scale, DL_DEVICE_CPU, false);
        Tensor *grad_x = tensor_broadcast_to(
            tensor_mul_scalar(grad_output, scale), x->ndim, x->shape);
        tensor_copy_to((Tensor *)grad_fn->inputs[0]->grad, grad_x);
        tensor_free(grad_x);
        tensor_free(grad_scale);
    }
}

/* Reshape: z = reshape(x, shape), dz/dx = reshape(dz/dz, x.shape) */
void compute_reshape_backward(GradientFunction *grad_fn, Tensor *grad_output) {
    if (!grad_fn || !grad_output || grad_fn->n_saved < 1) return;

    Tensor *x = grad_fn->saved_tensors[0];
    size_t *orig_shape = (size_t *)grad_fn->saved_data;

    if (grad_fn->inputs[0] && grad_fn->inputs[0]->requires_grad) {
        tensor_alloc_grad((Tensor *)grad_fn->inputs[0]);
        Tensor *grad_x = tensor_reshape(grad_output, x->ndim, orig_shape);
        tensor_copy_to((Tensor *)grad_fn->inputs[0]->grad, grad_x);
        tensor_free(grad_x);
    }
}

/* Transpose: z = transpose(x), dz/dx = transpose(dz/dz) */
void compute_transpose_backward(GradientFunction *grad_fn, Tensor *grad_output) {
    if (!grad_fn || !grad_output) return;

    if (grad_fn->inputs[0] && grad_fn->inputs[0]->requires_grad) {
        tensor_alloc_grad((Tensor *)grad_fn->inputs[0]);
        Tensor *grad_x = tensor_transpose(grad_output);
        tensor_copy_to((Tensor *)grad_fn->inputs[0]->grad, grad_x);
        tensor_free(grad_x);
    }
}

/* ReLU: z = relu(x), dz/dx = dz/dz * (x > 0) */
void compute_relu_backward(GradientFunction *grad_fn, Tensor *grad_output) {
    if (!grad_fn || !grad_output || grad_fn->n_saved < 1) return;

    Tensor *x = grad_fn->saved_tensors[0];

    if (grad_fn->inputs[0] && grad_fn->inputs[0]->requires_grad) {
        tensor_alloc_grad((Tensor *)grad_fn->inputs[0]);

        /* Create mask where x > 0 */
        Tensor *mask = tensor_zeros(x->ndim, x->shape, DL_DEVICE_CPU, false);
        for (size_t i = 0; i < x->size; i++) {
            if (x->data[i] > 0) mask->data[i] = 1.0;
        }

        Tensor *grad_x = tensor_mul(grad_output, mask);
        tensor_copy_to((Tensor *)grad_fn->inputs[0]->grad, grad_x);
        tensor_free(grad_x);
        tensor_free(mask);
    }
}

/* Sigmoid: z = sigmoid(x), dz/dx = dz/dz * sigmoid(x) * (1 - sigmoid(x)) */
void compute_sigmoid_backward(GradientFunction *grad_fn, Tensor *grad_output) {
    if (!grad_fn || !grad_output || grad_fn->n_saved < 1) return;

    Tensor *sigmoid_x = grad_fn->saved_tensors[0];

    if (grad_fn->inputs[0] && grad_fn->inputs[0]->requires_grad) {
        tensor_alloc_grad((Tensor *)grad_fn->inputs[0]);

        /* derivative = sigmoid(x) * (1 - sigmoid(x)) */
        Tensor *one_minus_sigmoid = tensor_sub(
            tensor_ones(sigmoid_x->ndim, sigmoid_x->shape, DL_DEVICE_CPU, false),
            sigmoid_x
        );
        Tensor *deriv = tensor_mul(sigmoid_x, one_minus_sigmoid);
        Tensor *grad_x = tensor_mul(grad_output, deriv);

        tensor_copy_to((Tensor *)grad_fn->inputs[0]->grad, grad_x);
        tensor_free(grad_x);
        tensor_free(deriv);
        tensor_free(one_minus_sigmoid);
    }
}

/* Tanh: z = tanh(x), dz/dx = dz/dz * (1 - tanh^2(x)) */
void compute_tanh_backward(GradientFunction *grad_fn, Tensor *grad_output) {
    if (!grad_fn || !grad_output || grad_fn->n_saved < 1) return;

    Tensor *tanh_x = grad_fn->saved_tensors[0];

    if (grad_fn->inputs[0] && grad_fn->inputs[0]->requires_grad) {
        tensor_alloc_grad((Tensor *)grad_fn->inputs[0]);

        /* derivative = 1 - tanh^2(x) */
        Tensor *tanh_sq = tensor_mul(tanh_x, tanh_x);
        Tensor *one_minus_tanh_sq = tensor_sub(
            tensor_ones(tanh_x->ndim, tanh_x->shape, DL_DEVICE_CPU, false),
            tanh_sq
        );
        Tensor *grad_x = tensor_mul(grad_output, one_minus_tanh_sq);

        tensor_copy_to((Tensor *)grad_fn->inputs[0]->grad, grad_x);
        tensor_free(grad_x);
        tensor_free(one_minus_tanh_sq);
        tensor_free(tanh_sq);
    }
}

/* ============================================================================
 * MAIN BACKWARD PASS
 * ============================================================================ */

typedef struct {
    Tensor *tensor;
    GradientFunction *grad_fn;
    bool visited;
} GraphNode;

static GraphNode *graph_nodes = NULL;
static size_t graph_size = 0;
static size_t graph_capacity = 0;

static void graph_add_node(Tensor *tensor, GradientFunction *grad_fn) {
    if (graph_capacity == 0) {
        graph_capacity = 256;
        graph_nodes = (GraphNode *)calloc(graph_capacity, sizeof(GraphNode));
    }

    if (graph_size >= graph_capacity) {
        graph_capacity *= 2;
        graph_nodes = (GraphNode *)realloc(graph_nodes, graph_capacity * sizeof(GraphNode));
    }

    graph_nodes[graph_size].tensor = tensor;
    graph_nodes[graph_size].grad_fn = grad_fn;
    graph_nodes[graph_size].visited = false;
    graph_size++;
}

static void graph_reset(void) {
    for (size_t i = 0; i < graph_size; i++) {
        graph_nodes[i].visited = false;
    }
}

void dl_backward(Tensor *t, Tensor *grad_output) {
    if (!t) return;

    /* Initialize grad_output to ones if NULL */
    if (grad_output == NULL) {
        grad_output = tensor_ones(1, (size_t[]){1}, DL_DEVICE_CPU, false);
    }

    /* Build graph from output to leaves */
    graph_reset();
    graph_size = 0;

    /* Traverse graph via grad_fn references */
    Tensor *current = t;
    GradientFunction *current_grad_fn = tensor_get_grad_fn(current);

    if (current->requires_grad) {
        if (!current->grad) {
            current->grad = tensor_clone(grad_output);
        } else {
            Tensor *sum = tensor_add((Tensor *)current->grad, grad_output);
            tensor_free((Tensor *)current->grad);
            current->grad = sum;
        }
    }

    /* Simple backward traversal - just process immediate grad_fn */
    while (current_grad_fn != NULL) {
        GradientFunction *gf = current_grad_fn;

        /* Mark as visited */
        for (size_t i = 0; i < graph_size; i++) {
            if (graph_nodes[i].grad_fn == gf) {
                graph_nodes[i].visited = true;
                break;
            }
        }

        /* Call backward function */
        if (gf->backward) {
            Tensor *grad_out = (Tensor *)current->grad;
            gf->backward(gf, grad_out);
        }

        /* Move to first input with grad_fn */
        current_grad_fn = NULL;
        for (size_t i = 0; i < gf->n_inputs; i++) {
            if (gf->inputs[i] && gf->inputs[i]->ctx) {
                current = gf->inputs[i];
                current_grad_fn = tensor_get_grad_fn(current);
                break;
            }
        }
    }

    /* Accumulate gradients through the graph */
    for (size_t iter = 0; iter < graph_size; iter++) {
        for (size_t i = 0; i < graph_size; i++) {
            if (graph_nodes[i].visited) continue;

            GradientFunction *gf = graph_nodes[i].grad_fn;
            Tensor *input = graph_nodes[i].tensor;

            /* Check if all outputs that use this input have computed gradients */
            bool can_process = true;
            if (input->grad == NULL) {
                /* Not ready yet */
                continue;
            }

            /* Process this node */
            if (gf && gf->backward) {
                gf->backward(gf, (Tensor *)input->grad);
                graph_nodes[i].visited = true;
            }
        }
    }

    /* Cleanup */
    if (grad_output && grad_output != t->grad && !t->grad) {
        /* grad_output was temporary */
    }
}

void dl_backward_target(Tensor *t, Tensor **targets, size_t n_targets,
                       Tensor *grad_output) {
    (void)t;
    (void)targets;
    (void)n_targets;
    (void)grad_output;
    /* TODO: Implement targeted backprop */
}
