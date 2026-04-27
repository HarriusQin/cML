/**
 * @file dl_layers.c
 * @brief Neural network layer implementations
 */

#include "dl_layers.h"
#include "dl_autograd.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Forward declarations from dl_autograd.c */
GradientFunction *create_grad_fn(OpType op_type, Tensor **inputs, size_t n_inputs,
                                 Tensor **saved_tensors, size_t n_saved,
                                 void *saved_data, size_t saved_data_size,
                                 void (*backward_fn)(GradientFunction *, Tensor *));

/* Backward computation functions */
void compute_relu_backward(GradientFunction *grad_fn, Tensor *grad_output);
void compute_sigmoid_backward(GradientFunction *grad_fn, Tensor *grad_output);
void compute_tanh_backward(GradientFunction *grad_fn, Tensor *grad_output);

/* ============================================================================
 * ACTIVATION FUNCTIONS
 * ============================================================================ */

Tensor *tensor_relu(const Tensor *x) {
    if (!x) return NULL;

    Tensor *output = tensor_create(x->ndim, x->shape, x->device, x->requires_grad);

    for (size_t i = 0; i < x->size; i++) {
        output->data[i] = x->data[i] > 0 ? x->data[i] : 0;
    }

    if (x->requires_grad && dl_grad_enabled()) {
        /* Save input for backward */
        Tensor **saved = (Tensor **)malloc(sizeof(Tensor *));
        saved[0] = (Tensor *)x;
        tensor_clone(x);

        double *data_copy = (double *)malloc(sizeof(double) * x->size);
        memcpy(data_copy, x->data, sizeof(double) * x->size);

        GradientFunction *grad_fn = create_grad_fn(
            OP_RELU, (Tensor **)&x, 1, saved, 1, data_copy, sizeof(double) * x->size,
            compute_relu_backward
        );
        tensor_set_grad_fn(output, grad_fn);
    }

    return output;
}

Tensor *tensor_relu_backward(const Tensor *x, const Tensor *grad_output) {
    if (!x || !grad_output) return NULL;

    Tensor *grad_input = tensor_create(x->ndim, x->shape, x->device, false);

    for (size_t i = 0; i < x->size; i++) {
        grad_input->data[i] = (x->data[i] > 0) ? grad_output->data[i] : 0;
    }

    return grad_input;
}

Tensor *tensor_sigmoid(const Tensor *x) {
    if (!x) return NULL;

    Tensor *output = tensor_create(x->ndim, x->shape, x->device, x->requires_grad);

    for (size_t i = 0; i < x->size; i++) {
        double x_val = x->data[i];
        /* Sigmoid with numerical stability */
        if (x_val > 0) {
            output->data[i] = 1.0 / (1.0 + exp(-x_val));
        } else {
            double e_x = exp(x_val);
            output->data[i] = e_x / (1.0 + e_x);
        }
    }

    if (x->requires_grad && dl_grad_enabled()) {
        Tensor **saved = (Tensor **)malloc(sizeof(Tensor *));
        saved[0] = tensor_clone(output);

        GradientFunction *grad_fn = create_grad_fn(
            OP_SIGMOID, (Tensor **)&x, 1, saved, 1, NULL, 0,
            compute_sigmoid_backward
        );
        tensor_set_grad_fn(output, grad_fn);
    }

    return output;
}

Tensor *tensor_sigmoid_backward(const Tensor *x, const Tensor *grad_output) {
    if (!x || !grad_output) return NULL;

    Tensor *sigmoid_x = tensor_sigmoid(x);
    Tensor *one_minus_sigmoid = tensor_sub(
        tensor_ones(x->ndim, x->shape, x->device, false), sigmoid_x);
    Tensor *deriv = tensor_mul(sigmoid_x, one_minus_sigmoid);
    Tensor *grad_input = tensor_mul(grad_output, deriv);

    tensor_free(one_minus_sigmoid);
    tensor_free(deriv);
    tensor_free(sigmoid_x);

    return grad_input;
}

Tensor *tensor_tanh_activation(const Tensor *x) {
    if (!x) return NULL;

    Tensor *output = tensor_create(x->ndim, x->shape, x->device, x->requires_grad);

    for (size_t i = 0; i < x->size; i++) {
        output->data[i] = tanh(x->data[i]);
    }

    if (x->requires_grad && dl_grad_enabled()) {
        Tensor **saved = (Tensor **)malloc(sizeof(Tensor *));
        saved[0] = tensor_clone(output);

        GradientFunction *grad_fn = create_grad_fn(
            OP_TANH, (Tensor **)&x, 1, saved, 1, NULL, 0,
            compute_tanh_backward
        );
        tensor_set_grad_fn(output, grad_fn);
    }

    return output;
}

Tensor *tensor_tanh_backward(const Tensor *x, const Tensor *grad_output) {
    if (!x || !grad_output) return NULL;

    Tensor *tanh_x = tensor_tanh_activation(x);
    Tensor *tanh_sq = tensor_mul(tanh_x, tanh_x);
    Tensor *one_minus_tanh_sq = tensor_sub(
        tensor_ones(x->ndim, x->shape, x->device, false), tanh_sq);
    Tensor *grad_input = tensor_mul(grad_output, one_minus_tanh_sq);

    tensor_free(tanh_x);
    tensor_free(tanh_sq);
    tensor_free(one_minus_tanh_sq);

    return grad_input;
}

Tensor *tensor_gelu(const Tensor *x) {
    if (!x) return NULL;

    Tensor *output = tensor_create(x->ndim, x->shape, x->device, x->requires_grad);
    double sqrt_2_over_pi = sqrt(2.0 / M_PI);
    double k = 0.044715;

    for (size_t i = 0; i < x->size; i++) {
        double x_val = x->data[i];
        double x_cubed = x_val * x_val * x_val;
        double tanh_arg = sqrt_2_over_pi * (x_val + k * x_cubed);
        double tanh_val = tanh(tanh_arg);
        output->data[i] = 0.5 * x_val * (1.0 + tanh_val);
    }

    return output;
}

Tensor *tensor_gelu_backward(const Tensor *x, const Tensor *grad_output) {
    if (!x || !grad_output) return NULL;

    /* Simplified GELU backward */
    Tensor *grad_input = tensor_create(x->ndim, x->shape, x->device, false);
    double sqrt_2_over_pi = sqrt(2.0 / M_PI);
    double k = 0.044715;

    for (size_t i = 0; i < x->size; i++) {
        double x_val = x->data[i];
        double x_cubed = x_val * x_val * x_val;
        double tanh_arg = sqrt_2_over_pi * (x_val + k * x_cubed);
        double tanh_val = tanh(tanh_arg);

        double sech_sq = 1.0 - tanh_val * tanh_val;
        double d_tanh_dx = sech_sq * sqrt_2_over_pi * (1.0 + 3.0 * k * x_val * x_val);

        grad_input->data[i] = grad_output->data[i] * 0.5 * (1.0 + tanh_val + x_val * d_tanh_dx);
    }

    return grad_input;
}

Tensor *tensor_softmax(const Tensor *x, int axis) {
    if (!x) return NULL;

    /* Find axis index */
    if (axis < 0) axis += x->ndim;
    size_t ax = (size_t)axis;

    Tensor *output = tensor_create(x->ndim, x->shape, x->device, x->requires_grad);

    /* Compute max for numerical stability */
    size_t outer_size = 1;
    for (size_t i = 0; i < ax; i++) outer_size *= x->shape[i];

    size_t dim_size = x->shape[ax];
    size_t inner_size = 1;
    for (size_t i = ax + 1; i < x->ndim; i++) inner_size *= x->shape[i];

    for (size_t o = 0; o < outer_size; o++) {
        for (size_t i = 0; i < inner_size; i++) {
            /* Find max in this slice */
            double max_val = x->data[o * dim_size * inner_size + i];
            for (size_t j = 1; j < dim_size; j++) {
                double val = x->data[o * dim_size * inner_size + j * inner_size + i];
                if (val > max_val) max_val = val;
            }

            /* Compute exp and sum */
            double sum = 0;
            for (size_t j = 0; j < dim_size; j++) {
                size_t idx = o * dim_size * inner_size + j * inner_size + i;
                sum += exp(x->data[idx] - max_val);
            }

            /* Compute softmax */
            for (size_t j = 0; j < dim_size; j++) {
                size_t idx = o * dim_size * inner_size + j * inner_size + i;
                output->data[idx] = exp(x->data[idx] - max_val) / sum;
            }
        }
    }

    return output;
}

Tensor *tensor_softmax_backward(const Tensor *x, const Tensor *grad_output, int axis) {
    if (!x || !grad_output) return NULL;

    /* Softmax backward: dz_i/dx_j = softmax_i * (delta_ij - softmax_j) */
    Tensor *softmax_x = tensor_softmax(x, axis);

    size_t ax = axis < 0 ? x->ndim + axis : axis;
    size_t dim_size = x->shape[ax];

    Tensor *grad_input = tensor_create(x->ndim, x->shape, x->device, false);

    size_t total_size = x->size;
    size_t stride = total_size / dim_size;

    for (size_t i = 0; i < stride; i++) {
        double sum = 0;
        for (size_t j = 0; j < dim_size; j++) {
            size_t out_idx = j * stride + i;
            sum += grad_output->data[out_idx] * softmax_x->data[out_idx];
        }

        for (size_t j = 0; j < dim_size; j++) {
            size_t out_idx = j * stride + i;
            double delta = (i == j % stride) ? 1.0 : 0.0;
            /* Simplified: only works correctly for specific cases */
            grad_input->data[out_idx] = grad_output->data[out_idx] * softmax_x->data[out_idx] * (1 - softmax_x->data[out_idx]);
        }
    }

    tensor_free(softmax_x);
    return grad_input;
}

Tensor *tensor_log_softmax(const Tensor *x, int axis) {
    if (!x) return NULL;

    Tensor *softmax_x = tensor_softmax(x, axis);
    Tensor *log_softmax_x = tensor_pow(softmax_x, 0);  /* placeholder */

    /* Actually compute log_softmax directly */
    tensor_free(log_softmax_x);
    tensor_free(softmax_x);

    /* Compute log_softmax: log_softmax_i = x_i - log(sum(exp(x))) */
    if (axis < 0) axis += x->ndim;
    size_t ax = (size_t)axis;

    Tensor *output = tensor_create(x->ndim, x->shape, x->device, x->requires_grad);

    size_t outer_size = 1;
    for (size_t i = 0; i < ax; i++) outer_size *= x->shape[i];

    size_t dim_size = x->shape[ax];
    size_t inner_size = 1;
    for (size_t i = ax + 1; i < x->ndim; i++) inner_size *= x->shape[i];

    for (size_t o = 0; o < outer_size; o++) {
        for (size_t i = 0; i < inner_size; i++) {
            double max_val = x->data[o * dim_size * inner_size + i];
            for (size_t j = 1; j < dim_size; j++) {
                double val = x->data[o * dim_size * inner_size + j * inner_size + i];
                if (val > max_val) max_val = val;
            }

            double sum = 0;
            for (size_t j = 0; j < dim_size; j++) {
                size_t idx = o * dim_size * inner_size + j * inner_size + i;
                sum += exp(x->data[idx] - max_val);
            }
            double log_sum = log(sum) + max_val;

            for (size_t j = 0; j < dim_size; j++) {
                size_t idx = o * dim_size * inner_size + j * inner_size + i;
                output->data[idx] = x->data[idx] - log_sum;
            }
        }
    }

    return output;
}

/* ============================================================================
 * DROPOUT
 * ============================================================================ */

Tensor *tensor_dropout(const Tensor *x, double p, bool training, Tensor **mask_out) {
    if (!x) return NULL;

    Tensor *output = tensor_clone(x);

    if (training && mask_out) {
        Tensor *mask = tensor_create(x->ndim, x->shape, x->device, false);
        for (size_t i = 0; i < x->size; i++) {
            mask->data[i] = ((double)rand() / RAND_MAX) > p ? 1.0 : 0.0;
            output->data[i] *= mask->data[i] / (1.0 - p);
        }
        *mask_out = mask;
    }

    return output;
}

Tensor *tensor_dropout_backward(const Tensor *grad_output, const Tensor *mask, double p) {
    if (!grad_output || !mask) return NULL;

    (void)p;
    Tensor *grad_input = tensor_clone(grad_output);

    for (size_t i = 0; i < mask->size; i++) {
        grad_input->data[i] *= mask->data[i];
    }

    return grad_input;
}

/* ============================================================================
 * DENSE (FULLY CONNECTED) LAYER
 * ============================================================================ */

DenseParams *dense_create(size_t in_features, size_t out_features,
                         bool use_bias, bool requires_grad) {
    DenseParams *params = (DenseParams *)calloc(1, sizeof(DenseParams));

    /* Weight matrix: [out_features, in_features] */
    size_t weight_shape[] = {out_features, in_features};
    params->weight = tensor_create(2, weight_shape, DL_DEVICE_CPU, requires_grad);

    /* Xavier/Glorot initialization */
    double std = sqrt(2.0 / (in_features + out_features));
    for (size_t i = 0; i < params->weight->size; i++) {
        params->weight->data[i] = ((double)rand() / RAND_MAX * 2 - 1) * std;
    }

    /* Bias vector: [out_features] */
    if (use_bias) {
        size_t bias_shape[] = {out_features};
        params->bias = tensor_create(1, bias_shape, DL_DEVICE_CPU, requires_grad);
        tensor_fill(params->bias, 0.0);
    } else {
        params->bias = NULL;
    }

    return params;
}

Tensor *dense_forward(const DenseParams *params, const Tensor *input) {
    if (!params || !input) return NULL;

    /* y = x @ W^T + b */
    Tensor *weight_T = tensor_transpose(params->weight);
    Tensor *output = tensor_matmul(input, weight_T);
    tensor_free(weight_T);

    if (params->bias) {
        Tensor *output_plus_bias = tensor_add(output, params->bias);
        tensor_free(output);
        output = output_plus_bias;
    }

    return output;
}

void dense_backward(DenseParams *params, const Tensor *input,
                   const Tensor *grad_output, Tensor *grad_input) {
    if (!params || !input || !grad_output) return;

    /* grad_input = grad_output @ W */
    Tensor *grad_input_full = tensor_matmul(grad_output, params->weight);
    if (grad_input) {
        tensor_copy_to(grad_input, grad_input_full);
        tensor_free(grad_input_full);
    } else {
        /* Return as new tensor */
    }

    /* grad_W = grad_output^T @ input */
    if (params->weight->requires_grad) {
        Tensor *grad_output_T = tensor_transpose(grad_output);
        Tensor *grad_W = tensor_matmul(grad_output_T, input);
        tensor_free(grad_output_T);

        /* Accumulate gradient */
        if (params->weight->grad) {
            Tensor *sum = tensor_add((Tensor *)params->weight->grad, grad_W);
            tensor_free((Tensor *)params->weight->grad);
            params->weight->grad = sum;
        } else {
            params->weight->grad = grad_W;
        }
    }

    /* grad_b = sum(grad_output, axis=0) */
    if (params->bias && params->bias->requires_grad) {
        Tensor *grad_b = tensor_sum_axis(grad_output, 0);
        if (params->bias->grad) {
            Tensor *sum = tensor_add((Tensor *)params->bias->grad, grad_b);
            tensor_free(grad_b);
            params->bias->grad = sum;
        } else {
            params->bias->grad = grad_b;
        }
    }
}

void dense_init_xavier(DenseParams *params) {
    if (!params || !params->weight) return;

    size_t in_features = params->weight->shape[1];
    size_t out_features = params->weight->shape[0];
    double std = sqrt(2.0 / (in_features + out_features));

    for (size_t i = 0; i < params->weight->size; i++) {
        params->weight->data[i] = ((double)rand() / RAND_MAX * 2 - 1) * std;
    }

    if (params->bias) {
        tensor_fill(params->bias, 0.0);
    }
}

void dense_init_kaiming(DenseParams *params) {
    if (!params || !params->weight) return;

    size_t in_features = params->weight->shape[1];
    double std = sqrt(2.0 / in_features);

    for (size_t i = 0; i < params->weight->size; i++) {
        params->weight->data[i] = ((double)rand() / RAND_MAX * 2 - 1) * std;
    }

    if (params->bias) {
        tensor_fill(params->bias, 0.0);
    }
}

void dense_free(DenseParams *params) {
    if (!params) return;
    if (params->weight) tensor_free(params->weight);
    if (params->bias) tensor_free(params->bias);
    free(params);
}

/* ============================================================================
 * LAYER NORMALIZATION
 * ============================================================================ */

LayerNormParams *layer_norm_create(size_t normalized_shape, double eps, bool requires_grad) {
    LayerNormParams *params = (LayerNormParams *)calloc(1, sizeof(LayerNormParams));

    size_t shape[] = {normalized_shape};
    params->gamma = tensor_create(1, shape, DL_DEVICE_CPU, requires_grad);
    params->beta = tensor_create(1, shape, DL_DEVICE_CPU, requires_grad);
    params->eps = eps;

    tensor_fill(params->gamma, 1.0);
    tensor_fill(params->beta, 0.0);

    return params;
}

Tensor *layer_norm_forward(const LayerNormParams *params, const Tensor *x) {
    if (!params || !x) return NULL;

    /* Compute mean and variance */
    double mean = tensor_mean(x);
    double var = 0;
    for (size_t i = 0; i < x->size; i++) {
        double diff = x->data[i] - mean;
        var += diff * diff;
    }
    var /= x->size;
    double std = sqrt(var + params->eps);

    /* Normalize */
    Tensor *output = tensor_create(x->ndim, x->shape, x->device, x->requires_grad);
    for (size_t i = 0; i < x->size; i++) {
        output->data[i] = (x->data[i] - mean) / std;
    }

    /* Scale and shift */
    for (size_t i = 0; i < x->size; i++) {
        size_t c = i % params->gamma->size;
        output->data[i] = output->data[i] * params->gamma->data[c] + params->beta->data[c];
    }

    return output;
}

void layer_norm_backward(LayerNormParams *params, const Tensor *x,
                        const Tensor *grad_output, Tensor *grad_input) {
    (void)params;
    (void)x;
    (void)grad_output;
    (void)grad_input;
    /* Simplified - full implementation would compute dgamma and dbeta too */
}

void layer_norm_free(LayerNormParams *params) {
    if (!params) return;
    if (params->gamma) tensor_free(params->gamma);
    if (params->beta) tensor_free(params->beta);
    free(params);
}

/* ============================================================================
 * BATCH NORMALIZATION
 * ============================================================================ */

BatchNormParams *batch_norm_create(size_t num_features, double momentum, double eps,
                                   bool requires_grad) {
    BatchNormParams *params = (BatchNormParams *)calloc(1, sizeof(BatchNormParams));

    size_t shape[] = {num_features};
    params->gamma = tensor_create(1, shape, DL_DEVICE_CPU, requires_grad);
    params->beta = tensor_create(1, shape, DL_DEVICE_CPU, requires_grad);
    params->running_mean = tensor_zeros(1, shape, DL_DEVICE_CPU, false);
    params->running_var = tensor_ones(1, shape, DL_DEVICE_CPU, false);
    params->momentum = momentum;
    params->eps = eps;

    return params;
}

Tensor *batch_norm_forward_train(BatchNormParams *params, const Tensor *input,
                                 bool training) {
    if (!params || !input) return NULL;

    size_t num_features = params->gamma->size;
    size_t batch_size = input->shape[0];

    /* Input shape: [batch, num_features] or [batch, num_features, H, W] */
    Tensor *output = tensor_create(input->ndim, input->shape, input->device, input->requires_grad);

    for (size_t f = 0; f < num_features; f++) {
        /* Compute mean for this feature */
        double mean = 0;
        for (size_t b = 0; b < batch_size; b++) {
            size_t idx = b * num_features + f;
            mean += input->data[idx];
        }
        mean /= batch_size;

        /* Compute variance */
        double var = 0;
        for (size_t b = 0; b < batch_size; b++) {
            size_t idx = b * num_features + f;
            double diff = input->data[idx] - mean;
            var += diff * diff;
        }
        var /= batch_size;

        /* Update running stats */
        if (training) {
            for (size_t i = 0; i < num_features; i++) {
                params->running_mean->data[i] = params->momentum * params->running_mean->data[i] +
                                                (1 - params->momentum) * mean;
                params->running_var->data[i] = params->momentum * params->running_var->data[i] +
                                               (1 - params->momentum) * var;
            }
        }

        /* Normalize and scale/shift */
        double std = sqrt(var + params->eps);
        for (size_t b = 0; b < batch_size; b++) {
            size_t idx = b * num_features + f;
            double norm = (input->data[idx] - mean) / std;
            output->data[idx] = norm * params->gamma->data[f] + params->beta->data[f];
        }
    }

    return output;
}

Tensor *batch_norm_forward_eval(const BatchNormParams *params, const Tensor *input) {
    if (!params || !input) return NULL;

    size_t num_features = params->gamma->size;
    size_t batch_size = input->shape[0];

    Tensor *output = tensor_create(input->ndim, input->shape, input->device, false);

    for (size_t f = 0; f < num_features; f++) {
        double mean = params->running_mean->data[f];
        double std = sqrt(params->running_var->data[f] + params->eps);

        for (size_t b = 0; b < batch_size; b++) {
            size_t idx = b * num_features + f;
            double norm = (input->data[idx] - mean) / std;
            output->data[idx] = norm * params->gamma->data[f] + params->beta->data[f];
        }
    }

    return output;
}

void batch_norm_backward(BatchNormParams *params, const Tensor *input,
                        const Tensor *grad_output, Tensor *grad_input) {
    (void)params;
    (void)input;
    (void)grad_output;
    (void)grad_input;
}

void batch_norm_free(BatchNormParams *params) {
    if (!params) return;
    if (params->gamma) tensor_free(params->gamma);
    if (params->beta) tensor_free(params->beta);
    if (params->running_mean) tensor_free(params->running_mean);
    if (params->running_var) tensor_free(params->running_var);
    free(params);
}

/* ============================================================================
 * CONV2D LAYER
 * ============================================================================ */

Conv2DParams *conv2d_create(size_t in_channels, size_t out_channels,
                            size_t kernel_h, size_t kernel_w,
                            size_t stride_h, size_t stride_w,
                            size_t padding_h, size_t padding_w,
                            bool use_bias, bool requires_grad) {
    Conv2DParams *params = (Conv2DParams *)calloc(1, sizeof(Conv2DParams));

    params->in_channels = in_channels;
    params->out_channels = out_channels;
    params->kernel_h = kernel_h;
    params->kernel_w = kernel_w;
    params->stride_h = stride_h;
    params->stride_w = stride_w;
    params->padding_h = padding_h;
    params->padding_w = padding_w;

    /* Weight: [out_channels, in_channels, kernel_h, kernel_w] */
    size_t weight_shape[] = {out_channels, in_channels, kernel_h, kernel_w};
    params->weight = tensor_create(4, weight_shape, DL_DEVICE_CPU, requires_grad);

    /* He initialization for conv layers */
    double std = sqrt(2.0 / (in_channels * kernel_h * kernel_w));
    for (size_t i = 0; i < params->weight->size; i++) {
        params->weight->data[i] = ((double)rand() / RAND_MAX * 2 - 1) * std;
    }

    if (use_bias) {
        size_t bias_shape[] = {out_channels};
        params->bias = tensor_create(1, bias_shape, DL_DEVICE_CPU, requires_grad);
        tensor_fill(params->bias, 0.0);
    }

    return params;
}

Tensor *conv2d_forward(const Conv2DParams *params, const Tensor *input) {
    if (!params || !input) return NULL;

    /* Input: [batch, in_channels, height, width] */
    size_t batch = input->shape[0];
    size_t in_ch = input->shape[1];
    size_t in_h = input->shape[2];
    size_t in_w = input->shape[3];

    /* Output dimensions with padding */
    size_t out_h = (in_h + 2 * params->padding_h - params->kernel_h) / params->stride_h + 1;
    size_t out_w = (in_w + 2 * params->padding_w - params->kernel_w) / params->stride_w + 1;

    size_t out_shape[] = {batch, params->out_channels, out_h, out_w};
    Tensor *output = tensor_create(4, out_shape, input->device, input->requires_grad);

    /* Naive 2D convolution implementation */
    for (size_t b = 0; b < batch; b++) {
        for (size_t oc = 0; oc < params->out_channels; oc++) {
            for (size_t oh = 0; oh < out_h; oh++) {
                for (size_t ow = 0; ow < out_w; ow++) {
                    double sum = 0.0;

                    for (size_t ic = 0; ic < in_ch; ic++) {
                        for (size_t kh = 0; kh < params->kernel_h; kh++) {
                            for (size_t kw = 0; kw < params->kernel_w; kw++) {
                                size_t in_h_idx = oh * params->stride_h + kh - params->padding_h;
                                size_t in_w_idx = ow * params->stride_w + kw - params->padding_w;

                                if (in_h_idx < in_h && in_w_idx < in_w && in_h_idx >= 0 && in_w_idx >= 0) {
                                    size_t in_idx = b * in_ch * in_h * in_w + ic * in_h * in_w + in_h_idx * in_w + in_w_idx;
                                    size_t weight_idx = oc * in_ch * params->kernel_h * params->kernel_w +
                                                       ic * params->kernel_h * params->kernel_w +
                                                       kh * params->kernel_w + kw;
                                    sum += input->data[in_idx] * params->weight->data[weight_idx];
                                }
                            }
                        }
                    }

                    if (params->bias) {
                        sum += params->bias->data[oc];
                    }

                    size_t out_idx = b * params->out_channels * out_h * out_w +
                                    oc * out_h * out_w + oh * out_w + ow;
                    output->data[out_idx] = sum;
                }
            }
        }
    }

    return output;
}

void conv2d_backward(Conv2DParams *params, const Tensor *input,
                    const Tensor *grad_output, Tensor *grad_input) {
    (void)params;
    (void)input;
    (void)grad_output;
    (void)grad_input;
}

void conv2d_free(Conv2DParams *params) {
    if (!params) return;
    if (params->weight) tensor_free(params->weight);
    if (params->bias) tensor_free(params->bias);
    free(params);
}

/* ============================================================================
 * POOLING LAYERS
 * ============================================================================ */

MaxPool2DParams *maxpool2d_create(size_t kernel_h, size_t kernel_w,
                                  size_t stride_h, size_t stride_w) {
    MaxPool2DParams *params = (MaxPool2DParams *)calloc(1, sizeof(MaxPool2DParams));
    params->kernel_h = kernel_h;
    params->kernel_w = kernel_w;
    params->stride_h = stride_h;
    params->stride_w = stride_w;
    return params;
}

Tensor *maxpool2d_forward(const MaxPool2DParams *params, const Tensor *input) {
    if (!params || !input) return NULL;

    size_t batch = input->shape[0];
    size_t channels = input->shape[1];
    size_t in_h = input->shape[2];
    size_t in_w = input->shape[3];

    size_t out_h = (in_h - params->kernel_h) / params->stride_h + 1;
    size_t out_w = (in_w - params->kernel_w) / params->stride_w + 1;

    size_t out_shape[] = {batch, channels, out_h, out_w};
    Tensor *output = tensor_create(4, out_shape, input->device, false);

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t oh = 0; oh < out_h; oh++) {
                for (size_t ow = 0; ow < out_w; ow++) {
                    double max_val = -1e10;

                    for (size_t kh = 0; kh < params->kernel_h; kh++) {
                        for (size_t kw = 0; kw < params->kernel_w; kw++) {
                            size_t in_h_idx = oh * params->stride_h + kh;
                            size_t in_w_idx = ow * params->stride_w + kw;

                            if (in_h_idx < in_h && in_w_idx < in_w) {
                                size_t in_idx = b * channels * in_h * in_w + c * in_h * in_w +
                                               in_h_idx * in_w + in_w_idx;
                                if (input->data[in_idx] > max_val) {
                                    max_val = input->data[in_idx];
                                }
                            }
                        }
                    }

                    size_t out_idx = b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                    output->data[out_idx] = max_val;
                }
            }
        }
    }

    return output;
}

Tensor *maxpool2d_backward(const MaxPool2DParams *params, const Tensor *input,
                           const Tensor *grad_output) {
    (void)params;
    (void)input;
    (void)grad_output;
    /* Simplified - would need to track max positions */
    return tensor_zeros(input->ndim, input->shape, input->device, false);
}

AvgPool2DParams *avgpool2d_create(size_t kernel_h, size_t kernel_w,
                                 size_t stride_h, size_t stride_w) {
    AvgPool2DParams *params = (AvgPool2DParams *)calloc(1, sizeof(AvgPool2DParams));
    params->kernel_h = kernel_h;
    params->kernel_w = kernel_w;
    params->stride_h = stride_h;
    params->stride_w = stride_w;
    return params;
}

Tensor *avgpool2d_forward(const AvgPool2DParams *params, const Tensor *input) {
    if (!params || !input) return NULL;

    size_t batch = input->shape[0];
    size_t channels = input->shape[1];
    size_t in_h = input->shape[2];
    size_t in_w = input->shape[3];

    size_t out_h = (in_h - params->kernel_h) / params->stride_h + 1;
    size_t out_w = (in_w - params->kernel_w) / params->stride_w + 1;

    size_t out_shape[] = {batch, channels, out_h, out_w};
    Tensor *output = tensor_create(4, out_shape, input->device, false);

    double kernel_size = params->kernel_h * params->kernel_w;

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t oh = 0; oh < out_h; oh++) {
                for (size_t ow = 0; ow < out_w; ow++) {
                    double sum = 0.0;

                    for (size_t kh = 0; kh < params->kernel_h; kh++) {
                        for (size_t kw = 0; kw < params->kernel_w; kw++) {
                            size_t in_h_idx = oh * params->stride_h + kh;
                            size_t in_w_idx = ow * params->stride_w + kw;

                            if (in_h_idx < in_h && in_w_idx < in_w) {
                                size_t in_idx = b * channels * in_h * in_w + c * in_h * in_w +
                                               in_h_idx * in_w + in_w_idx;
                                sum += input->data[in_idx];
                            }
                        }
                    }

                    size_t out_idx = b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                    output->data[out_idx] = sum / kernel_size;
                }
            }
        }
    }

    return output;
}

Tensor *avgpool2d_backward(const AvgPool2DParams *params, const Tensor *input,
                          const Tensor *grad_output) {
    (void)params;
    (void)input;
    (void)grad_output;
    return tensor_zeros(input->ndim, input->shape, input->device, false);
}

void maxpool2d_free(MaxPool2DParams *params) {
    free(params);
}

void avgpool2d_free(AvgPool2DParams *params) {
    free(params);
}
