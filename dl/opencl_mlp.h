#ifndef __OPENCL_MLP_H__
#define __OPENCL_MLP_H__

/* ============================================================================
 * OpenCL-based Multi-Layer Perceptron (MLP)
 * Uses opencl_tensor.h for GPU-accelerated computation
 * ============================================================================ */

#include "opencl_tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * FC Layer with GPU Cache
 * ============================================================================ */

typedef struct {
    CLTensor* weight;      // [out_features, in_features]
    CLTensor* bias;        // [out_features]
    CLTensor* grad_w;      // gradient for weight
    CLTensor* grad_b;      // gradient for bias

    // Cache for backward pass (stored on GPU)
    CLTensor* input_cache;    // input to FC
    CLTensor* relu_mask;      // mask for ReLU: 1 if x > 0, 0 otherwise
} CLFCLayer;

/* ============================================================================
 * Optimizer
 * ============================================================================ */

typedef struct {
    float lr;
    float momentum;
    float weight_decay;
} CLSGDConfig;

typedef struct {
    CLSGDConfig config;
    CLTensor** velocity_w;  // momentum buffers for weights
    CLTensor** velocity_b;  // momentum buffers for biases
    size_t num_layers;
} CLSGDOptimizer;

/* ============================================================================
 * MLP Network
 * ============================================================================ */

typedef struct {
    CLFCLayer** layers;
    size_t num_layers;
    size_t input_dim;
    size_t output_dim;
} CLOpenCLMLP;

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/* Clone a CLTensor by downloading and re-uploading */
static CLTensor* cl_tensor_clone(CLOpenCL* cl, const CLTensor* t) {
    float* data = (float*)malloc(t->nbytes);
    cl_tensor_download(t, data);
    CLTensor* copy = cl_tensor_create_from_host(cl, t->dtype, t->layout, t->shape, t->ndim, data);
    free(data);
    return copy;
}

/* ============================================================================
 * Layer Operations
 * ============================================================================ */

static CLFCLayer* cl_fc_layer_create(CLOpenCL* cl, size_t in_features, size_t out_features) {
    CLFCLayer* layer = (CLFCLayer*)malloc(sizeof(CLFCLayer));

    // Initialize weights with Xavier initialization
    size_t w_shape[] = {out_features, in_features};
    float* h_weight = (float*)malloc(out_features * in_features * sizeof(float));
    float std = sqrtf(2.0f / (in_features + out_features));

    // Xavier initialization using Gaussian (Box-Muller)
    for (size_t i = 0; i < out_features * in_features; i++) {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        float z = sqrtf(-2.0f * logf(u1 + 1e-10f)) * cosf(2.0f * M_PI * u2);
        h_weight[i] = z * std;
    }
    layer->weight = cl_tensor_create_from_host(cl, CL_TENSOR_DTYPE_F32,
                                                CL_TENSOR_LAYOUT_NCHW, w_shape, 2, h_weight);
    free(h_weight);

    // Initialize bias with zeros
    size_t b_shape[] = {out_features};
    float* h_bias = (float*)calloc(out_features, sizeof(float));
    layer->bias = cl_tensor_create_from_host(cl, CL_TENSOR_DTYPE_F32,
                                             CL_TENSOR_LAYOUT_NCHW, b_shape, 1, h_bias);
    free(h_bias);

    // Gradient buffers
    layer->grad_w = cl_tensor_create(cl, CL_TENSOR_DTYPE_F32, CL_TENSOR_LAYOUT_NCHW, w_shape, 2);
    layer->grad_b = cl_tensor_create(cl, CL_TENSOR_DTYPE_F32, CL_TENSOR_LAYOUT_NCHW, b_shape, 1);

    // Cache buffers
    layer->input_cache = NULL;
    layer->relu_mask = NULL;

    return layer;
}

static void cl_fc_layer_free(CLFCLayer* layer) {
    if (!layer) return;
    if (layer->weight) cl_tensor_free(layer->weight);
    if (layer->bias) cl_tensor_free(layer->bias);
    if (layer->grad_w) cl_tensor_free(layer->grad_w);
    if (layer->grad_b) cl_tensor_free(layer->grad_b);
    if (layer->input_cache) cl_tensor_free(layer->input_cache);
    if (layer->relu_mask) cl_tensor_free(layer->relu_mask);
    free(layer);
}

/* --------------------------------------------------------------------------
 * Forward: y = x @ W.T + b (GPU)
 * -------------------------------------------------------------------------- */
static CLTensor* cl_fc_layer_forward(CLOpenCL* cl, cl_kernel_cache_t* cache,
                                     CLFCLayer* layer, const CLTensor* x) {
    size_t batch = x->shape[0];
    size_t out_features = layer->weight->shape[0];
    size_t in_features = layer->weight->shape[1];

    // Save input cache for backward (GPU copy)
    layer->input_cache = cl_tensor_create(cl, x->dtype, x->layout, x->shape, x->ndim);
    cl_tensor_copy_impl(cl, cache, layer->input_cache, x);

    // Compute preact = x @ W.T via GPU GEMM
    // weight shape [out_features, in_features], needs W.T -> [in_features, out_features]
    // But our gemm kernel expects A[M,K] @ B[K,N] = C[M,N]
    // x is [batch, in_features] = [M, K], weight is [out, in] = [K, N]... wait
    // Let's construct: x @ W^T where W^T = cl_tensor_transpose(weight)
    // Actually simpler: create weight_T as transpose [in, out], then matmul
    CLTensor* weight_T = cl_tensor_create(cl, layer->weight->dtype, layer->weight->layout,
                                          (size_t[]){in_features, out_features}, 2);
    // TODO: GPU transpose kernel; for now download-transpose-upload
    float* h_w = (float*)malloc(layer->weight->nbytes);
    cl_tensor_download(layer->weight, h_w);
    float* h_wt = (float*)malloc(layer->weight->nbytes);
    for (size_t o = 0; o < out_features; o++) {
        for (size_t i = 0; i < in_features; i++) {
            h_wt[i * out_features + o] = h_w[o * in_features + i];
        }
    }
    cl_tensor_upload(weight_T, h_wt);
    free(h_w); free(h_wt);

    CLTensor* preact = cl_tensor_matmul_impl(cl, cache, x, weight_T);
    cl_tensor_free(weight_T);

    // Add bias via GPU
    size_t out_shape[] = {batch, out_features};
    CLTensor* output = cl_tensor_create(cl, CL_TENSOR_DTYPE_F32, x->layout, out_shape, 2);
    // bias_add modifies inout; copy preact to output first
    cl_tensor_copy_impl(cl, cache, output, preact);
    cl_tensor_bias_add_impl(cl, cache, output, layer->bias);
    cl_tensor_free(preact);

    return output;
}

/* --------------------------------------------------------------------------
 * ReLU forward: y = max(0, x), saves mask
 * -------------------------------------------------------------------------- */
static CLTensor* cl_relu_forward(CLOpenCL* cl, cl_kernel_cache_t* cache,
                                  CLTensor* x, CLTensor** mask_out) {
    (void)cache;
    float* data = (float*)malloc(x->nbytes);
    cl_tensor_download(x, data);

    float* mask_data = (float*)malloc(x->nbytes);
    for (size_t i = 0; i < x->size; i++) {
        mask_data[i] = (data[i] > 0) ? 1.0f : 0.0f;
        if (data[i] < 0) data[i] = 0;
    }
    cl_tensor_upload(x, data);

    CLTensor* mask_result = cl_tensor_create(cl, x->dtype, x->layout, x->shape, x->ndim);
    cl_tensor_upload(mask_result, mask_data);

    free(data);
    free(mask_data);
    *mask_out = mask_result;
    return x;
}

/* --------------------------------------------------------------------------
 * ReLU backward
 * -------------------------------------------------------------------------- */
static CLTensor* cl_relu_backward(CLOpenCL* cl, cl_kernel_cache_t* cache,
                                    const CLTensor* grad_output, const CLTensor* mask) {
    CLTensor* grad = cl_tensor_create(cl, CL_TENSOR_DTYPE_F32,
                                       grad_output->layout, grad_output->shape, grad_output->ndim);
    // GPU element-wise multiply
    cl_tensor_elem_mul_impl(cl, cache, grad_output, mask, grad);
    return grad;
}

/* --------------------------------------------------------------------------
 * Softmax
 * -------------------------------------------------------------------------- */
static CLTensor* cl_softmax_forward(CLOpenCL* cl, CLTensor* x, size_t axis) {
    return cl_tensor_softmax(cl, x, axis);
}

/* ============================================================================
 * MLP Network
 * ============================================================================ */

static CLOpenCLMLP* cl_mlp_create(CLOpenCL* cl, size_t input_dim, size_t hidden_dim,
                                   size_t output_dim, size_t num_layers) {
    CLOpenCLMLP* mlp = (CLOpenCLMLP*)malloc(sizeof(CLOpenCLMLP));
    mlp->input_dim = input_dim;
    mlp->output_dim = output_dim;
    mlp->num_layers = num_layers;
    mlp->layers = (CLFCLayer**)malloc(sizeof(CLFCLayer*) * num_layers);

    size_t prev_dim = input_dim;
    for (size_t i = 0; i < num_layers; i++) {
        size_t curr_dim = (i == num_layers - 1) ? output_dim : hidden_dim;
        mlp->layers[i] = cl_fc_layer_create(cl, prev_dim, curr_dim);
        prev_dim = curr_dim;
    }

    return mlp;
}

static void cl_mlp_free(CLOpenCLMLP* mlp) {
    if (!mlp) return;
    for (size_t i = 0; i < mlp->num_layers; i++) {
        cl_fc_layer_free(mlp->layers[i]);
    }
    free(mlp->layers);
    free(mlp);
}

/* --------------------------------------------------------------------------
 * Forward Pass (GPU)
 * -------------------------------------------------------------------------- */
static CLTensor* cl_mlp_forward(CLOpenCL* cl, cl_kernel_cache_t* cache,
                                 CLOpenCLMLP* mlp, const CLTensor* input) {
    // Clone input to GPU
    float* h_input = (float*)malloc(input->nbytes);
    cl_tensor_download(input, h_input);
    CLTensor* x = cl_tensor_create_from_host(cl, input->dtype, input->layout,
                                              input->shape, input->ndim, h_input);
    free(h_input);

    for (size_t i = 0; i < mlp->num_layers; i++) {
        CLFCLayer* layer = mlp->layers[i];

        CLTensor* preact = cl_fc_layer_forward(cl, cache, layer, x);
        cl_tensor_free(x);
        x = preact;

        if (i < mlp->num_layers - 1) {
            x = cl_relu_forward(cl, cache, x, &layer->relu_mask);
        }
    }

    CLTensor* output = cl_tensor_softmax_impl(cl, cache, x);
    cl_tensor_free(x);

    return output;
}

/* --------------------------------------------------------------------------
 * Cross-Entropy Gradient
 * -------------------------------------------------------------------------- */
static CLTensor* cl_cross_entropy_grad(CLOpenCL* cl, cl_kernel_cache_t* cache,
                                         const CLTensor* pred, const int* targets,
                                         size_t batch, size_t num_classes) {
    CLTensor* grad = cl_tensor_create(cl, CL_TENSOR_DTYPE_F32,
                                        pred->layout, pred->shape, pred->ndim);
    cl_tensor_cross_entropy_grad_impl(cl, cache, grad, pred, targets, batch, num_classes);
    return grad;
}

/* --------------------------------------------------------------------------
 * Backward Pass (GPU)
 * -------------------------------------------------------------------------- */
static void cl_mlp_backward(CLOpenCL* cl, cl_kernel_cache_t* cache,
                             CLOpenCLMLP* mlp, const CLTensor* grad_output) {
    CLTensor* grad = cl_tensor_clone(cl, grad_output);

    for (int i = (int)mlp->num_layers - 1; i >= 0; i--) {
        CLFCLayer* layer = mlp->layers[i];

        if (i < (int)mlp->num_layers - 1 && layer->relu_mask) {
            CLTensor* new_grad = cl_relu_backward(cl, cache, grad, layer->relu_mask);
            cl_tensor_free(grad);
            grad = new_grad;
        }

        size_t batch = layer->input_cache->shape[0];
        size_t out_features = layer->weight->shape[0];
        size_t in_features = layer->weight->shape[1];

        // Download weight to host for transpose
        float* h_weight = (float*)malloc(layer->weight->nbytes);
        cl_tensor_download(layer->weight, h_weight);

        // Transpose weight to [in, out] for grad @ W
        float* h_weight_T = (float*)malloc(layer->weight->nbytes);
        for (size_t o = 0; o < out_features; o++) {
            for (size_t inn = 0; inn < in_features; inn++) {
                h_weight_T[inn * out_features + o] = h_weight[o * in_features + inn];
            }
        }
        cl_tensor_upload(layer->grad_w, h_weight_T);  // reuse grad_w buffer for transpose
        free(h_weight); free(h_weight_T);

        CLTensor* weight_T_gpu = cl_tensor_create(cl, layer->weight->dtype, layer->weight->layout,
                                                  (size_t[]){in_features, out_features}, 2);
        float* h_wt2 = (float*)malloc(layer->weight->nbytes);
        cl_tensor_download(layer->grad_w, h_wt2);  // get transposed
        cl_tensor_upload(weight_T_gpu, h_wt2);
        free(h_wt2);

        // grad_input = grad @ W^T (weight stored [out, in])
        // Need W as [in, out], so we have weight_T_gpu [in, out]
        CLTensor* grad_input = cl_tensor_matmul_impl(cl, cache, grad, weight_T_gpu);
        cl_tensor_free(weight_T_gpu);

        // grad_w = (grad).T @ input_cache => shape [out, in]
        // grad is [batch, out], input_cache is [batch, in]
        // grad.T is [out, batch], input_cache is [batch, in]
        // (grad).T @ input = [out, batch] @ [batch, in] = [out, in]
        CLTensor* grad_T = cl_tensor_create(cl, grad->dtype, grad->layout,
                                            (size_t[]){out_features, batch}, 2);
        float* h_g = (float*)malloc(grad->nbytes);
        cl_tensor_download(grad, h_g);
        float* h_gt = (float*)malloc(out_features * batch * sizeof(float));
        for (size_t b = 0; b < batch; b++) {
            for (size_t o = 0; o < out_features; o++) {
                h_gt[o * batch + b] = h_g[b * out_features + o];
            }
        }
        cl_tensor_upload(grad_T, h_gt);
        free(h_g); free(h_gt);

        CLTensor* grad_w_gpu = cl_tensor_matmul_impl(cl, cache, grad_T, layer->input_cache);
        cl_tensor_free(grad_T);

        // Copy grad_w to layer->grad_w (no batch scaling for vanilla SGD)
        float* h_gw = (float*)malloc(grad_w_gpu->nbytes);
        cl_tensor_download(grad_w_gpu, h_gw);
        cl_tensor_upload(layer->grad_w, h_gw);
        free(h_gw);
        cl_tensor_free(grad_w_gpu);

        // grad_b = sum over batch (reduce axis=0)
        // Download grad, sum over batch, store to grad_b
        float* h_grad_all = (float*)malloc(grad->nbytes);
        cl_tensor_download(grad, h_grad_all);
        float* h_gb = (float*)calloc(out_features, sizeof(float));
        for (size_t b = 0; b < batch; b++) {
            for (size_t o = 0; o < out_features; o++) {
                h_gb[o] += h_grad_all[b * out_features + o];
            }
        }
        cl_tensor_upload(layer->grad_b, h_gb);
        free(h_grad_all); free(h_gb);

        // grad = grad_input for next iteration
        cl_tensor_free(grad);
        grad = grad_input;
    }

    cl_tensor_free(grad);
}

/* --------------------------------------------------------------------------
 * Gradient Clipping
 * -------------------------------------------------------------------------- */
static void cl_clip_gradients(CLOpenCL* cl, CLTensor* grad, float max_norm) {
    if (!grad) return;
    float* h_grad = (float*)malloc(grad->nbytes);
    cl_tensor_download(grad, h_grad);

    float norm_sq = 0.0f;
    for (size_t i = 0; i < grad->size; i++) norm_sq += h_grad[i] * h_grad[i];
    float norm = sqrtf(norm_sq);

    if (norm > max_norm) {
        float scale = max_norm / norm;
        for (size_t i = 0; i < grad->size; i++) h_grad[i] *= scale;
        cl_tensor_upload(grad, h_grad);
    }
    free(h_grad);
}

/* --------------------------------------------------------------------------
 * Update Parameters (GPU)
 * -------------------------------------------------------------------------- */
static void cl_mlp_update(CLOpenCL* cl, cl_kernel_cache_t* cache,
                           CLOpenCLMLP* mlp, CLSGDOptimizer* opt) {
    CLSGDConfig* cfg = &opt->config;

    for (size_t i = 0; i < mlp->num_layers; i++) {
        CLFCLayer* layer = mlp->layers[i];

        cl_clip_gradients(cl, layer->grad_w, 5.0f);
        cl_clip_gradients(cl, layer->grad_b, 5.0f);

        // SGD update on GPU
        if (opt->velocity_w && opt->velocity_b) {
            cl_tensor_sgd_update_impl(cl, cache, layer->weight, layer->grad_w,
                                      opt->velocity_w[i], cfg->lr, cfg->momentum, cfg->weight_decay);
            cl_tensor_sgd_update_impl(cl, cache, layer->bias, layer->grad_b,
                                      opt->velocity_b[i], cfg->lr, cfg->momentum, 0.0f);
        } else {
            // No momentum: pass NULL for velocity
            CLTensor* null_vel_w = NULL;
            CLTensor* null_vel_b = NULL;
            cl_tensor_sgd_update_impl(cl, cache, layer->weight, layer->grad_w,
                                      null_vel_w, cfg->lr, 0.0f, cfg->weight_decay);
            cl_tensor_sgd_update_impl(cl, cache, layer->bias, layer->grad_b,
                                      null_vel_b, cfg->lr, 0.0f, 0.0f);
        }
    }
}

/* --------------------------------------------------------------------------
 * Training Step
 * -------------------------------------------------------------------------- */
static float cl_mlp_train_step(CLOpenCL* cl, cl_kernel_cache_t* cache,
                                 CLOpenCLMLP* mlp, CLSGDOptimizer* opt,
                                 const CLTensor* input, const int* targets) {
    CLTensor* output = cl_mlp_forward(cl, cache, mlp, input);

    float* h_output = (float*)malloc(output->nbytes);
    cl_tensor_download(output, h_output);

    size_t batch = output->shape[0];
    size_t num_classes = output->shape[1];
    float loss = 0.0f;
    for (size_t b = 0; b < batch; b++) {
        size_t target_class = (size_t)targets[b];
        float pred_prob = h_output[b * num_classes + target_class];
        if (pred_prob < 1e-10f) pred_prob = 1e-10f;
        loss += -logf(pred_prob);
    }
    loss /= batch;

    free(h_output);

    CLTensor* grad = cl_cross_entropy_grad(cl, cache, output, targets, batch, num_classes);
    cl_mlp_backward(cl, cache, mlp, grad);
    cl_mlp_update(cl, cache, mlp, opt);

    cl_tensor_free(output);
    cl_tensor_free(grad);

    return loss;
}

/* --------------------------------------------------------------------------
 * Prediction
 * -------------------------------------------------------------------------- */
static CLTensor* cl_mlp_predict(CLOpenCL* cl, cl_kernel_cache_t* cache,
                                 CLOpenCLMLP* mlp, const CLTensor* input) {
    return cl_mlp_forward(cl, cache, mlp, input);
}

/* --------------------------------------------------------------------------
 * Accuracy
 * -------------------------------------------------------------------------- */
static float cl_mlp_accuracy(CLOpenCL* cl, cl_kernel_cache_t* cache,
                              CLOpenCLMLP* mlp,
                              const CLTensor* input, const int* targets) {
    CLTensor* pred = cl_mlp_predict(cl, cache, mlp, input);

    float* h_pred = (float*)malloc(pred->nbytes);
    cl_tensor_download(pred, h_pred);

    size_t batch = pred->shape[0];
    size_t num_classes = pred->shape[1];
    size_t correct = 0;

    for (size_t b = 0; b < batch; b++) {
        size_t pred_class = 0;
        float pred_max = h_pred[b * num_classes];
        for (size_t c = 1; c < num_classes; c++) {
            if (h_pred[b * num_classes + c] > pred_max) {
                pred_max = h_pred[b * num_classes + c];
                pred_class = c;
            }
        }
        size_t target_class = (size_t)targets[b];
        if (pred_class == target_class) correct++;
    }

    free(h_pred);
    cl_tensor_free(pred);

    return (float)correct / batch;
}

/* ============================================================================
 * SGD Optimizer
 * ============================================================================ */

static CLSGDOptimizer* cl_sgd_create(CLOpenCL* cl, CLOpenCLMLP* mlp,
                                      float lr, float momentum, float weight_decay) {
    (void)cl;  // cl not used directly here
    CLSGDOptimizer* opt = (CLSGDOptimizer*)malloc(sizeof(CLSGDOptimizer));
    opt->config.lr = lr;
    opt->config.momentum = momentum;
    opt->config.weight_decay = weight_decay;
    opt->num_layers = mlp->num_layers;

    if (momentum > 0) {
        opt->velocity_w = (CLTensor**)malloc(sizeof(CLTensor*) * mlp->num_layers);
        opt->velocity_b = (CLTensor**)malloc(sizeof(CLTensor*) * mlp->num_layers);

        for (size_t i = 0; i < mlp->num_layers; i++) {
            CLFCLayer* layer = mlp->layers[i];
            size_t w_shape[] = {layer->weight->shape[0], layer->weight->shape[1]};
            float* zeros_w = (float*)calloc(layer->weight->size, sizeof(float));
            float* zeros_b = (float*)calloc(layer->bias->size, sizeof(float));

            // Note: We can't create from host here without cl context, so we create empty
            // and fill with zeros using the cl context
            opt->velocity_w[i] = cl_tensor_create(cl, CL_TENSOR_DTYPE_F32,
                                                   layer->weight->layout, w_shape, 2);
            opt->velocity_b[i] = cl_tensor_create(cl, CL_TENSOR_DTYPE_F32,
                                                   layer->bias->layout,
                                                   (size_t[]){layer->bias->shape[0]}, 1);
            // Fill with zeros
            float* zeros_w_big = (float*)calloc(opt->velocity_w[i]->size, sizeof(float));
            float* zeros_b_big = (float*)calloc(opt->velocity_b[i]->size, sizeof(float));
            cl_tensor_upload(opt->velocity_w[i], zeros_w_big);
            cl_tensor_upload(opt->velocity_b[i], zeros_b_big);
            free(zeros_w_big);
            free(zeros_b_big);
            free(zeros_w);
            free(zeros_b);
        }
    } else {
        opt->velocity_w = NULL;
        opt->velocity_b = NULL;
    }

    return opt;
}

static void cl_sgd_free(CLSGDOptimizer* opt) {
    if (!opt) return;
    if (opt->velocity_w) {
        for (size_t i = 0; i < opt->num_layers; i++) {
            if (opt->velocity_w[i]) cl_tensor_free(opt->velocity_w[i]);
            if (opt->velocity_b[i]) cl_tensor_free(opt->velocity_b[i]);
        }
        free(opt->velocity_w);
        free(opt->velocity_b);
    }
    free(opt);
}

#endif /* __OPENCL_MLP_H__ */