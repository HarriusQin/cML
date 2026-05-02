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
static CLTensor* cl_fc_layer_forward(CLOpenCL* cl, CLFCLayer* layer, const CLTensor* x) {
    size_t batch = x->shape[0];
    size_t out_features = layer->weight->shape[0];
    size_t in_features = layer->weight->shape[1];

    // Save input for backward
    float* h_x_save = (float*)malloc(x->nbytes);
    cl_tensor_download(x, h_x_save);
    layer->input_cache = cl_tensor_create_from_host(cl, CL_TENSOR_DTYPE_F32,
                                                     x->layout, x->shape, x->ndim, h_x_save);
    free(h_x_save);

    // Create output tensor
    size_t out_shape[] = {batch, out_features};
    CLTensor* preact = cl_tensor_create(cl, CL_TENSOR_DTYPE_F32, x->layout, out_shape, 2);

    // Download x and weight to host for GEMM
    float* h_x = (float*)malloc(x->nbytes);
    float* h_w = (float*)malloc(layer->weight->nbytes);
    float* h_preact = (float*)malloc(preact->nbytes);
    cl_tensor_download(x, h_x);
    cl_tensor_download(layer->weight, h_w);

    // Compute y = x @ W.T = y[b,o] = sum_i x[b,i] * W[o,i]
    for (size_t b = 0; b < batch; b++) {
        for (size_t o = 0; o < out_features; o++) {
            float sum = 0.0f;
            for (size_t i = 0; i < in_features; i++) {
                sum += h_x[b * in_features + i] * h_w[o * in_features + i];
            }
            h_preact[b * out_features + o] = sum;
        }
    }

    // Add bias
    float* h_bias = (float*)malloc(layer->bias->nbytes);
    cl_tensor_download(layer->bias, h_bias);
    for (size_t b = 0; b < batch; b++) {
        for (size_t o = 0; o < out_features; o++) {
            h_preact[b * out_features + o] += h_bias[o];
        }
    }
    free(h_bias);

    // Upload result
    cl_tensor_upload(preact, h_preact);
    free(h_x);
    free(h_w);
    free(h_preact);

    return preact;
}

/* --------------------------------------------------------------------------
 * ReLU forward: y = max(0, x), saves mask
 * -------------------------------------------------------------------------- */
static CLTensor* cl_relu_forward(CLOpenCL* cl, CLTensor* x, CLTensor** mask_out) {
    CLTensor* mask = cl_tensor_create(cl, CL_TENSOR_DTYPE_F32, x->layout, x->shape, x->ndim);
    *mask_out = mask;

    float* h_x = (float*)malloc(x->nbytes);
    float* h_mask = (float*)malloc(x->nbytes);
    cl_tensor_download(x, h_x);

    for (size_t i = 0; i < x->size; i++) {
        h_mask[i] = (h_x[i] > 0) ? 1.0f : 0.0f;
        if (h_x[i] < 0) h_x[i] = 0;
    }

    cl_tensor_upload(mask, h_mask);
    cl_tensor_upload(x, h_x);
    free(h_x);
    free(h_mask);

    return x;
}

/* --------------------------------------------------------------------------
 * ReLU backward
 * -------------------------------------------------------------------------- */
static CLTensor* cl_relu_backward(CLOpenCL* cl, const CLTensor* grad_output, const CLTensor* mask) {
    CLTensor* grad = cl_tensor_create(cl, CL_TENSOR_DTYPE_F32,
                                       grad_output->layout, grad_output->shape, grad_output->ndim);

    float* h_grad_out = (float*)malloc(grad_output->nbytes);
    float* h_mask = (float*)malloc(mask->nbytes);
    float* h_grad = (float*)malloc(grad->nbytes);
    cl_tensor_download(grad_output, h_grad_out);
    cl_tensor_download(mask, h_mask);

    for (size_t i = 0; i < grad_output->size; i++) {
        h_grad[i] = h_grad_out[i] * h_mask[i];
    }

    cl_tensor_upload(grad, h_grad);
    free(h_grad_out);
    free(h_mask);
    free(h_grad);

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
static CLTensor* cl_mlp_forward(CLOpenCL* cl, CLOpenCLMLP* mlp, const CLTensor* input) {
    // Clone input to GPU
    float* h_input = (float*)malloc(input->nbytes);
    cl_tensor_download(input, h_input);
    CLTensor* x = cl_tensor_create_from_host(cl, input->dtype, input->layout,
                                              input->shape, input->ndim, h_input);
    free(h_input);

    for (size_t i = 0; i < mlp->num_layers; i++) {
        CLFCLayer* layer = mlp->layers[i];

        CLTensor* preact = cl_fc_layer_forward(cl, layer, x);
        // Don't store preact in layer_preacts since it will be freed
        // mlp->layer_preacts[i] = preact;
        cl_tensor_free(x);
        x = preact;

        if (i < mlp->num_layers - 1) {
            x = cl_relu_forward(cl, x, &layer->relu_mask);
        }
        // Don't store x in layer_outputs since it will be freed
        // mlp->layer_outputs[i] = x;
    }

    CLTensor* output = cl_softmax_forward(cl, x, 1);
    cl_tensor_free(x);

    return output;
}

/* --------------------------------------------------------------------------
 * Cross-Entropy Gradient
 * -------------------------------------------------------------------------- */
static CLTensor* cl_cross_entropy_grad(CLOpenCL* cl, const CLTensor* pred, const CLTensor* targets) {
    size_t batch = pred->shape[0];
    size_t num_classes = pred->shape[1];

    float* h_pred = (float*)malloc(pred->nbytes);
    float* h_target = (float*)malloc(targets->nbytes);
    cl_tensor_download(pred, h_pred);
    cl_tensor_download(targets, h_target);

    float* h_grad = (float*)malloc(batch * num_classes * sizeof(float));
    for (size_t b = 0; b < batch; b++) {
        size_t target_class = (size_t)h_target[b];
        for (size_t c = 0; c < num_classes; c++) {
            h_grad[b * num_classes + c] = h_pred[b * num_classes + c];
            if (c == target_class) {
                h_grad[b * num_classes + c] -= 1.0f;
            }
        }
    }

    size_t grad_shape[] = {batch, num_classes};
    CLTensor* grad = cl_tensor_create_from_host(cl, CL_TENSOR_DTYPE_F32,
                                                  pred->layout, grad_shape, 2, h_grad);
    free(h_pred);
    free(h_target);
    free(h_grad);
    return grad;
}

/* --------------------------------------------------------------------------
 * Backward Pass
 * -------------------------------------------------------------------------- */
static void cl_mlp_backward(CLOpenCL* cl, CLOpenCLMLP* mlp, const CLTensor* grad_output) {
    CLTensor* grad = cl_tensor_clone(cl, grad_output);

    for (int i = (int)mlp->num_layers - 1; i >= 0; i--) {
        CLFCLayer* layer = mlp->layers[i];

        if (i < (int)mlp->num_layers - 1 && layer->relu_mask) {
            CLTensor* new_grad = cl_relu_backward(cl, grad, layer->relu_mask);
            cl_tensor_free(grad);
            grad = new_grad;
        }

        size_t batch = layer->input_cache->shape[0];
        size_t out_features = layer->weight->shape[0];
        size_t in_features = layer->weight->shape[1];

        float* h_grad = (float*)malloc(grad->nbytes);
        float* h_input = (float*)malloc(layer->input_cache->nbytes);
        float* h_weight = (float*)malloc(layer->weight->nbytes);
        cl_tensor_download(grad, h_grad);
        cl_tensor_download(layer->input_cache, h_input);
        cl_tensor_download(layer->weight, h_weight);

        // grad_w = (grad).T @ input
        float* h_grad_w = (float*)calloc(out_features * in_features, sizeof(float));
        float* h_grad_b = (float*)calloc(out_features, sizeof(float));

        for (size_t o = 0; o < out_features; o++) {
            for (size_t b = 0; b < batch; b++) {
                h_grad_b[o] += h_grad[b * out_features + o];
                for (size_t inn = 0; inn < in_features; inn++) {
                    h_grad_w[o * in_features + inn] += h_grad[b * out_features + o] * h_input[b * in_features + inn];
                }
            }
        }

        // Scale by batch
        float inv_batch = 1.0f / batch;
        for (size_t j = 0; j < out_features * in_features; j++) {
            h_grad_w[j] *= inv_batch;
        }
        for (size_t j = 0; j < out_features; j++) {
            h_grad_b[j] *= inv_batch;
        }

        cl_tensor_upload(layer->grad_w, h_grad_w);
        cl_tensor_upload(layer->grad_b, h_grad_b);

        // Compute gradient for previous layer: grad @ W
        float* h_grad_input = (float*)malloc(batch * in_features * sizeof(float));
        for (size_t b = 0; b < batch; b++) {
            for (size_t inn = 0; inn < in_features; inn++) {
                float sum = 0.0f;
                for (size_t o = 0; o < out_features; o++) {
                    sum += h_grad[b * out_features + o] * h_weight[o * in_features + inn];
                }
                h_grad_input[b * in_features + inn] = sum;
            }
        }

        free(h_grad);
        free(h_input);
        free(h_weight);
        free(h_grad_w);
        free(h_grad_b);

        cl_tensor_free(grad);
        size_t grad_shape[] = {batch, in_features};
        grad = cl_tensor_create_from_host(cl, CL_TENSOR_DTYPE_F32,
                                           CL_TENSOR_LAYOUT_NCHW, grad_shape, 2, h_grad_input);
        free(h_grad_input);
    }

    cl_tensor_free(grad);
}

/* --------------------------------------------------------------------------
 * Gradient Clipping
 * -------------------------------------------------------------------------- */
static void cl_clip_gradients(CLOpenCL* cl, CLTensor* grad, float max_norm) {
    (void)cl;  // cl not used in this implementation
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
 * Update Parameters
 * -------------------------------------------------------------------------- */
static void cl_mlp_update(CLOpenCL* cl, CLOpenCLMLP* mlp, CLSGDOptimizer* opt) {
    CLSGDConfig* cfg = &opt->config;

    for (size_t i = 0; i < mlp->num_layers; i++) {
        CLFCLayer* layer = mlp->layers[i];

        cl_clip_gradients(cl, layer->grad_w, 5.0f);
        cl_clip_gradients(cl, layer->grad_b, 5.0f);

        float* h_w = (float*)malloc(layer->weight->nbytes);
        float* h_b = (float*)malloc(layer->bias->nbytes);
        float* h_gw = (float*)malloc(layer->grad_w->nbytes);
        float* h_gb = (float*)malloc(layer->grad_b->nbytes);
        cl_tensor_download(layer->weight, h_w);
        cl_tensor_download(layer->bias, h_b);
        cl_tensor_download(layer->grad_w, h_gw);
        cl_tensor_download(layer->grad_b, h_gb);

        size_t w_size = layer->weight->size;
        size_t b_size = layer->bias->size;

        if (opt->velocity_w && opt->velocity_b) {
            float* h_vw = (float*)malloc(layer->weight->nbytes);
            float* h_vb = (float*)malloc(layer->bias->nbytes);
            cl_tensor_download(opt->velocity_w[i], h_vw);
            cl_tensor_download(opt->velocity_b[i], h_vb);

            for (size_t j = 0; j < w_size; j++) {
                h_vw[j] = cfg->momentum * h_vw[j] - cfg->lr * h_gw[j] - cfg->weight_decay * h_w[j];
                h_w[j] += h_vw[j];
            }
            for (size_t j = 0; j < b_size; j++) {
                h_vb[j] = cfg->momentum * h_vb[j] - cfg->lr * h_gb[j];
                h_b[j] += h_vb[j];
            }

            cl_tensor_upload(opt->velocity_w[i], h_vw);
            cl_tensor_upload(opt->velocity_b[i], h_vb);
            free(h_vw);
            free(h_vb);
        } else {
            for (size_t j = 0; j < w_size; j++) {
                h_w[j] -= cfg->lr * h_gw[j] + cfg->weight_decay * h_w[j];
            }
            for (size_t j = 0; j < b_size; j++) {
                h_b[j] -= cfg->lr * h_gb[j];
            }
        }

        cl_tensor_upload(layer->weight, h_w);
        cl_tensor_upload(layer->bias, h_b);

        free(h_w);
        free(h_b);
        free(h_gw);
        free(h_gb);
    }
}

/* --------------------------------------------------------------------------
 * Training Step
 * -------------------------------------------------------------------------- */
static float cl_mlp_train_step(CLOpenCL* cl, CLOpenCLMLP* mlp, CLSGDOptimizer* opt,
                                const CLTensor* input, const CLTensor* targets) {
    CLTensor* output = cl_mlp_forward(cl, mlp, input);

    float* h_output = (float*)malloc(output->nbytes);
    float* h_targets = (float*)malloc(targets->nbytes);
    cl_tensor_download(output, h_output);
    cl_tensor_download(targets, h_targets);

    size_t batch = output->shape[0];
    size_t num_classes = output->shape[1];
    float loss = 0.0f;
    for (size_t b = 0; b < batch; b++) {
        size_t target_class = (size_t)h_targets[b];
        float pred_prob = h_output[b * num_classes + target_class];
        if (pred_prob < 1e-10f) pred_prob = 1e-10f;
        loss += -logf(pred_prob);
    }
    loss /= batch;

    free(h_output);
    free(h_targets);

    CLTensor* grad = cl_cross_entropy_grad(cl, output, targets);
    cl_mlp_backward(cl, mlp, grad);
    cl_mlp_update(cl, mlp, opt);

    cl_tensor_free(output);
    cl_tensor_free(grad);

    return loss;
}

/* --------------------------------------------------------------------------
 * Prediction
 * -------------------------------------------------------------------------- */
static CLTensor* cl_mlp_predict(CLOpenCL* cl, CLOpenCLMLP* mlp, const CLTensor* input) {
    return cl_mlp_forward(cl, mlp, input);
}

/* --------------------------------------------------------------------------
 * Accuracy
 * -------------------------------------------------------------------------- */
static float cl_mlp_accuracy(CLOpenCL* cl, CLOpenCLMLP* mlp,
                              const CLTensor* input, const CLTensor* targets) {
    CLTensor* pred = cl_mlp_predict(cl, mlp, input);

    float* h_pred = (float*)malloc(pred->nbytes);
    float* h_targets = (float*)malloc(targets->nbytes);
    cl_tensor_download(pred, h_pred);
    cl_tensor_download(targets, h_targets);

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
        size_t target_class = (size_t)h_targets[b];
        if (pred_class == target_class) correct++;
    }

    free(h_pred);
    free(h_targets);
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