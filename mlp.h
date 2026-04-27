#ifndef __C_MLP_H__
#define __C_MLP_H__

/* ============================================================================
 * Multi-Layer Perceptron (MLP) Implementation
 * Features: FC layers, ReLU, Softmax, Cross-Entropy Loss, SGD
 * ============================================================================ */

#include "tensor.h"
#include "idx.h"
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * MLP Layer with Cache for Backprop
 * ============================================================================ */

typedef struct {
    Tensor* weight;      // [out_features, in_features]
    Tensor* bias;        // [out_features]
    Tensor* grad_w;      // gradient for weight
    Tensor* grad_b;      // gradient for bias

    // Cache for backward pass
    Tensor* input_cache;      // input to FC
    Tensor* preact_cache;     // pre-activation (x @ W.T + b)
    Tensor* relu_mask;        // mask for ReLU: 1 if x > 0, 0 otherwise
} FCLayer;

/* ============================================================================
 * Optimizer
 * ============================================================================ */

typedef struct {
    float lr;
    float momentum;
    float weight_decay;
} SGDConfig;

typedef struct {
    SGDConfig config;
    Tensor** velocity_w;  // momentum buffers for weights
    Tensor** velocity_b;  // momentum buffers for biases
    size_t num_layers;
} SGDOptimizer;

/* ============================================================================
 * MLP Network
 * ============================================================================ */

typedef struct {
    FCLayer** layers;
    size_t num_layers;
    size_t input_dim;
    size_t output_dim;
    Tensor** activations;  // stored for backward pass
    size_t num_activations;
} MLP;

/* ============================================================================
 * Layer Operations
 * ============================================================================ */

static FCLayer* fc_layer_create(size_t in_features, size_t out_features) {
    FCLayer* layer = (FCLayer*)malloc(sizeof(FCLayer));

    // Initialize weights with Xavier initialization
    size_t w_shape[] = {out_features, in_features};
    layer->weight = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, w_shape, 2);
    float std = sqrtf(2.0f / (in_features + out_features));
    tensor_fill_randn(layer->weight, 0.0f, std);

    // Initialize bias with zeros
    size_t b_shape[] = {out_features};
    layer->bias = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, b_shape, 1);
    tensor_fill_f32(layer->bias, 0.0f);

    // Gradient buffers
    layer->grad_w = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, w_shape, 2);
    tensor_fill_f32(layer->grad_w, 0.0f);
    layer->grad_b = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, b_shape, 1);
    tensor_fill_f32(layer->grad_b, 0.0f);

    // Cache buffers (for backward pass)
    layer->input_cache = NULL;
    layer->preact_cache = NULL;
    layer->relu_mask = NULL;

    return layer;
}

static void fc_layer_free(FCLayer* layer) {
    if (!layer) return;
    tensor_free(layer->weight);
    tensor_free(layer->bias);
    tensor_free(layer->grad_w);
    tensor_free(layer->grad_b);
    tensor_free(layer->input_cache);
    tensor_free(layer->preact_cache);
    tensor_free(layer->relu_mask);
    free(layer);
}

/* --------------------------------------------------------------------------
 * Forward: y = x @ W.T + b
 * -------------------------------------------------------------------------- */
static Tensor* fc_layer_forward(FCLayer* layer, const Tensor* x) {
    // x: [batch, in_features]
    // W: [out_features, in_features]
    // y: [batch, out_features]

    size_t batch = x->shape[0];
    size_t out_features = layer->weight->shape[0];
    size_t in_features = layer->weight->shape[1];

    // Save input for backward
    layer->input_cache = tensor_clone(x);

    // y = x @ W.T
    // Compute directly using gemm: x [batch, in_features] @ W.T [in_features, out_features]
    // Result: [batch, out_features]
    Tensor* preact = tensor_create(TENSOR_DTYPE_F32, x->layout,
                                   (size_t[]){batch, out_features}, 2);

    // Use tensor_gemm: C = A @ B where A=x, B=W.T
    // But W.T element [i,o] = W[o,i] = weight[o*in_features + i]
    // We can compute y[b,o] = sum_i x[b,i] * W[o,i]
    float* w_data = (float*)layer->weight->data;
    float* x_data = (float*)x->data;
    float* preact_data = (float*)preact->data;

    // Clear preact
    for (size_t i = 0; i < preact->size; i++) preact_data[i] = 0.0f;

    // Compute matmul manually to avoid transpose issue
    // y[b,o] = sum_i x[b,i] * W[o,i]
    for (size_t b = 0; b < batch; b++) {
        for (size_t o = 0; o < out_features; o++) {
            float sum = 0.0f;
            for (size_t i = 0; i < in_features; i++) {
                sum += x_data[b * in_features + i] * w_data[o * in_features + i];
            }
            preact_data[b * out_features + o] = sum;
        }
    }

    // Save pre-activation
    layer->preact_cache = tensor_clone(preact);

    // Add bias: y = preact + bias (broadcast)
    for (size_t b = 0; b < batch; b++) {
        for (size_t o = 0; o < out_features; o++) {
            preact_data[b * out_features + o] += ((float*)layer->bias->data)[o];
        }
    }

    return preact;
}

/* --------------------------------------------------------------------------
 * ReLU: y = max(0, x)
 * Saves mask for backward pass
 * -------------------------------------------------------------------------- */
static void relu_forward_inplace(Tensor* x, Tensor* mask) {
    // mask: 1 if x > 0, 0 otherwise (stored as float)
    float* x_data = (float*)x->data;
    float* mask_data = (float*)mask->data;
    for (size_t i = 0; i < x->size; i++) {
        mask_data[i] = (x_data[i] > 0) ? 1.0f : 0.0f;
        if (x_data[i] < 0) x_data[i] = 0;
    }
}

/* --------------------------------------------------------------------------
 * ReLU backward: returns new tensor (does not modify input)
 * -------------------------------------------------------------------------- */
static Tensor* relu_backward(const Tensor* grad_output, const Tensor* mask) {
    Tensor* grad = tensor_create(grad_output->dtype, grad_output->layout,
                                  grad_output->shape, grad_output->ndim);
    float* grad_data = (float*)grad->data;
    float* mask_data = (float*)mask->data;
    float* out_data = (float*)grad_output->data;
    for (size_t i = 0; i < grad_output->size; i++) {
        grad_data[i] = out_data[i] * mask_data[i];
    }
    return grad;
}

/* --------------------------------------------------------------------------
 * Softmax: softmax(x_i) = exp(x_i) / sum(exp(x_j))
 * -------------------------------------------------------------------------- */
static void softmax_forward(Tensor* x, size_t axis) {
    tensor_softmax(x, axis);
}

/* --------------------------------------------------------------------------
 * Cross-Entropy Loss: L = -sum(y_true * log(y_pred))
 * Returns scalar loss
 * -------------------------------------------------------------------------- */
static float cross_entropy_loss(const Tensor* pred, const Tensor* targets) {
    // pred: [batch, num_classes], targets: [batch] (class indices)
    // or targets: [batch, num_classes] (one-hot)

    size_t batch = pred->shape[0];
    size_t num_classes = pred->shape[1];
    float loss = 0.0f;
    float* pred_data = (float*)pred->data;

    for (size_t b = 0; b < batch; b++) {
        // Find the predicted class and target class
        size_t pred_class = 0;
        size_t target_class = 0;

        // Predicted class = argmax
        float pred_max = pred_data[b * num_classes];
        for (size_t c = 1; c < num_classes; c++) {
            if (pred_data[b * num_classes + c] > pred_max) {
                pred_max = pred_data[b * num_classes + c];
                pred_class = c;
            }
        }

        // Target class from one-hot encoding in targets data
        // Assuming targets is passed as class indices encoded in first column
        // Actually targets should be a flat array of class indices
        // But here we interpret targets->data as class indices
        // Let's assume targets is [batch] with class indices
        target_class = (size_t)((float*)targets->data)[b];

        // Compute cross-entropy: -log(pred[target_class])
        // But pred is probabilities from softmax, already normalized
        // So we use: loss += -log(pred[b, target_class])
        float pred_prob = pred_data[b * num_classes + target_class];
        if (pred_prob < 1e-10f) pred_prob = 1e-10f;  // numerical stability
        loss += -logf(pred_prob);
    }

    return loss / batch;
}

/* ============================================================================
 * MLP Network
 * ============================================================================ */

static MLP* mlp_create(size_t input_dim, size_t hidden_dim, size_t output_dim, size_t num_layers) {
    MLP* mlp = (MLP*)malloc(sizeof(MLP));
    mlp->input_dim = input_dim;
    mlp->output_dim = output_dim;
    mlp->num_layers = num_layers;
    mlp->layers = (FCLayer**)malloc(sizeof(FCLayer*) * num_layers);

    size_t prev_dim = input_dim;
    for (size_t i = 0; i < num_layers; i++) {
        size_t curr_dim = (i == num_layers - 1) ? output_dim : hidden_dim;
        mlp->layers[i] = fc_layer_create(prev_dim, curr_dim);
        prev_dim = curr_dim;
    }

    mlp->activations = NULL;
    mlp->num_activations = 0;

    return mlp;
}

static void mlp_free(MLP* mlp) {
    if (!mlp) return;
    for (size_t i = 0; i < mlp->num_layers; i++) {
        fc_layer_free(mlp->layers[i]);
    }
    free(mlp->layers);
    free(mlp->activations);
    free(mlp);
}

/* --------------------------------------------------------------------------
 * Forward Pass
 * Returns final output tensor (caller must free)
 * -------------------------------------------------------------------------- */
static Tensor* mlp_forward(MLP* mlp, const Tensor* input) {
    Tensor* x = tensor_clone(input);

    for (size_t i = 0; i < mlp->num_layers; i++) {
        FCLayer* layer = mlp->layers[i];

        // FC + Bias
        Tensor* preact = fc_layer_forward(layer, x);
        tensor_free(x);
        x = preact;

        // Activation (except last layer)
        if (i < mlp->num_layers - 1) {
            // Allocate relu_mask if not exists
            if (layer->relu_mask == NULL) {
                layer->relu_mask = tensor_create(TENSOR_DTYPE_F32, x->layout, x->shape, x->ndim);
            } else if (layer->relu_mask->size != x->size || layer->relu_mask->ndim != x->ndim) {
                // Resize if needed
                tensor_free(layer->relu_mask);
                layer->relu_mask = tensor_create(TENSOR_DTYPE_F32, x->layout, x->shape, x->ndim);
            }
            // ReLU with mask saving
            relu_forward_inplace(x, layer->relu_mask);
        }
    }

    // Softmax on output
    softmax_forward(x, 1);

    return x;
}

/* --------------------------------------------------------------------------
 * Backward Pass (Gradient Computation)
 * grad_output: gradient from loss function [batch, output_dim]
 * -------------------------------------------------------------------------- */
static void mlp_backward(MLP* mlp, const Tensor* grad_output) {
    // Backpropagate through layers in reverse order
    // We'll compute gradients layer by layer

    // For simplicity, this implements a simplified backprop
    // that assumes we have stored necessary activations

    // Actually, let's do a proper backprop

    // Start with gradient from loss
    Tensor* grad = tensor_clone(grad_output);

    // Backprop through softmax + FC layers
    for (int i = (int)mlp->num_layers - 1; i >= 0; i--) {
        FCLayer* layer = mlp->layers[i];

        if (i < (int)mlp->num_layers - 1) {
            // ReLU backward: multiply by saved mask
            Tensor* new_grad = relu_backward(grad, layer->relu_mask);
            tensor_free(grad);
            grad = new_grad;
        }

        // FC backward: compute gradients
        // dL/dW = (dL/dy).T @ x
        // dL/db = sum(dL/dy, axis=0)
        // dL/dx = dL/dy @ W

        size_t batch = layer->input_cache->shape[0];
        size_t out_features = layer->weight->shape[0];
        size_t in_features = layer->weight->shape[1];

        // dL/dy is in [batch, out_features], stored in grad
        // x is in [batch, in_features], stored in layer->input_cache

        // grad_w = (dL/dy).T @ x
        // Transpose grad: [out_features, batch]
        Tensor* grad_t = tensor_create(TENSOR_DTYPE_F32, grad->layout,
                                       (size_t[]){out_features, batch}, 2);
        float* grad_t_data = (float*)grad_t->data;
        float* grad_data = (float*)grad->data;
        for (size_t b = 0; b < batch; b++) {
            for (size_t o = 0; o < out_features; o++) {
                grad_t_data[o * batch + b] = grad_data[b * out_features + o];
            }
        }

        // grad_w = grad_t @ x: [out_features, batch] @ [batch, in_features] -> [out_features, in_features]
        // Using tensor_gemm for efficiency
        float* grad_w_data = (float*)layer->grad_w->data;
        float* grad_b_data = (float*)layer->grad_b->data;
        float* input_data = (float*)layer->input_cache->data;

        // Clear gradients
        memset(grad_w_data, 0, tensor_nbytes(layer->grad_w));
        memset(grad_b_data, 0, tensor_nbytes(layer->grad_b));

        // grad_w = grad_t @ input: [out_features, batch] @ [batch, in_features]
        tensor_gemm(grad_w_data, grad_t_data, input_data,
                    out_features, in_features, batch,
                    1.0f, 0.0f);

        // grad_b = sum over batch of grad
        for (size_t b = 0; b < batch; b++) {
            for (size_t o = 0; o < out_features; o++) {
                grad_b_data[o] += grad_data[b * out_features + o];
            }
        }
        // Scale by batch size
        float inv_batch = 1.0f / batch;
        for (size_t o = 0; o < out_features; o++) {
            grad_b_data[o] *= inv_batch;
        }

        // Scale grad_w by batch (gemm computes sum over batch)
        for (size_t i = 0; i < layer->grad_w->size; i++) {
            grad_w_data[i] *= inv_batch;
        }

        // Compute gradient to pass to previous layer: dL/dx = dL/dy @ W
        // dx = grad @ W: [batch, out_features] @ [out_features, in_features] -> [batch, in_features]
        Tensor* grad_input = tensor_create(TENSOR_DTYPE_F32, grad->layout,
                                           (size_t[]){batch, in_features}, 2);

        // Use tensor_gemm: grad @ W
        // grad: [batch, out_features], W: [out_features, in_features]
        // Need to transpose W first for gemm
        float* w_t_data = (float*)malloc(sizeof(float) * out_features * in_features);
        float* w_data = (float*)layer->weight->data;
        for (size_t o = 0; o < out_features; o++) {
            for (size_t inn = 0; inn < in_features; inn++) {
                w_t_data[inn * out_features + o] = w_data[o * in_features + inn];
            }
        }

        tensor_gemm((float*)grad_input->data, grad_data, w_t_data,
                    batch, in_features, out_features,
                    1.0f, 0.0f);
        free(w_t_data);

        tensor_free(grad);
        tensor_free(grad_t);
        grad = grad_input;
    }

    tensor_free(grad);
}

/* --------------------------------------------------------------------------
 * Gradient Clipping
 * -------------------------------------------------------------------------- */
static void clip_gradients(Tensor* grad, float max_norm) {
    if (!grad) return;
    float* data = (float*)grad->data;
    float norm_sq = 0.0f;
    for (size_t i = 0; i < grad->size; i++) norm_sq += data[i] * data[i];
    float norm = sqrtf(norm_sq);
    if (norm > max_norm) {
        float scale = max_norm / norm;
        for (size_t i = 0; i < grad->size; i++) data[i] *= scale;
    }
}

/* --------------------------------------------------------------------------
 * Update Parameters using SGD
 * -------------------------------------------------------------------------- */
static void mlp_update(MLP* mlp, SGDOptimizer* opt) {
    for (size_t i = 0; i < mlp->num_layers; i++) {
        FCLayer* layer = mlp->layers[i];
        SGDConfig* cfg = &opt->config;

        // Clip gradients to prevent explosion
        clip_gradients(layer->grad_w, 5.0f);
        clip_gradients(layer->grad_b, 5.0f);

        // Update weights: W = W - lr * grad_w - weight_decay * W
        float* w_data = (float*)layer->weight->data;
        float* gw_data = (float*)layer->grad_w->data;
        float* b_data = (float*)layer->bias->data;
        float* gb_data = (float*)layer->grad_b->data;

        size_t w_size = layer->weight->size;
        size_t b_size = layer->bias->size;

        // Weight update with optional momentum
        if (opt->velocity_w && opt->velocity_b) {
            float* vw = (float*)opt->velocity_w[i]->data;
            float* vb = (float*)opt->velocity_b[i]->data;

            for (size_t j = 0; j < w_size; j++) {
                vw[j] = cfg->momentum * vw[j] - cfg->lr * gw_data[j] - cfg->weight_decay * w_data[j];
                w_data[j] += vw[j];
            }
            for (size_t j = 0; j < b_size; j++) {
                vb[j] = cfg->momentum * vb[j] - cfg->lr * gb_data[j];
                b_data[j] += vb[j];
            }
        } else {
            // Simple SGD update
            for (size_t j = 0; j < w_size; j++) {
                w_data[j] -= cfg->lr * gw_data[j] + cfg->weight_decay * w_data[j];
            }
            for (size_t j = 0; j < b_size; j++) {
                b_data[j] -= cfg->lr * gb_data[j];
            }
        }
    }
}

/* --------------------------------------------------------------------------
 * Cross-Entropy Gradient for Softmax
 * Returns gradient with respect to softmax input
 * -------------------------------------------------------------------------- */
static Tensor* cross_entropy_grad(const Tensor* pred, const Tensor* targets) {
    // pred: [batch, num_classes] (after softmax)
    // targets: [batch] class indices
    // Returns: gradient w.r.t. pre-softmax values

    size_t batch = pred->shape[0];
    size_t num_classes = pred->shape[1];

    Tensor* grad = tensor_create(TENSOR_DTYPE_F32, pred->layout,
                                 (size_t[]){batch, num_classes}, 2);
    float* grad_data = (float*)grad->data;
    float* pred_data = (float*)pred->data;
    float* target_data = (float*)targets->data;

    for (size_t b = 0; b < batch; b++) {
        size_t target_class = (size_t)target_data[b];
        for (size_t c = 0; c < num_classes; c++) {
            // dL/dx_c = pred_c - target_c
            // where target_c is 1 for correct class, 0 otherwise
            grad_data[b * num_classes + c] = pred_data[b * num_classes + c];
            if (c == target_class) {
                grad_data[b * num_classes + c] -= 1.0f;
            }
        }
    }

    return grad;
}

/* --------------------------------------------------------------------------
 * Training Step
 * -------------------------------------------------------------------------- */
static float mlp_train_step(MLP* mlp, SGDOptimizer* opt,
                            const Tensor* input, const Tensor* targets) {
    // Forward pass
    Tensor* output = mlp_forward(mlp, input);

    // Compute loss
    float loss = cross_entropy_loss(output, targets);

    // Compute gradient of loss w.r.t. output (pre-softmax)
    Tensor* grad = cross_entropy_grad(output, targets);

    // Backward pass
    mlp_backward(mlp, grad);

    // Update parameters
    mlp_update(mlp, opt);

    tensor_free(output);
    tensor_free(grad);

    return loss;
}

/* --------------------------------------------------------------------------
 * Prediction (no gradient computation)
 * -------------------------------------------------------------------------- */
static Tensor* mlp_predict(MLP* mlp, const Tensor* input) {
    return mlp_forward(mlp, input);
}

/* --------------------------------------------------------------------------
 * Evaluate Accuracy
 * -------------------------------------------------------------------------- */
static float mlp_accuracy(MLP* mlp, const Tensor* input, const Tensor* targets) {
    Tensor* pred = mlp_predict(mlp, input);

    size_t batch = pred->shape[0];
    size_t num_classes = pred->shape[1];
    float* pred_data = (float*)pred->data;
    float* target_data = (float*)targets->data;

    size_t correct = 0;
    for (size_t b = 0; b < batch; b++) {
        // Find predicted class
        size_t pred_class = 0;
        float pred_max = pred_data[b * num_classes];
        for (size_t c = 1; c < num_classes; c++) {
            if (pred_data[b * num_classes + c] > pred_max) {
                pred_max = pred_data[b * num_classes + c];
                pred_class = c;
            }
        }

        size_t target_class = (size_t)target_data[b];
        if (pred_class == target_class) correct++;
    }

    tensor_free(pred);
    return (float)correct / batch;
}

/* ============================================================================
 * SGD Optimizer
 * ============================================================================ */

static SGDOptimizer* sgd_create(MLP* mlp, float lr, float momentum, float weight_decay) {
    SGDOptimizer* opt = (SGDOptimizer*)malloc(sizeof(SGDOptimizer));
    opt->config.lr = lr;
    opt->config.momentum = momentum;
    opt->config.weight_decay = weight_decay;
    opt->num_layers = mlp->num_layers;

    if (momentum > 0) {
        opt->velocity_w = (Tensor**)malloc(sizeof(Tensor*) * mlp->num_layers);
        opt->velocity_b = (Tensor**)malloc(sizeof(Tensor*) * mlp->num_layers);

        for (size_t i = 0; i < mlp->num_layers; i++) {
            FCLayer* layer = mlp->layers[i];
            opt->velocity_w[i] = tensor_create(TENSOR_DTYPE_F32, layer->weight->layout,
                                                layer->weight->shape, layer->weight->ndim);
            tensor_fill_f32(opt->velocity_w[i], 0.0f);
            opt->velocity_b[i] = tensor_create(TENSOR_DTYPE_F32, layer->bias->layout,
                                                layer->bias->shape, layer->bias->ndim);
            tensor_fill_f32(opt->velocity_b[i], 0.0f);
        }
    } else {
        opt->velocity_w = NULL;
        opt->velocity_b = NULL;
    }

    return opt;
}

static void sgd_free(SGDOptimizer* opt) {
    if (!opt) return;
    if (opt->velocity_w) {
        for (size_t i = 0; i < opt->num_layers; i++) {
            tensor_free(opt->velocity_w[i]);
            tensor_free(opt->velocity_b[i]);
        }
        free(opt->velocity_w);
        free(opt->velocity_b);
    }
    free(opt);
}

/* ============================================================================
 * MNIST Data Loading
 * ============================================================================ */

#define MNIST_NUM_TRAIN 60000
#define MNIST_NUM_TEST 10000
#define MNIST_IMAGE_SIZE 784  // 28x28

typedef struct {
    Tensor* train_images;
    Tensor* train_labels;
    Tensor* test_images;
    Tensor* test_labels;
} MNISTData;

static MNISTData* mnist_load(const char* data_dir) {
    MNISTData* data = (MNISTData*)malloc(sizeof(MNISTData));

    char path[256];

    // Load training images
    snprintf(path, sizeof(path), "%s/train-images-idx3-ubyte", data_dir);
    cIDX* idx_images = idx_load(path);
    if (!idx_images) { printf("Failed to load %s\n", path); free(data); return NULL; }

    // Convert to tensor [60000, 784], normalize to [0, 1]
    size_t train_shape[] = {60000, 784};
    data->train_images = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, train_shape, 2);
    uint8_t* img_data = (uint8_t*)idx_images->idx_data;
    float* tensor_data = (float*)data->train_images->data;
    for (size_t i = 0; i < 60000 * 784; i++) {
        tensor_data[i] = img_data[i] / 255.0f;
    }
    free_idx(idx_images);

    // Load training labels
    snprintf(path, sizeof(path), "%s/train-labels-idx1-ubyte", data_dir);
    cIDX* idx_labels = idx_load(path);
    if (!idx_labels) { printf("Failed to load %s\n", path); free(data); return NULL; }

    size_t label_shape[] = {60000};
    data->train_labels = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, label_shape, 1);
    uint8_t* lbl_data = (uint8_t*)idx_labels->idx_data;
    float* label_tensor = (float*)data->train_labels->data;
    for (size_t i = 0; i < 60000; i++) {
        label_tensor[i] = (float)lbl_data[i];
    }
    free_idx(idx_labels);

    // Load test images
    snprintf(path, sizeof(path), "%s/t10k-images-idx3-ubyte", data_dir);
    idx_images = idx_load(path);
    if (!idx_images) { printf("Failed to load %s\n", path); free(data); return NULL; }

    size_t test_shape[] = {10000, 784};
    data->test_images = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, test_shape, 2);
    img_data = (uint8_t*)idx_images->idx_data;
    tensor_data = (float*)data->test_images->data;
    for (size_t i = 0; i < 10000 * 784; i++) {
        tensor_data[i] = img_data[i] / 255.0f;
    }
    free_idx(idx_images);

    // Load test labels
    snprintf(path, sizeof(path), "%s/t10k-labels-idx1-ubyte", data_dir);
    idx_labels = idx_load(path);
    if (!idx_labels) { printf("Failed to load %s\n", path); free(data); return NULL; }

    size_t test_label_shape[] = {10000};
    data->test_labels = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, test_label_shape, 1);
    lbl_data = (uint8_t*)idx_labels->idx_data;
    label_tensor = (float*)data->test_labels->data;
    for (size_t i = 0; i < 10000; i++) {
        label_tensor[i] = (float)lbl_data[i];
    }
    free_idx(idx_labels);

    return data;
}

static void mnist_free(MNISTData* data) {
    if (!data) return;
    tensor_free(data->train_images);
    tensor_free(data->train_labels);
    tensor_free(data->test_images);
    tensor_free(data->test_labels);
    free(data);
}

/* ============================================================================
 * Mini-Batch Utilities
 * ============================================================================ */

static Tensor* get_batch(const Tensor* images, size_t start, size_t batch_size) {
    size_t shape[] = {batch_size, images->shape[1]};
    Tensor* batch = tensor_create(TENSOR_DTYPE_F32, images->layout, shape, 2);
    memcpy(batch->data,
           (float*)images->data + start * images->shape[1],
           batch_size * images->shape[1] * sizeof(float));
    return batch;
}

static Tensor* get_labels_batch(const Tensor* labels, size_t start, size_t batch_size) {
    size_t shape[] = {batch_size};
    Tensor* batch = tensor_create(TENSOR_DTYPE_F32, labels->layout, shape, 1);
    memcpy(batch->data,
           (float*)labels->data + start,
           batch_size * sizeof(float));
    return batch;
}

/* ============================================================================
 * Machine Learning Framework Adapter
 * ============================================================================ */

#ifdef C_MACHINE_LEARNING_H

/**
 * MLP Model Configuration
 */
typedef struct {
    size_t hidden_dim;      /**< Hidden layer dimension */
    size_t num_layers;      /**< Number of layers (2 = 1 hidden + 1 output) */
    float learning_rate;    /**< Learning rate */
    float momentum;         /**< Momentum term */
    float weight_decay;     /**< Weight decay (L2 regularization) */
    size_t epochs;          /**< Number of training epochs */
    size_t batch_size;      /**< Mini-batch size */
} MLP_Config;

/**
 * MLP Model State (internal)
 */
typedef struct {
    MLP* mlp;
    SGDOptimizer* optimizer;
    size_t input_dim;
    size_t output_dim;
    MLP_Config config;
} MLP_State;

/**
 * @brief Create MLP model for the ML framework
 * @param feature_dim Input feature dimension
 * @param n_classes Number of output classes
 * @param hidden_dim Hidden layer dimension
 * @param num_layers Number of layers
 * @param lr Learning rate
 * @param momentum Momentum
 * @param weight_decay Weight decay
 * @param epochs Number of epochs
 * @param batch_size Batch size
 * @return MLP_State* or NULL on error
 */
static MLP_State* mlp_state_create(size_t feature_dim, size_t n_classes,
                                   size_t hidden_dim, size_t num_layers,
                                   float lr, float momentum, float weight_decay,
                                   size_t epochs, size_t batch_size) {
    MLP_State* state = (MLP_State*)malloc(sizeof(MLP_State));
    if (!state) return NULL;

    state->input_dim = feature_dim;
    state->output_dim = n_classes;
    state->config.hidden_dim = hidden_dim;
    state->config.num_layers = num_layers;
    state->config.learning_rate = lr;
    state->config.momentum = momentum;
    state->config.weight_decay = weight_decay;
    state->config.epochs = epochs;
    state->config.batch_size = batch_size;

    state->mlp = mlp_create(feature_dim, hidden_dim, n_classes, num_layers);
    if (!state->mlp) {
        free(state);
        return NULL;
    }

    state->optimizer = sgd_create(state->mlp, lr, momentum, weight_decay);
    if (!state->optimizer) {
        mlp_free(state->mlp);
        free(state);
        return NULL;
    }

    return state;
}

static void mlp_state_free(MLP_State* state) {
    if (!state) return;
    if (state->optimizer) sgd_free(state->optimizer);
    if (state->mlp) mlp_free(state->mlp);
    free(state);
}

/**
 * Convert dataset features to normalized Tensor
 */
static Tensor* mlp_dataset_to_tensor(const dataset* ds, const size_t* feature_indices,
                                     size_t n_features, const size_t* sample_indices,
                                     size_t n_samples) {
    size_t shape[] = {n_samples, n_features};
    Tensor* t = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 2);
    float* data = (float*)t->data;

    double* min_vals = malloc(sizeof(double) * n_features);
    double* max_vals = malloc(sizeof(double) * n_features);
    if (!min_vals || !max_vals) {
        free(min_vals); free(max_vals); tensor_free(t); return NULL;
    }

    // Find min/max for each feature
    for (size_t f = 0; f < n_features; f++) {
        min_vals[f] = ds->features[feature_indices[f]].data[sample_indices[0]];
        max_vals[f] = ds->features[feature_indices[f]].data[sample_indices[0]];
        for (size_t i = 1; i < n_samples; i++) {
            double val = ds->features[feature_indices[f]].data[sample_indices[i]];
            if (val < min_vals[f]) min_vals[f] = val;
            if (val > max_vals[f]) max_vals[f] = val;
        }
    }

    // Normalize to [0, 1]
    for (size_t i = 0; i < n_samples; i++) {
        for (size_t f = 0; f < n_features; f++) {
            double range = max_vals[f] - min_vals[f];
            double val = ds->features[feature_indices[f]].data[sample_indices[i]];
            data[i * n_features + f] = (range > 0) ?
                (float)((val - min_vals[f]) / range) : 0.0f;
        }
    }

    free(min_vals);
    free(max_vals);
    return t;
}

/**
 * Convert dataset labels to Tensor
 */
static Tensor* mlp_labels_to_tensor(const dataset* ds, size_t target_index,
                                    const size_t* sample_indices, size_t n_samples) {
    size_t shape[] = {n_samples};
    Tensor* t = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 1);
    float* data = (float*)t->data;

    for (size_t i = 0; i < n_samples; i++) {
        data[i] = (float)ds->labels[target_index].labels[sample_indices[i]];
    }

    return t;
}

/**
 * MLP fit function for ML framework
 */
static int mlp_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
                  const dataset* ds, const size_t* feature_indices, size_t n_features,
                  size_t target_index, const size_t* sample_indices,
                  size_t n_samples) {
    (void)config;  // Use default config from state

    if (!ds || !feature_indices || !sample_indices || !state)
        return -1;

    MLP_State* mlp_state = (MLP_State*)state->weights;
    if (!mlp_state || !mlp_state->mlp || !mlp_state->optimizer)
        return -1;

    // Convert to tensors
    Tensor* X = mlp_dataset_to_tensor(ds, feature_indices, n_features, sample_indices, n_samples);
    Tensor* y = mlp_labels_to_tensor(ds, target_index, sample_indices, n_samples);
    if (!X || !y) {
        if (X) tensor_free(X);
        if (y) tensor_free(y);
        return -1;
    }

    size_t batch_size = mlp_state->config.batch_size;
    size_t epochs = mlp_state->config.epochs;
    size_t batches = n_samples / batch_size;
    if (batches == 0) batches = 1;

    // Training loop
    for (size_t epoch = 0; epoch < epochs; epoch++) {
        for (size_t b = 0; b < batches; b++) {
            size_t start = b * batch_size;
            size_t actual_batch = (start + batch_size > n_samples) ?
                                  (n_samples - start) : batch_size;
            if (actual_batch == 0) break;

            Tensor* X_batch = tensor_create(TENSOR_DTYPE_F32, X->layout,
                                            (size_t[]){actual_batch, n_features}, 2);
            Tensor* y_batch = tensor_create(TENSOR_DTYPE_F32, y->layout,
                                            (size_t[]){actual_batch}, 1);
            memcpy(X_batch->data, (float*)X->data + start * n_features,
                   actual_batch * n_features * sizeof(float));
            memcpy(y_batch->data, (float*)y->data + start,
                   actual_batch * sizeof(float));

            mlp_train_step(mlp_state->mlp, mlp_state->optimizer, X_batch, y_batch);

            tensor_free(X_batch);
            tensor_free(y_batch);
        }
    }

    tensor_free(X);
    tensor_free(y);
    return 0;
}

/**
 * MLP predict function for ML framework
 */
static int mlp_model_predict(const ML_Weights_t* state, const dataset* ds,
                            const size_t* feature_indices, size_t n_features,
                            const size_t* sample_indices, size_t n_samples,
                            void* output) {
    if (!state || !ds || !feature_indices || !sample_indices || !output)
        return -1;

    MLP_State* mlp_state = (MLP_State*)state->weights;
    if (!mlp_state || !mlp_state->mlp)
        return -1;

    // Convert to tensor
    Tensor* X = mlp_dataset_to_tensor(ds, feature_indices, n_features, sample_indices, n_samples);
    if (!X) return -1;

    // Predict
    Tensor* pred = mlp_predict(mlp_state->mlp, X);
    tensor_free(X);
    if (!pred) return -1;

    int* preds = (int*)output;
    float* pred_data = (float*)pred->data;
    size_t n_classes = mlp_state->output_dim;

    // Argmax
    for (size_t i = 0; i < n_samples; i++) {
        size_t best_class = 0;
        float best_prob = pred_data[i * n_classes];
        for (size_t c = 1; c < n_classes; c++) {
            if (pred_data[i * n_classes + c] > best_prob) {
                best_prob = pred_data[i * n_classes + c];
                best_class = c;
            }
        }
        preds[i] = (int)best_class;
    }

    tensor_free(pred);
    return 0;
}

/**
 * MLP predict_proba function for ML framework
 */
static int mlp_predict_proba(const ML_Weights_t* state, const dataset* ds,
                              const size_t* feature_indices, size_t n_features,
                              const size_t* sample_indices, size_t n_samples,
                              size_t n_classes, void* output) {
    if (!state || !ds || !feature_indices || !sample_indices || !output)
        return -1;

    MLP_State* mlp_state = (MLP_State*)state->weights;
    if (!mlp_state || !mlp_state->mlp)
        return -1;

    // Convert to tensor
    Tensor* X = mlp_dataset_to_tensor(ds, feature_indices, n_features, sample_indices, n_samples);
    if (!X) return -1;

    // Predict probabilities
    Tensor* pred = mlp_predict(mlp_state->mlp, X);
    tensor_free(X);
    if (!pred) return -1;

    float* proba = (float*)output;
    float* pred_data = (float*)pred->data;
    size_t out_classes = mlp_state->output_dim;

    // Copy probabilities (handle different n_classes)
    for (size_t i = 0; i < n_samples; i++) {
        for (size_t c = 0; c < n_classes; c++) {
            if (c < out_classes) {
                proba[i * n_classes + c] = pred_data[i * out_classes + c];
            } else {
                proba[i * n_classes + c] = 0.0f;
            }
        }
    }

    tensor_free(pred);
    return 0;
}

/**
 * MLP get coefficients function
 */
static int mlp_get_coefficients(const ML_Weights_t* state, void** coeffs, size_t* size) {
    if (!state || !coeffs || !size)
        return -1;

    MLP_State* mlp_state = (MLP_State*)state->weights;
    if (!mlp_state || !mlp_state->mlp)
        return -1;

    // Calculate total size needed
    size_t total_weights = 0;
    for (size_t i = 0; i < mlp_state->mlp->num_layers; i++) {
        FCLayer* layer = mlp_state->mlp->layers[i];
        total_weights += layer->weight->size + layer->bias->size;
    }

    float* data = (float*)malloc(sizeof(float) * total_weights);
    if (!data) return -1;

    // Copy weights
    size_t offset = 0;
    for (size_t i = 0; i < mlp_state->mlp->num_layers; i++) {
        FCLayer* layer = mlp_state->mlp->layers[i];
        size_t w_size = layer->weight->size;
        size_t b_size = layer->bias->size;
        memcpy(data + offset, layer->weight->data, w_size * sizeof(float));
        offset += w_size;
        memcpy(data + offset, layer->bias->data, b_size * sizeof(float));
        offset += b_size;
    }

    *coeffs = data;
    *size = total_weights;
    return 0;
}

/**
 * MLP free state function
 */
static void mlp_free_state(ML_Weights_t* state) {
    if (!state) return;
    if (state->weights) {
        mlp_state_free((MLP_State*)state->weights);
        state->weights = NULL;
    }
    state->size = 0;
}

/**
 * @brief Create an ML_Model_t wrapping MLP
 * @param feature_dim Input feature dimension
 * @param n_classes Number of output classes
 * @param hidden_dim Hidden layer dimension
 * @param num_layers Number of layers
 * @param lr Learning rate
 * @param momentum Momentum
 * @param weight_decay Weight decay
 * @param epochs Number of epochs
 * @param batch_size Batch size
 * @return ML_Model_t or NULL on error
 */
static ML_Model_t* mlp_model_create(size_t feature_dim, size_t n_classes,
                                   size_t hidden_dim, size_t num_layers,
                                   float lr, float momentum, float weight_decay,
                                   size_t epochs, size_t batch_size) {
    ML_Model_t* model = (ML_Model_t*)malloc(sizeof(ML_Model_t));
    if (!model) return NULL;

    model->type = ML_CLASSIFICATION;

    MLP_State* state = mlp_state_create(feature_dim, n_classes, hidden_dim,
                                         num_layers, lr, momentum, weight_decay,
                                         epochs, batch_size);
    if (!state) {
        free(model);
        return NULL;
    }

    model->state.weights = state;
    model->state.size = sizeof(MLP_State);

    model->config.params = &state->config;
    model->config.size = sizeof(MLP_Config);

    // Set method function pointers
    model->methods.fit = mlp_fit;
    model->methods.predict = mlp_model_predict;
    model->methods.predict_proba = mlp_predict_proba;
    model->methods.get_coefficients = mlp_get_coefficients;
    model->methods.serialize = NULL;  // Not implemented
    model->methods.deserialize = NULL;
    model->methods.free_state = mlp_free_state;

    return model;
}

#endif /* C_MACHINE_LEARNING_H */

#endif /* __C_MLP_H__ */
