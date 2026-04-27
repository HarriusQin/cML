/**
 * @file rnn.h
 * @brief Simple Elman RNN (Recurrent Neural Network)
 *
 * Architecture:
 *   h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
 *
 * Input: [batch, seq_len, input_size]
 * Hidden: [batch, seq_len+1, hidden_size] (includes h_{-1} = 0)
 * Output: [batch, seq_len, hidden_size] (final hidden states)
 */

#ifndef __C_RNN_H__
#define __C_RNN_H__

#include "tensor.h"
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * RNN Layer
 * ============================================================================ */

typedef struct {
    Tensor* W_ih;   // [hidden_size, input_size]
    Tensor* W_hh;   // [hidden_size, hidden_size]
    Tensor* b;      // [hidden_size]
    Tensor* grad_W_ih;
    Tensor* grad_W_hh;
    Tensor* grad_b;

    // Cache for backward
    Tensor** h_cache;      // hidden states for each timestep
    Tensor** x_cache;      // inputs for each timestep
    size_t seq_len;
} RNNLayer;

/* ============================================================================
 * RNN Model
 * ============================================================================ */

typedef struct {
    RNNLayer* layer;
    size_t input_size;
    size_t hidden_size;
    size_t num_layers;
    bool training;
} RNN;

/* ============================================================================
 * Layer Creation
 * ============================================================================ */

static RNNLayer* rnn_layer_create(size_t input_size, size_t hidden_size) {
    RNNLayer* layer = (RNNLayer*)malloc(sizeof(RNNLayer));

    // W_ih: [hidden_size, input_size]
    size_t w_ih_shape[] = {hidden_size, input_size};
    layer->W_ih = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, w_ih_shape, 2);
    float std_ih = sqrtf(2.0f / (input_size + hidden_size));
    tensor_fill_randn(layer->W_ih, 0.0f, std_ih);

    // W_hh: [hidden_size, hidden_size]
    size_t w_hh_shape[] = {hidden_size, hidden_size};
    layer->W_hh = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, w_hh_shape, 2);
    float std_hh = sqrtf(1.0f / hidden_size);
    tensor_fill_randn(layer->W_hh, 0.0f, std_hh);

    // bias: [hidden_size]
    size_t b_shape[] = {hidden_size};
    layer->b = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, b_shape, 1);
    tensor_fill_f32(layer->b, 0.0f);

    // Gradients
    layer->grad_W_ih = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, w_ih_shape, 2);
    tensor_fill_f32(layer->grad_W_ih, 0.0f);
    layer->grad_W_hh = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, w_hh_shape, 2);
    tensor_fill_f32(layer->grad_W_hh, 0.0f);
    layer->grad_b = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, b_shape, 1);
    tensor_fill_f32(layer->grad_b, 0.0f);

    layer->h_cache = NULL;
    layer->x_cache = NULL;
    layer->seq_len = 0;

    return layer;
}

static void rnn_layer_free(RNNLayer* layer) {
    if (!layer) return;
    tensor_free(layer->W_ih);
    tensor_free(layer->W_hh);
    tensor_free(layer->b);
    tensor_free(layer->grad_W_ih);
    tensor_free(layer->grad_W_hh);
    tensor_free(layer->grad_b);

    if (layer->h_cache) {
        for (size_t t = 0; t < layer->seq_len; t++) {
            tensor_free(layer->h_cache[t]);
        }
        free(layer->h_cache);
    }
    if (layer->x_cache) {
        for (size_t t = 0; t < layer->seq_len; t++) {
            tensor_free(layer->x_cache[t]);
        }
        free(layer->x_cache);
    }
    free(layer);
}

/* ============================================================================
 * Model Creation
 * ============================================================================ */

static RNN* rnn_create(size_t input_size, size_t hidden_size, size_t num_layers) {
    (void)num_layers; // For now, single layer
    RNN* model = (RNN*)malloc(sizeof(RNN));
    model->layer = rnn_layer_create(input_size, hidden_size);
    model->input_size = input_size;
    model->hidden_size = hidden_size;
    model->num_layers = 1;
    model->training = true;
    return model;
}

static void rnn_free(RNN* model) {
    if (!model) return;
    rnn_layer_free(model->layer);
    free(model);
}

/* ============================================================================
 * Forward Pass
 * h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
 * ============================================================================ */

static Tensor* rnn_forward(RNN* model, const Tensor* input) {
    // input: [batch, seq_len, input_size]
    // output: [batch, seq_len, hidden_size]
    size_t batch = input->shape[0];
    size_t seq_len = input->shape[1];
    size_t input_size = model->input_size;
    size_t hidden_size = model->hidden_size;

    RNNLayer* layer = model->layer;

    // Allocate cache
    layer->seq_len = seq_len;
    layer->h_cache = (Tensor**)malloc(sizeof(Tensor*) * seq_len);
    layer->x_cache = (Tensor**)malloc(sizeof(Tensor*) * seq_len);

    float* w_ih = (float*)layer->W_ih->data;
    float* w_hh = (float*)layer->W_hh->data;
    float* b = (float*)layer->b->data;

    // Output tensor
    size_t out_shape[] = {batch, seq_len, hidden_size};
    Tensor* output = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, out_shape, 3);
    float* out_data = (float*)output->data;
    float* in_data = (float*)input->data;

    // h_{-1} = zeros
    float* h_prev = (float*)calloc(hidden_size, sizeof(float));

    for (size_t t = 0; t < seq_len; t++) {
        // Get x_t: [batch, input_size]
        float* x_t = &in_data[t * batch * input_size];

        // Cache input
        layer->x_cache[t] = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW,
                                         (size_t[]){batch, input_size}, 2);
        memcpy(layer->x_cache[t]->data, x_t, batch * input_size * sizeof(float));

        // Compute h_t = tanh(x_t @ W_ih.T + h_prev @ W_hh.T + b)
        // x_t @ W_ih.T: [batch, input_size] @ [input_size, hidden_size] = [batch, hidden_size]
        // h_prev @ W_hh.T: [batch, hidden_size] @ [hidden_size, hidden_size] = [batch, hidden_size]

        // Cache h_t
        layer->h_cache[t] = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW,
                                         (size_t[]){batch, hidden_size}, 2);
        float* h_t = (float*)layer->h_cache[t]->data;

        for (size_t b_idx = 0; b_idx < batch; b_idx++) {
            for (size_t h = 0; h < hidden_size; h++) {
                float sum = b[h];

                // x_t @ W_ih.T
                for (size_t i = 0; i < input_size; i++) {
                    sum += x_t[b_idx * input_size + i] * w_ih[h * input_size + i];
                }

                // h_prev @ W_hh.T
                for (size_t hh = 0; hh < hidden_size; hh++) {
                    sum += h_prev[hh] * w_hh[h * hidden_size + hh];
                }

                h_t[b_idx * hidden_size + h] = tanhf(sum);
            }
        }

        // Copy h_t to output
        memcpy(&out_data[t * batch * hidden_size], h_t, batch * hidden_size * sizeof(float));

        // Update h_prev for next timestep
        memcpy(h_prev, h_t, hidden_size * sizeof(float));
    }

    free(h_prev);
    return output;
}

/* ============================================================================
 * Backward Pass (BPTT - Backpropagation Through Time)
 * ============================================================================ */

static void rnn_backward(RNN* model, const Tensor* grad_output) {
    // grad_output: [batch, seq_len, hidden_size]
    size_t batch = grad_output->shape[0];
    size_t seq_len = grad_output->shape[1];
    size_t hidden_size = model->hidden_size;
    size_t input_size = model->input_size;

    RNNLayer* layer = model->layer;
    float* w_ih = (float*)layer->W_ih->data;
    float* w_hh = (float*)layer->W_hh->data;
    float* grad_w_ih = (float*)layer->grad_W_ih->data;
    float* grad_w_hh = (float*)layer->grad_W_hh->data;
    float* grad_b = (float*)layer->grad_b->data;
    float* grad_out = (float*)grad_output->data;

    // Gradient w.r.t. hidden states
    float* dh_next = (float*)calloc(hidden_size, sizeof(float));

    // Backward through time (reverse order)
    for (int t = (int)seq_len - 1; t >= 0; t--) {
        // dh_t = dh_next (from next timestep) + grad_output at t
        float* dh = (float*)malloc(sizeof(float) * hidden_size);
        float* h_t = (float*)layer->h_cache[t]->data;

        for (size_t h = 0; h < hidden_size; h++) {
            dh[h] = dh_next[h];

            // Add gradient from output at this timestep
            // Actually grad_output is already gradient w.r.t. output
            // We need gradient w.r.t. h_t (pre-tanh)
            // dL/dh_t = sum_j (dL/dh_j) * (1 - tanh^2(h_j))
            float grad_from_output = grad_out[t * batch * hidden_size + h];
            float tanh_deriv = 1.0f - h_t[h] * h_t[h];
            dh[h] += grad_from_output * tanh_deriv;
        }

        // Gradient w.r.t. inputs at this timestep
        float* x_t = (float*)layer->x_cache[t]->data;

        // dL/dW_ih = sum_b (dh_t_b).T @ x_t_b
        for (size_t h = 0; h < hidden_size; h++) {
            for (size_t i = 0; i < input_size; i++) {
                float sum = 0.0f;
                for (size_t b_idx = 0; b_idx < batch; b_idx++) {
                    sum += dh[b_idx * hidden_size + h] * x_t[b_idx * input_size + i];
                }
                grad_w_ih[h * input_size + i] += sum / batch;
            }
        }

        // dL/dW_hh = sum_b (dh_t_b).T @ h_{t-1}_b
        float* h_prev = NULL;
        int need_free_h_prev = 0;
        if (t > 0) {
            h_prev = (float*)layer->h_cache[t-1]->data;
        } else {
            h_prev = (float*)calloc(batch * hidden_size, sizeof(float));
            need_free_h_prev = 1;
        }
        for (size_t h = 0; h < hidden_size; h++) {
            for (size_t hh = 0; hh < hidden_size; hh++) {
                float sum = 0.0f;
                for (size_t b_idx = 0; b_idx < batch; b_idx++) {
                    sum += dh[b_idx * hidden_size + h] * h_prev[b_idx * hidden_size + hh];
                }
                grad_w_hh[h * hidden_size + hh] += sum / batch;
            }
        }
        if (need_free_h_prev) free(h_prev);

        // dL/db = sum_b dh_t_b
        for (size_t h = 0; h < hidden_size; h++) {
            float sum = 0.0f;
            for (size_t b_idx = 0; b_idx < batch; b_idx++) {
                sum += dh[b_idx * hidden_size + h];
            }
            grad_b[h] += sum / batch;
        }

        // Gradient flowing back to previous hidden state
        // dh_next = dh_t @ W_hh (for next timestep)
        for (size_t hh = 0; hh < hidden_size; hh++) {
            float sum = 0.0f;
            for (size_t h = 0; h < hidden_size; h++) {
                sum += dh[h] * w_hh[h * hidden_size + hh];
            }
            dh_next[hh] = sum;  // Replace, not accumulate (dh_next is reset each iteration)
        }

        free(dh);
    }

    free(dh_next);
}

/* ============================================================================
 * Training Step
 * ============================================================================ */

static float rnn_train_step(RNN* model, float lr,
                           const Tensor* input, const Tensor* targets) {
    // Forward
    Tensor* output = rnn_forward(model, input);

    // Simple MSE loss (for regression) or cross-entropy for classification
    // Here we use MSE for simplicity
    float* out_data = (float*)output->data;
    float* target_data = (float*)targets->data;
    size_t size = output->size;
    float loss = 0.0f;

    for (size_t i = 0; i < size; i++) {
        float diff = out_data[i] - target_data[i];
        loss += diff * diff;
    }
    loss /= size;

    // Compute gradient of loss w.r.t. output (dL/dout = 2/N * (out - target))
    Tensor* grad = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, output->shape, output->ndim);
    float* grad_data = (float*)grad->data;
    for (size_t i = 0; i < size; i++) {
        grad_data[i] = 2.0f * (out_data[i] - target_data[i]) / size;
    }

    // Backward
    rnn_backward(model, grad);

    // Update weights
    RNNLayer* layer = model->layer;
    float* w_ih = (float*)layer->W_ih->data;
    float* w_hh = (float*)layer->W_hh->data;
    float* b = (float*)layer->b->data;
    float* gw_ih = (float*)layer->grad_W_ih->data;
    float* gw_hh = (float*)layer->grad_W_hh->data;
    float* gb = (float*)layer->grad_b->data;

    size_t w_ih_size = layer->W_ih->size;
    size_t w_hh_size = layer->W_hh->size;

    for (size_t i = 0; i < w_ih_size; i++) w_ih[i] -= lr * gw_ih[i];
    for (size_t i = 0; i < w_hh_size; i++) w_hh[i] -= lr * gw_hh[i];
    for (size_t i = 0; i < layer->b->size; i++) b[i] -= lr * gb[i];

    // Clear gradients
    tensor_fill_f32(layer->grad_W_ih, 0.0f);
    tensor_fill_f32(layer->grad_W_hh, 0.0f);
    tensor_fill_f32(layer->grad_b, 0.0f);

    tensor_free(output);
    tensor_free(grad);
    return loss;
}

/* ============================================================================
 * Prediction (get final hidden state)
 * ============================================================================ */

static Tensor* rnn_predict(RNN* model, const Tensor* input) {
    return rnn_forward(model, input);
}

#endif /* __C_RNN_H__ */
