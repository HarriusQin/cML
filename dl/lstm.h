/**
 * @file lstm.h
 * @brief Long Short-Term Memory (LSTM) Network
 *
 * Architecture:
 *   f_t = sigmoid(W_f @ x_t + W_h @ h_{t-1} + b_f)  // forget gate
 *   i_t = sigmoid(W_i @ x_t + W_h @ h_{t-1} + b_i)  // input gate
 *   c_t = f_t * c_{t-1} + i_t * tanh(W_c @ x_t + W_h @ h_{t-1} + b_c)  // cell
 *   o_t = sigmoid(W_o @ x_t + W_h @ h_{t-1} + b_o)  // output gate
 *   h_t = o_t * tanh(c_t)
 *
 * Input: [batch, seq_len, input_size]
 * Output: [batch, seq_len, hidden_size] (hidden states)
 */

#ifndef __C_LSTM_H__
#define __C_LSTM_H__

#include "tensor.h"
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * LSTM Layer
 * ============================================================================ */

typedef struct {
    Tensor* W_f;    // [hidden_size, input_size] forget gate
    Tensor* W_i;    // [hidden_size, input_size] input gate
    Tensor* W_c;    // [hidden_size, input_size] cell candidate
    Tensor* W_o;    // [hidden_size, input_size] output gate
    Tensor* R_f;    // [hidden_size, hidden_size] recurrent
    Tensor* R_i;
    Tensor* R_c;
    Tensor* R_o;
    Tensor* b_f;
    Tensor* b_i;
    Tensor* b_c;
    Tensor* b_o;

    // Gradients
    Tensor* grad_W_f, *grad_W_i, *grad_W_c, *grad_W_o;
    Tensor* grad_R_f, *grad_R_i, *grad_R_c, *grad_R_o;
    Tensor* grad_b_f, *grad_b_i, *grad_b_c, *grad_b_o;

    // Cache for backward (need all intermediate values)
    Tensor** h_cache;      // hidden states [seq_len]
    Tensor** c_cache;      // cell states [seq_len]
    Tensor** x_cache;      // inputs [seq_len]
    Tensor** f_cache;      // forget gates [seq_len]
    Tensor** i_cache;      // input gates [seq_len]
    Tensor** c_tilde_cache; // cell candidates [seq_len]
    Tensor** o_cache;      // output gates [seq_len]
    size_t seq_len;
} LSTMLayer;

/* ============================================================================
 * LSTM Model
 * ============================================================================ */

typedef struct {
    LSTMLayer* layer;
    size_t input_size;
    size_t hidden_size;
    size_t num_layers;
    bool training;
    float clip_threshold;  // gradient clipping threshold
} LSTM;

/* ============================================================================
 * Layer Creation
 * ============================================================================ */

static LSTMLayer* lstm_layer_create(size_t input_size, size_t hidden_size) {
    LSTMLayer* layer = (LSTMLayer*)malloc(sizeof(LSTMLayer));

    size_t w_shape[] = {hidden_size, input_size};
    size_t r_shape[] = {hidden_size, hidden_size};
    size_t b_shape[] = {hidden_size};

    float std_w = sqrtf(2.0f / (input_size + hidden_size));
    float std_r = sqrtf(1.0f / hidden_size);

    // Forget gate
    layer->W_f = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, w_shape, 2);
    tensor_fill_randn(layer->W_f, 0.0f, std_w);
    layer->R_f = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, r_shape, 2);
    tensor_fill_randn(layer->R_f, 0.0f, std_r);
    layer->b_f = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, b_shape, 1);
    tensor_fill_f32(layer->b_f, 0.0f);

    // Input gate
    layer->W_i = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, w_shape, 2);
    tensor_fill_randn(layer->W_i, 0.0f, std_w);
    layer->R_i = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, r_shape, 2);
    tensor_fill_randn(layer->R_i, 0.0f, std_r);
    layer->b_i = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, b_shape, 1);
    tensor_fill_f32(layer->b_i, 0.0f);

    // Cell candidate
    layer->W_c = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, w_shape, 2);
    tensor_fill_randn(layer->W_c, 0.0f, std_w);
    layer->R_c = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, r_shape, 2);
    tensor_fill_randn(layer->R_c, 0.0f, std_r);
    layer->b_c = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, b_shape, 1);
    tensor_fill_f32(layer->b_c, 0.0f);

    // Output gate
    layer->W_o = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, w_shape, 2);
    tensor_fill_randn(layer->W_o, 0.0f, std_w);
    layer->R_o = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, r_shape, 2);
    tensor_fill_randn(layer->R_o, 0.0f, std_r);
    layer->b_o = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, b_shape, 1);
    tensor_fill_f32(layer->b_o, 0.0f);

    // Gradients (allocate but don't initialize)
    layer->grad_W_f = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, w_shape, 2);
    layer->grad_W_i = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, w_shape, 2);
    layer->grad_W_c = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, w_shape, 2);
    layer->grad_W_o = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, w_shape, 2);
    layer->grad_R_f = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, r_shape, 2);
    layer->grad_R_i = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, r_shape, 2);
    layer->grad_R_c = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, r_shape, 2);
    layer->grad_R_o = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, r_shape, 2);
    layer->grad_b_f = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, b_shape, 1);
    layer->grad_b_i = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, b_shape, 1);
    layer->grad_b_c = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, b_shape, 1);
    layer->grad_b_o = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, b_shape, 1);

    // Caches
    layer->h_cache = NULL;
    layer->c_cache = NULL;
    layer->x_cache = NULL;
    layer->f_cache = NULL;
    layer->i_cache = NULL;
    layer->c_tilde_cache = NULL;
    layer->o_cache = NULL;
    layer->seq_len = 0;

    return layer;
}

static void lstm_layer_free(LSTMLayer* layer) {
    if (!layer) return;

    #define FREE_TENSOR(t) do { if (t) tensor_free(t); } while(0)
    FREE_TENSOR(layer->W_f); FREE_TENSOR(layer->W_i);
    FREE_TENSOR(layer->W_c); FREE_TENSOR(layer->W_o);
    FREE_TENSOR(layer->R_f); FREE_TENSOR(layer->R_i);
    FREE_TENSOR(layer->R_c); FREE_TENSOR(layer->R_o);
    FREE_TENSOR(layer->b_f); FREE_TENSOR(layer->b_i);
    FREE_TENSOR(layer->b_c); FREE_TENSOR(layer->b_o);
    FREE_TENSOR(layer->grad_W_f); FREE_TENSOR(layer->grad_W_i);
    FREE_TENSOR(layer->grad_W_c); FREE_TENSOR(layer->grad_W_o);
    FREE_TENSOR(layer->grad_R_f); FREE_TENSOR(layer->grad_R_i);
    FREE_TENSOR(layer->grad_R_c); FREE_TENSOR(layer->grad_R_o);
    FREE_TENSOR(layer->grad_b_f); FREE_TENSOR(layer->grad_b_i);
    FREE_TENSOR(layer->grad_b_c); FREE_TENSOR(layer->grad_b_o);

    if (layer->h_cache) {
        for (size_t t = 0; t < layer->seq_len; t++) tensor_free(layer->h_cache[t]);
        free(layer->h_cache);
    }
    if (layer->c_cache) {
        for (size_t t = 0; t < layer->seq_len; t++) tensor_free(layer->c_cache[t]);
        free(layer->c_cache);
    }
    if (layer->x_cache) {
        for (size_t t = 0; t < layer->seq_len; t++) tensor_free(layer->x_cache[t]);
        free(layer->x_cache);
    }
    if (layer->f_cache) {
        for (size_t t = 0; t < layer->seq_len; t++) tensor_free(layer->f_cache[t]);
        free(layer->f_cache);
    }
    if (layer->i_cache) {
        for (size_t t = 0; t < layer->seq_len; t++) tensor_free(layer->i_cache[t]);
        free(layer->i_cache);
    }
    if (layer->c_tilde_cache) {
        for (size_t t = 0; t < layer->seq_len; t++) tensor_free(layer->c_tilde_cache[t]);
        free(layer->c_tilde_cache);
    }
    if (layer->o_cache) {
        for (size_t t = 0; t < layer->seq_len; t++) tensor_free(layer->o_cache[t]);
        free(layer->o_cache);
    }
    #undef FREE_TENSOR
    free(layer);
}

/* ============================================================================
 * Model Creation
 * ============================================================================ */

static LSTM* lstm_create(size_t input_size, size_t hidden_size, size_t num_layers) {
    (void)num_layers;
    LSTM* model = (LSTM*)malloc(sizeof(LSTM));
    model->layer = lstm_layer_create(input_size, hidden_size);
    model->input_size = input_size;
    model->hidden_size = hidden_size;
    model->num_layers = 1;
    model->training = true;
    model->clip_threshold = 5.0f;
    return model;
}

static void lstm_free(LSTM* model) {
    if (!model) return;
    lstm_layer_free(model->layer);
    free(model);
}

/* ============================================================================
 * Sigmoid and Tanh (in-place helpers)
 * ============================================================================ */

static float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* ============================================================================
 * Forward Pass
 * ============================================================================ */

static Tensor* lstm_forward(LSTM* model, const Tensor* input) {
    // input: [batch, seq_len, input_size]
    // output: [batch, seq_len, hidden_size]
    size_t batch = input->shape[0];
    size_t seq_len = input->shape[1];
    size_t input_size = model->input_size;
    size_t hidden_size = model->hidden_size;

    LSTMLayer* layer = model->layer;

    // Allocate caches
    layer->seq_len = seq_len;
    layer->h_cache = (Tensor**)malloc(sizeof(Tensor*) * seq_len);
    layer->c_cache = (Tensor**)malloc(sizeof(Tensor*) * seq_len);
    layer->x_cache = (Tensor**)malloc(sizeof(Tensor*) * seq_len);
    layer->f_cache = (Tensor**)malloc(sizeof(Tensor*) * seq_len);
    layer->i_cache = (Tensor**)malloc(sizeof(Tensor*) * seq_len);
    layer->c_tilde_cache = (Tensor**)malloc(sizeof(Tensor*) * seq_len);
    layer->o_cache = (Tensor**)malloc(sizeof(Tensor*) * seq_len);

    // Output
    size_t out_shape[] = {batch, seq_len, hidden_size};
    Tensor* output = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, out_shape, 3);
    float* out_data = (float*)output->data;
    float* in_data = (float*)input->data;

    // Initial hidden and cell states
    float* h_prev = (float*)calloc(batch * hidden_size, sizeof(float));
    float* c_prev = (float*)calloc(batch * hidden_size, sizeof(float));

    float* w_f = (float*)layer->W_f->data;
    float* w_i = (float*)layer->W_i->data;
    float* w_c = (float*)layer->W_c->data;
    float* w_o = (float*)layer->W_o->data;
    float* r_f = (float*)layer->R_f->data;
    float* r_i = (float*)layer->R_i->data;
    float* r_c = (float*)layer->R_c->data;
    float* r_o = (float*)layer->R_o->data;
    float* b_f = (float*)layer->b_f->data;
    float* b_i = (float*)layer->b_i->data;
    float* b_c = (float*)layer->b_c->data;
    float* b_o = (float*)layer->b_o->data;

    for (size_t t = 0; t < seq_len; t++) {
        float* x_t = &in_data[t * batch * input_size];
        float* h_t = &out_data[t * batch * hidden_size];

        // Cache x_t
        layer->x_cache[t] = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW,
                                         (size_t[]){batch, input_size}, 2);
        memcpy(layer->x_cache[t]->data, x_t, batch * input_size * sizeof(float));

        // Temporary buffers
        float* gates = (float*)malloc(sizeof(float) * 4 * hidden_size); // f, i, c_tilde, o
        float* f_t = gates;
        float* i_t = gates + hidden_size;
        float* c_tilde = gates + 2 * hidden_size;
        float* o_t = gates + 3 * hidden_size;

        // Compute gates for each batch element
        for (size_t b_idx = 0; b_idx < batch; b_idx++) {
            float* x_b = &x_t[b_idx * input_size];
            float* h_p = &h_prev[b_idx * hidden_size];

            for (size_t h = 0; h < hidden_size; h++) {
                // f_t = sigmoid(W_f @ x + R_f @ h_prev + b_f)
                float f_sum = b_f[h];
                for (size_t i = 0; i < input_size; i++) f_sum += x_b[i] * w_f[h * input_size + i];
                for (size_t hh = 0; hh < hidden_size; hh++) f_sum += h_p[hh] * r_f[h * hidden_size + hh];
                f_t[h] = sigmoidf(f_sum);

                // i_t = sigmoid(W_i @ x + R_i @ h_prev + b_i)
                float i_sum = b_i[h];
                for (size_t i = 0; i < input_size; i++) i_sum += x_b[i] * w_i[h * input_size + i];
                for (size_t hh = 0; hh < hidden_size; hh++) i_sum += h_p[hh] * r_i[h * hidden_size + hh];
                i_t[h] = sigmoidf(i_sum);

                // c_tilde = tanh(W_c @ x + R_c @ h_prev + b_c)
                float c_sum = b_c[h];
                for (size_t i = 0; i < input_size; i++) c_sum += x_b[i] * w_c[h * input_size + i];
                for (size_t hh = 0; hh < hidden_size; hh++) c_sum += h_p[hh] * r_c[h * hidden_size + hh];
                c_tilde[h] = tanhf(c_sum);

                // o_t = sigmoid(W_o @ x + R_o @ h_prev + b_o)
                float o_sum = b_o[h];
                for (size_t i = 0; i < input_size; i++) o_sum += x_b[i] * w_o[h * input_size + i];
                for (size_t hh = 0; hh < hidden_size; hh++) o_sum += h_p[hh] * r_o[h * hidden_size + hh];
                o_t[h] = sigmoidf(o_sum);
            }
        }

        // Cache gates
        layer->f_cache[t] = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW,
                                         (size_t[]){batch, hidden_size}, 2);
        layer->i_cache[t] = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW,
                                         (size_t[]){batch, hidden_size}, 2);
        layer->c_tilde_cache[t] = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW,
                                               (size_t[]){batch, hidden_size}, 2);
        layer->o_cache[t] = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW,
                                         (size_t[]){batch, hidden_size}, 2);
        memcpy(layer->f_cache[t]->data, f_t, batch * hidden_size * sizeof(float));
        memcpy(layer->i_cache[t]->data, i_t, batch * hidden_size * sizeof(float));
        memcpy(layer->c_tilde_cache[t]->data, c_tilde, batch * hidden_size * sizeof(float));
        memcpy(layer->o_cache[t]->data, o_t, batch * hidden_size * sizeof(float));

        // Compute cell and hidden state
        float* c_t = (float*)malloc(sizeof(float) * batch * hidden_size);
        for (size_t b_idx = 0; b_idx < batch; b_idx++) {
            for (size_t h = 0; h < hidden_size; h++) {
                size_t idx = b_idx * hidden_size + h;
                c_t[idx] = f_t[h] * c_prev[idx] + i_t[h] * c_tilde[h];
                h_t[idx] = o_t[h] * tanhf(c_t[idx]);
            }
        }

        // Cache c_t and h_t
        layer->c_cache[t] = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW,
                                         (size_t[]){batch, hidden_size}, 2);
        layer->h_cache[t] = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW,
                                         (size_t[]){batch, hidden_size}, 2);
        memcpy(layer->c_cache[t]->data, c_t, batch * hidden_size * sizeof(float));
        memcpy(layer->h_cache[t]->data, h_t, batch * hidden_size * sizeof(float));

        // Update h_prev for next timestep
        memcpy(h_prev, h_t, batch * hidden_size * sizeof(float));
        memcpy(c_prev, c_t, batch * hidden_size * sizeof(float));

        free(gates);
        free(c_t);
    }

    free(h_prev);
    free(c_prev);
    return output;
}

/* ============================================================================
 * Backward Pass (BPTT)
 * ============================================================================ */

static void lstm_backward(LSTM* model, const Tensor* grad_output) {
    size_t batch = grad_output->shape[0];
    size_t seq_len = grad_output->shape[1];
    size_t hidden_size = model->hidden_size;
    size_t input_size = model->input_size;

    LSTMLayer* layer = model->layer;

    // Gradient buffers
    float* dh_next = (float*)calloc(batch * hidden_size, sizeof(float));
    float* dc_next = (float*)calloc(batch * hidden_size, sizeof(float));

    // Initialize gradients to zero
    tensor_fill_f32(layer->grad_W_f, 0.0f);
    tensor_fill_f32(layer->grad_W_i, 0.0f);
    tensor_fill_f32(layer->grad_W_c, 0.0f);
    tensor_fill_f32(layer->grad_W_o, 0.0f);
    tensor_fill_f32(layer->grad_R_f, 0.0f);
    tensor_fill_f32(layer->grad_R_i, 0.0f);
    tensor_fill_f32(layer->grad_R_c, 0.0f);
    tensor_fill_f32(layer->grad_R_o, 0.0f);
    tensor_fill_f32(layer->grad_b_f, 0.0f);
    tensor_fill_f32(layer->grad_b_i, 0.0f);
    tensor_fill_f32(layer->grad_b_c, 0.0f);
    tensor_fill_f32(layer->grad_b_o, 0.0f);

    float* grad_out = (float*)grad_output->data;

    for (int t = (int)seq_len - 1; t >= 0; t--) {
        float* x_t = (float*)layer->x_cache[t]->data;
        float* h_t = (float*)layer->h_cache[t]->data;
        float* c_t = (float*)layer->c_cache[t]->data;
        float* c_prev = (t > 0) ? (float*)layer->c_cache[t-1]->data : (float*)calloc(batch * hidden_size, sizeof(float));
        float* f_t = (float*)layer->f_cache[t]->data;
        float* i_t = (float*)layer->i_cache[t]->data;
        float* c_tilde = (float*)layer->c_tilde_cache[t]->data;
        float* o_t = (float*)layer->o_cache[t]->data;

        // dh_t = grad_output + dh_next (from next timestep)
        float* dh = (float*)malloc(sizeof(float) * batch * hidden_size);
        for (size_t i = 0; i < batch * hidden_size; i++) {
            dh[i] = grad_out[t * batch * hidden_size + i] + dh_next[i];
        }

        // dc_t = dh_t * o_t * (1 - tanh^2(c_t)) + dc_next * f_t
        float* dc = (float*)malloc(sizeof(float) * batch * hidden_size);
        for (size_t i = 0; i < batch * hidden_size; i++) {
            float tanh_c = tanhf(c_t[i]);
            float do_t = dh[i] * tanh_c;  // gradient w.r.t. o_t
            float dc_t = dh[i] * o_t[i % hidden_size] * (1.0f - tanh_c * tanh_c);
            dc[i] = dc_t + dc_next[i] * f_t[i % hidden_size];
        }

        // Gradients for output gate parameters
        float* w_o = (float*)layer->W_o->data;
        float* r_o = (float*)layer->R_o->data;
        float* gw_o = (float*)layer->grad_W_o->data;
        float* gr_o = (float*)layer->grad_R_o->data;
        float* gb_o = (float*)layer->grad_b_o->data;

        float* w_f = (float*)layer->W_f->data;
        float* r_f = (float*)layer->R_f->data;
        float* gw_f = (float*)layer->grad_W_f->data;
        float* gr_f = (float*)layer->grad_R_f->data;
        float* gb_f = (float*)layer->grad_b_f->data;

        float* w_i = (float*)layer->W_i->data;
        float* r_i = (float*)layer->R_i->data;
        float* gw_i = (float*)layer->grad_W_i->data;
        float* gr_i = (float*)layer->grad_R_i->data;
        float* gb_i = (float*)layer->grad_b_i->data;

        float* w_c = (float*)layer->W_c->data;
        float* r_c = (float*)layer->R_c->data;
        float* gw_c = (float*)layer->grad_W_c->data;
        float* gr_c = (float*)layer->grad_R_c->data;
        float* gb_c = (float*)layer->grad_b_c->data;

        // Gradient accumulation (simplified - accumulating over batch)
        for (size_t b_idx = 0; b_idx < batch; b_idx++) {
            float* x_b = &x_t[b_idx * input_size];
            float* h_p = (t > 0) ? &((float*)layer->h_cache[t-1]->data)[b_idx * hidden_size] : (float*)calloc(hidden_size, sizeof(float));
            if (t == 0) memset(h_p, 0, hidden_size * sizeof(float));

            for (size_t h = 0; h < hidden_size; h++) {
                size_t idx = b_idx * hidden_size + h;

                // dL/dh_t * tanh(c_t) gives gradient for o_t
                float grad_o = dh[idx] * tanhf(c_t[idx]);

                // Compute dL/dW_o, dL/dR_o, dL/db_o
                for (size_t i = 0; i < input_size; i++) {
                    gw_o[h * input_size + i] += grad_o * x_b[i];
                }
                for (size_t hh = 0; hh < hidden_size; hh++) {
                    gr_o[h * hidden_size + hh] += grad_o * h_p[hh];
                }
                gb_o[h] += grad_o;

                // dL/df_t = dL/dc_t * c_{t-1}
                float grad_f = dc[idx] * c_prev[idx];
                float sig_f = f_t[h];
                float sig_f_deriv = sig_f * (1.0f - sig_f);

                // dL/di_t = dL/dc_t * c_tilde
                float grad_i = dc[idx] * c_tilde[h];
                float sig_i = i_t[h];
                float sig_i_deriv = sig_i * (1.0f - sig_i);

                // dL/dc_tilde = dL/dc_t * i_t
                float grad_c_tilde = dc[idx] * i_t[h];
                float tanh_c_tilde = c_tilde[h];
                float tanh_c_tilde_deriv = 1.0f - tanh_c_tilde * tanh_c_tilde;

                // dL/dW_f, dL/dR_f, dL/db_f
                for (size_t i = 0; i < input_size; i++) {
                    gw_f[h * input_size + i] += grad_f * sig_f_deriv * x_b[i];
                }
                for (size_t hh = 0; hh < hidden_size; hh++) {
                    gr_f[h * hidden_size + hh] += grad_f * sig_f_deriv * h_p[hh];
                }
                gb_f[h] += grad_f * sig_f_deriv;

                // dL/dW_i, dL/dR_i, dL/db_i
                for (size_t i = 0; i < input_size; i++) {
                    gw_i[h * input_size + i] += grad_i * sig_i_deriv * x_b[i];
                }
                for (size_t hh = 0; hh < hidden_size; hh++) {
                    gr_i[h * hidden_size + hh] += grad_i * sig_i_deriv * h_p[hh];
                }
                gb_i[h] += grad_i * sig_i_deriv;

                // dL/dW_c, dL/dR_c, dL/db_c
                for (size_t i = 0; i < input_size; i++) {
                    gw_c[h * input_size + i] += grad_c_tilde * tanh_c_tilde_deriv * x_b[i];
                }
                for (size_t hh = 0; hh < hidden_size; hh++) {
                    gr_c[h * hidden_size + hh] += grad_c_tilde * tanh_c_tilde_deriv * h_p[hh];
                }
                gb_c[h] += grad_c_tilde * tanh_c_tilde_deriv;
            }

            if (t == 0) free(h_p);
        }

        // Compute dh_next and dc_next for previous timestep
        memset(dh_next, 0, batch * hidden_size * sizeof(float));
        memset(dc_next, 0, batch * hidden_size * sizeof(float));

        for (size_t b_idx = 0; b_idx < batch; b_idx++) {
            for (size_t h = 0; h < hidden_size; h++) {
                size_t idx = b_idx * hidden_size + h;
                float grad_f = dc[idx] * c_prev[idx] * f_t[h] * (1.0f - f_t[h]);
                float grad_i = dc[idx] * c_tilde[h] * i_t[h] * (1.0f - i_t[h]);
                float grad_c_tilde = dc[idx] * i_t[h] * (1.0f - c_tilde[h] * c_tilde[h]);
                float grad_o = dh[idx] * tanhf(c_t[idx]) * o_t[h] * (1.0f - o_t[h]);

                // dh_next accumulates from all gates
                for (size_t hh = 0; hh < hidden_size; hh++) {
                    dh_next[b_idx * hidden_size + hh] +=
                        grad_f * r_f[h * hidden_size + hh] +
                        grad_i * r_i[h * hidden_size + hh] +
                        grad_c_tilde * r_c[h * hidden_size + hh] +
                        grad_o * r_o[h * hidden_size + hh];
                }
                dc_next[idx] = dc[idx] * f_t[h];
            }
        }

        if (t == 0) free(c_prev);
        free(dh);
        free(dc);
    }

    // Gradient clipping
    if (model->clip_threshold > 0) {
        float clip = model->clip_threshold;
        #define CLIP_GRAD(t) do { \
            float* d = (float*)t->data; \
            for (size_t i = 0; i < t->size; i++) { \
                if (d[i] > clip) d[i] = clip; \
                else if (d[i] < -clip) d[i] = -clip; \
            } \
        } while(0)
        CLIP_GRAD(layer->grad_W_f); CLIP_GRAD(layer->grad_W_i);
        CLIP_GRAD(layer->grad_W_c); CLIP_GRAD(layer->grad_W_o);
        CLIP_GRAD(layer->grad_R_f); CLIP_GRAD(layer->grad_R_i);
        CLIP_GRAD(layer->grad_R_c); CLIP_GRAD(layer->grad_R_o);
        #undef CLIP_GRAD
    }

    free(dh_next);
    free(dc_next);
}

/* ============================================================================
 * Training Step
 * ============================================================================ */

static float lstm_train_step(LSTM* model, float lr,
                            const Tensor* input, const Tensor* targets) {
    Tensor* output = lstm_forward(model, input);

    // MSE loss
    float* out_data = (float*)output->data;
    float* target_data = (float*)targets->data;
    size_t size = output->size;
    float loss = 0.0f;

    for (size_t i = 0; i < size; i++) {
        float diff = out_data[i] - target_data[i];
        loss += diff * diff;
    }
    loss /= size;

    // Gradient of MSE
    Tensor* grad = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, output->shape, output->ndim);
    float* grad_data = (float*)grad->data;
    for (size_t i = 0; i < size; i++) {
        grad_data[i] = 2.0f * (out_data[i] - target_data[i]) / size;
    }

    lstm_backward(model, grad);

    // Update weights
    LSTMLayer* layer = model->layer;
    #define UPDATE(param, grad_param) do { \
        float* p = (float*)param->data; \
        float* g = (float*)grad_param->data; \
        for (size_t i = 0; i < param->size; i++) p[i] -= lr * g[i]; \
    } while(0)

    UPDATE(layer->W_f, layer->grad_W_f); UPDATE(layer->W_i, layer->grad_W_i);
    UPDATE(layer->W_c, layer->grad_W_c); UPDATE(layer->W_o, layer->grad_W_o);
    UPDATE(layer->R_f, layer->grad_R_f); UPDATE(layer->R_i, layer->grad_R_i);
    UPDATE(layer->R_c, layer->grad_R_c); UPDATE(layer->R_o, layer->grad_R_o);
    UPDATE(layer->b_f, layer->grad_b_f); UPDATE(layer->b_i, layer->grad_b_i);
    UPDATE(layer->b_c, layer->grad_b_c); UPDATE(layer->b_o, layer->grad_b_o);
    #undef UPDATE

    tensor_free(output);
    tensor_free(grad);
    return loss;
}

static Tensor* lstm_predict(LSTM* model, const Tensor* input) {
    return lstm_forward(model, input);
}

#endif /* __C_LSTM_H__ */
