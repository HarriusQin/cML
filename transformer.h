/**
 * @file transformer.h
 * @brief Transformer (Encoder-Decoder) Neural Network
 *
 * Architecture:
 *   Encoder: x → Embedding → PosEncoding → [EncoderLayer × N] → Output
 *   Decoder: y → Embedding → PosEncoding → [DecoderLayer × N] → FC → Output
 *
 * Key components:
 *   - Multi-Head Attention (MHSA)
 *   - Feed-Forward Network (FFN)
 *   - Layer Normalization
 *
 * Input (encoder): [batch, enc_seq_len, d_model]
 * Input (decoder): [batch, dec_seq_len, d_model]
 * Output: [batch, dec_seq_len, vocab_size] or [batch, seq_len, d_model]
 */

#ifndef __C_TRANSFORMER_H__
#define __C_TRANSFORMER_H__

#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * Transformer Config
 * ============================================================================ */

typedef struct {
    size_t vocab_size;
    size_t d_model;
    size_t n_heads;
    size_t n_encoder_layers;
    size_t n_decoder_layers;
    size_t d_ff;
    size_t max_seq_len;
    float dropout_p;
} TransformerConfig;

/* ============================================================================
 * Multi-Head Self-Attention
 * ============================================================================ */

typedef struct {
    Tensor* W_q;    // [d_model, d_model]
    Tensor* W_k;    // [d_model, d_model]
    Tensor* W_v;    // [d_model, d_model]
    Tensor* W_o;    // [d_model, d_model]
    Tensor* grad_W_q, *grad_W_k, *grad_W_v, *grad_W_o;

    // Cache for backward
    Tensor* q_cache;    // [batch, seq_len, d_model]
    Tensor* k_cache;    // [batch, seq_len, d_model]
    Tensor* v_cache;    // [batch, seq_len, d_model]
    Tensor* attn_weights; // [batch, n_heads, seq_len, seq_len]
    size_t seq_len;
    size_t n_heads;
    size_t d_k;
} MHAttention;

/* ============================================================================
 * Feed-Forward Network
 * ============================================================================ */

typedef struct {
    Tensor* W1;     // [d_ff, d_model]
    Tensor* b1;     // [d_ff]
    Tensor* W2;     // [d_model, d_ff]
    Tensor* b2;     // [d_model]
    Tensor* grad_W1, *grad_b1, *grad_W2, *grad_b2;

    Tensor* input_cache;
} FFN;

/* ============================================================================
 * Layer Normalization (with cached mean/var for backward)
 * ============================================================================ */

typedef struct {
    Tensor* gamma;  // [d_model]
    Tensor* beta;   // [d_model]
    Tensor* grad_gamma, *grad_beta;
    Tensor* mean_cache;
    Tensor* var_cache;
    float eps;
} LayerNorm;

/* ============================================================================
 * Encoder Layer
 * ============================================================================ */

typedef struct {
    MHAttention* mha;
    FFN* ffn;
    LayerNorm* ln1;
    LayerNorm* ln2;
} EncoderLayer;

/* ============================================================================
 * Decoder Layer
 * ============================================================================ */

typedef struct {
    MHAttention* self_attn;
    MHAttention* cross_attn;
    FFN* ffn;
    LayerNorm* ln1;
    LayerNorm* ln2;
    LayerNorm* ln3;
} DecoderLayer;

/* ============================================================================
 * Transformer Model
 * ============================================================================ */

typedef struct {
    // Embeddings
    Tensor* token_embedding;  // [vocab_size, d_model]
    Tensor* pos_embedding;   // [max_seq_len, d_model]

    // Encoder
    EncoderLayer** enc_layers;
    size_t n_encoder_layers;

    // Decoder
    DecoderLayer** dec_layers;
    size_t n_decoder_layers;

    // Output projection
    Tensor* W_out;    // [vocab_size, d_model]
    Tensor* b_out;    // [vocab_size]

    LayerNorm* final_ln;

    // Config
    TransformerConfig config;

    // Cache for generation
    bool training;
} Transformer;

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

static float sqrtf_local(float x) {
    if (x <= 0) return 0;
    float y = x;
    float x1 = 0.5f * (y + x / y);
    for (int i = 0; i < 10; i++) {
        y = x1;
        x1 = 0.5f * (y + x / y);
    }
    return x1;
}

/* ============================================================================
 * MHAttention
 * ============================================================================ */

static MHAttention* mha_create(size_t d_model, size_t n_heads) {
    MHAttention* mha = (MHAttention*)malloc(sizeof(MHAttention));
    size_t d_k = d_model / n_heads;

    size_t w_shape[] = {d_model, d_model};

    mha->W_q = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, w_shape, 2);
    mha->W_k = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, w_shape, 2);
    mha->W_v = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, w_shape, 2);
    mha->W_o = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, w_shape, 2);

    float std = sqrtf(2.0f / d_model);
    tensor_fill_randn(mha->W_q, 0.0f, std);
    tensor_fill_randn(mha->W_k, 0.0f, std);
    tensor_fill_randn(mha->W_v, 0.0f, std);
    tensor_fill_randn(mha->W_o, 0.0f, std);

    mha->grad_W_q = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, w_shape, 2);
    mha->grad_W_k = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, w_shape, 2);
    mha->grad_W_v = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, w_shape, 2);
    mha->grad_W_o = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, w_shape, 2);
    tensor_fill_f32(mha->grad_W_q, 0.0f);
    tensor_fill_f32(mha->grad_W_k, 0.0f);
    tensor_fill_f32(mha->grad_W_v, 0.0f);
    tensor_fill_f32(mha->grad_W_o, 0.0f);

    mha->q_cache = NULL;
    mha->k_cache = NULL;
    mha->v_cache = NULL;
    mha->attn_weights = NULL;
    mha->seq_len = 0;
    mha->n_heads = n_heads;
    mha->d_k = d_k;

    return mha;
}

static void mha_free(MHAttention* mha) {
    if (!mha) return;
    tensor_free(mha->W_q);
    tensor_free(mha->W_k);
    tensor_free(mha->W_v);
    tensor_free(mha->W_o);
    tensor_free(mha->grad_W_q);
    tensor_free(mha->grad_W_k);
    tensor_free(mha->grad_W_v);
    tensor_free(mha->grad_W_o);
    tensor_free(mha->q_cache);
    tensor_free(mha->k_cache);
    tensor_free(mha->v_cache);
    tensor_free(mha->attn_weights);
    free(mha);
}

/* ============================================================================
 * FFN
 * ============================================================================ */

static FFN* ffn_create(size_t d_model, size_t d_ff) {
    FFN* ffn = (FFN*)malloc(sizeof(FFN));

    size_t w1_shape[] = {d_ff, d_model};
    size_t w2_shape[] = {d_model, d_ff};
    size_t b1_shape[] = {d_ff};
    size_t b2_shape[] = {d_model};

    ffn->W1 = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, w1_shape, 2);
    ffn->b1 = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, b1_shape, 1);
    ffn->W2 = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, w2_shape, 2);
    ffn->b2 = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, b2_shape, 1);

    tensor_fill_randn(ffn->W1, 0.0f, sqrtf(2.0f / d_model));
    tensor_fill_randn(ffn->W2, 0.0f, sqrtf(2.0f / d_ff));
    tensor_fill_f32(ffn->b1, 0.0f);
    tensor_fill_f32(ffn->b2, 0.0f);

    ffn->grad_W1 = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, w1_shape, 2);
    ffn->grad_b1 = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, b1_shape, 1);
    ffn->grad_W2 = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, w2_shape, 2);
    ffn->grad_b2 = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, b2_shape, 1);
    tensor_fill_f32(ffn->grad_W1, 0.0f);
    tensor_fill_f32(ffn->grad_b1, 0.0f);
    tensor_fill_f32(ffn->grad_W2, 0.0f);
    tensor_fill_f32(ffn->grad_b2, 0.0f);

    ffn->input_cache = NULL;
    return ffn;
}

static void ffn_free(FFN* ffn) {
    if (!ffn) return;
    tensor_free(ffn->W1);
    tensor_free(ffn->b1);
    tensor_free(ffn->W2);
    tensor_free(ffn->b2);
    tensor_free(ffn->grad_W1);
    tensor_free(ffn->grad_b1);
    tensor_free(ffn->grad_W2);
    tensor_free(ffn->grad_b2);
    tensor_free(ffn->input_cache);
    free(ffn);
}

/* ============================================================================
 * LayerNorm
 * ============================================================================ */

static LayerNorm* layer_norm_create(size_t d_model, float eps) {
    LayerNorm* ln = (LayerNorm*)malloc(sizeof(LayerNorm));

    size_t shape[] = {d_model};
    ln->gamma = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 1);
    ln->beta = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 1);
    tensor_fill_f32(ln->gamma, 1.0f);
    tensor_fill_f32(ln->beta, 0.0f);

    ln->grad_gamma = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 1);
    ln->grad_beta = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 1);
    tensor_fill_f32(ln->grad_gamma, 0.0f);
    tensor_fill_f32(ln->grad_beta, 0.0f);

    ln->mean_cache = NULL;
    ln->var_cache = NULL;
    ln->eps = eps;
    return ln;
}

static void layer_norm_free(LayerNorm* ln) {
    if (!ln) return;
    tensor_free(ln->gamma);
    tensor_free(ln->beta);
    tensor_free(ln->grad_gamma);
    tensor_free(ln->grad_beta);
    tensor_free(ln->mean_cache);
    tensor_free(ln->var_cache);
    free(ln);
}

/* ============================================================================
 * Forward: Multi-Head Attention
 * ============================================================================ */

static Tensor* mha_forward(MHAttention* mha, const Tensor* x, bool causal) {
    // x: [batch, seq_len, d_model]
    size_t batch = x->shape[0];
    size_t seq_len = x->shape[1];
    size_t d_model = x->shape[2];
    size_t n_heads = mha->n_heads;
    size_t d_k = mha->d_k;

    float* x_data = (float*)x->data;
    float* w_q = (float*)mha->W_q->data;
    float* w_k = (float*)mha->W_k->data;
    float* w_v = (float*)mha->W_v->data;

    // Compute Q, K, V: [batch, seq_len, d_model]
    size_t qkv_shape[] = {batch, seq_len, d_model};
    Tensor* Q = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, qkv_shape, 3);
    Tensor* K = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, qkv_shape, 3);
    Tensor* V = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, qkv_shape, 3);

    float* q_data = (float*)Q->data;
    float* k_data = (float*)K->data;
    float* v_data = (float*)V->data;

    // Q = x @ W_q.T, K = x @ W_k.T, V = x @ W_v.T
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            for (size_t i = 0; i < d_model; i++) {
                float q_sum = 0, k_sum = 0, v_sum = 0;
                for (size_t j = 0; j < d_model; j++) {
                    float x_val = x_data[b * seq_len * d_model + s * d_model + j];
                    q_sum += x_val * w_q[i * d_model + j];
                    k_sum += x_val * w_k[i * d_model + j];
                    v_sum += x_val * w_v[i * d_model + j];
                }
                q_data[b * seq_len * d_model + s * d_model + i] = q_sum;
                k_data[b * seq_len * d_model + s * d_model + i] = k_sum;
                v_data[b * seq_len * d_model + s * d_model + i] = v_sum;
            }
        }
    }

    // Cache for backward
    mha->q_cache = tensor_clone(Q);
    mha->k_cache = tensor_clone(K);
    mha->v_cache = tensor_clone(V);
    mha->seq_len = seq_len;

    // Reshape for multi-head: [batch, n_heads, seq_len, d_k]
    size_t head_shape[] = {batch, n_heads, seq_len, d_k};
    Tensor* Q_reshaped = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, head_shape, 4);
    Tensor* K_reshaped = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, head_shape, 4);
    Tensor* V_reshaped = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, head_shape, 4);

    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < n_heads; h++) {
            for (size_t s = 0; s < seq_len; s++) {
                for (size_t k = 0; k < d_k; k++) {
                    size_t out_idx = b * n_heads * seq_len * d_k + h * seq_len * d_k + s * d_k + k;
                    size_t q_idx = b * seq_len * d_model + s * d_model + h * d_k + k;
                    size_t k_idx = b * seq_len * d_model + s * d_model + h * d_k + k;
                    size_t v_idx = b * seq_len * d_model + s * d_model + h * d_k + k;
                    ((float*)Q_reshaped->data)[out_idx] = q_data[q_idx];
                    ((float*)K_reshaped->data)[out_idx] = k_data[k_idx];
                    ((float*)V_reshaped->data)[out_idx] = v_data[v_idx];
                }
            }
        }
    }

    // Attention: S = Q @ K.T / sqrt(d_k)
    // A = softmax(S) @ V
    size_t attn_shape[] = {batch, n_heads, seq_len, seq_len};
    Tensor* attn_scores = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, attn_shape, 4);
    float scale = 1.0f / sqrtf_local((float)d_k);

    float* q_h = (float*)Q_reshaped->data;
    float* k_h = (float*)K_reshaped->data;
    float* scores = (float*)attn_scores->data;

    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < n_heads; h++) {
            for (size_t i = 0; i < seq_len; i++) {
                for (size_t j = 0; j < seq_len; j++) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < d_k; k++) {
                        size_t q_idx = b * n_heads * seq_len * d_k + h * seq_len * d_k + i * d_k + k;
                        size_t k_idx = b * n_heads * seq_len * d_k + h * seq_len * d_k + j * d_k + k;
                        sum += q_h[q_idx] * k_h[k_idx];
                    }
                    scores[b * n_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j] = sum * scale;
                }
            }
        }
    }

    // Apply causal mask if requested
    if (causal) {
        for (size_t b = 0; b < batch; b++) {
            for (size_t h = 0; h < n_heads; h++) {
                for (size_t i = 0; i < seq_len; i++) {
                    for (size_t j = 0; j < seq_len; j++) {
                        if (j > i) {
                            scores[b * n_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j] = -1e9f;
                        }
                    }
                }
            }
        }
    }

    // Softmax
    tensor_softmax(attn_scores, 3);

    // Cache attention weights
    mha->attn_weights = tensor_clone(attn_scores);

    // A @ V: [batch, n_heads, seq_len, d_k]
    size_t v_reshaped_shape[] = {batch, n_heads, seq_len, d_k};
    Tensor* V_transposed = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, v_reshaped_shape, 4);
    float* v_h = (float*)V_reshaped->data;
    float* v_t = (float*)V_transposed->data;

    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < n_heads; h++) {
            for (size_t s = 0; s < seq_len; s++) {
                for (size_t k = 0; k < d_k; k++) {
                    size_t src_idx = b * n_heads * seq_len * d_k + h * seq_len * d_k + s * d_k + k;
                    size_t dst_idx = b * n_heads * seq_len * d_k + h * seq_len * d_k + k * seq_len + s;
                    v_t[dst_idx] = v_h[src_idx];
                }
            }
        }
    }

    // Multiply attention scores with V
    Tensor* attn_output = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, v_reshaped_shape, 4);
    float* a_out = (float*)attn_output->data;
    float* scores_f = (float*)attn_scores->data;

    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < n_heads; h++) {
            for (size_t i = 0; i < seq_len; i++) {
                for (size_t k = 0; k < d_k; k++) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < seq_len; j++) {
                        size_t s_idx = b * n_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                        size_t v_idx = b * n_heads * seq_len * d_k + h * seq_len * d_k + k * seq_len + j;
                        sum += scores_f[s_idx] * v_t[v_idx];
                    }
                    size_t out_idx = b * n_heads * seq_len * d_k + h * seq_len * d_k + i * d_k + k;
                    a_out[out_idx] = sum;
                }
            }
        }
    }

    // Concatenate heads: [batch, seq_len, d_model]
    Tensor* output = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, qkv_shape, 3);
    float* out_data = (float*)output->data;

    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            for (size_t i = 0; i < d_model; i++) {
                size_t out_idx = b * seq_len * d_model + s * d_model + i;
                size_t h = i / d_k;
                size_t k = i % d_k;
                size_t src_idx = b * n_heads * seq_len * d_k + h * seq_len * d_k + s * d_k + k;
                out_data[out_idx] = a_out[src_idx];
            }
        }
    }

    // Final linear: output = output @ W_o.T
    float* w_o = (float*)mha->W_o->data;
    Tensor* final = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, qkv_shape, 3);
    float* final_data = (float*)final->data;

    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            for (size_t i = 0; i < d_model; i++) {
                float sum = 0.0f;
                for (size_t j = 0; j < d_model; j++) {
                    float x_val = out_data[b * seq_len * d_model + s * d_model + j];
                    sum += x_val * w_o[i * d_model + j];
                }
                final_data[b * seq_len * d_model + s * d_model + i] = sum;
            }
        }
    }

    tensor_free(Q);
    tensor_free(K);
    tensor_free(V);
    tensor_free(Q_reshaped);
    tensor_free(K_reshaped);
    tensor_free(V_reshaped);
    tensor_free(attn_scores);
    tensor_free(V_transposed);
    tensor_free(attn_output);
    tensor_free(output);

    return final;
}

/* ============================================================================
 * Forward: FFN
 * ============================================================================ */

static Tensor* ffn_forward(FFN* ffn, const Tensor* x) {
    // x: [batch, seq_len, d_model]
    size_t batch = x->shape[0];
    size_t seq_len = x->shape[1];
    size_t d_model = x->shape[2];
    size_t d_ff = ffn->W1->shape[0];

    ffn->input_cache = tensor_clone(x);

    size_t shape[] = {batch, seq_len, d_ff};
    Tensor* h = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 3);
    float* h_data = (float*)h->data;
    float* x_data = (float*)x->data;
    float* w1 = (float*)ffn->W1->data;
    float* b1 = (float*)ffn->b1->data;

    // h = ReLU(x @ W1.T + b1)
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            for (size_t i = 0; i < d_ff; i++) {
                float sum = b1[i];
                for (size_t j = 0; j < d_model; j++) {
                    float x_val = x_data[b * seq_len * d_model + s * d_model + j];
                    sum += x_val * w1[i * d_model + j];
                }
                h_data[b * seq_len * d_ff + s * d_ff + i] = (sum > 0) ? sum : 0;
            }
        }
    }

    // output = h @ W2.T + b2
    size_t out_shape[] = {batch, seq_len, d_model};
    Tensor* output = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, out_shape, 3);
    float* out_data = (float*)output->data;
    float* w2 = (float*)ffn->W2->data;
    float* b2 = (float*)ffn->b2->data;

    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            for (size_t i = 0; i < d_model; i++) {
                float sum = b2[i];
                for (size_t j = 0; j < d_ff; j++) {
                    float h_val = h_data[b * seq_len * d_ff + s * d_ff + j];
                    sum += h_val * w2[i * d_ff + j];
                }
                out_data[b * seq_len * d_model + s * d_model + i] = sum;
            }
        }
    }

    tensor_free(h);
    return output;
}

/* ============================================================================
 * Forward: LayerNorm
 * ============================================================================ */

static Tensor* layer_norm_forward(LayerNorm* ln, const Tensor* x) {
    size_t batch = x->shape[0];
    size_t seq_len = x->shape[1];
    size_t d_model = x->shape[2];
    float eps = ln->eps;

    size_t shape[] = {batch, seq_len, d_model};
    Tensor* output = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 3);
    float* out_data = (float*)output->data;
    float* x_data = (float*)x->data;
    float* gamma = (float*)ln->gamma->data;
    float* beta = (float*)ln->beta->data;

    // Cache mean and var for backward
    ln->mean_cache = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, (size_t[]){batch, seq_len}, 2);
    ln->var_cache = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, (size_t[]){batch, seq_len}, 2);
    float* mean_data = (float*)ln->mean_cache->data;
    float* var_data = (float*)ln->var_cache->data;

    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            // Compute mean
            float sum = 0.0f;
            for (size_t i = 0; i < d_model; i++) {
                sum += x_data[b * seq_len * d_model + s * d_model + i];
            }
            float mean = sum / d_model;
            mean_data[b * seq_len + s] = mean;

            // Compute variance
            float var_sum = 0.0f;
            for (size_t i = 0; i < d_model; i++) {
                float diff = x_data[b * seq_len * d_model + s * d_model + i] - mean;
                var_sum += diff * diff;
            }
            float var = var_sum / d_model;
            var_data[b * seq_len + s] = var;

            // Normalize and scale
            float std = sqrtf(var + eps);
            for (size_t i = 0; i < d_model; i++) {
                float norm = (x_data[b * seq_len * d_model + s * d_model + i] - mean) / std;
                out_data[b * seq_len * d_model + s * d_model + i] = gamma[i] * norm + beta[i];
            }
        }
    }

    return output;
}

/* ============================================================================
 * Encoder Layer
 * ============================================================================ */

static EncoderLayer* encoder_layer_create(size_t d_model, size_t n_heads, size_t d_ff) {
    EncoderLayer* layer = (EncoderLayer*)malloc(sizeof(EncoderLayer));
    layer->mha = mha_create(d_model, n_heads);
    layer->ffn = ffn_create(d_model, d_ff);
    layer->ln1 = layer_norm_create(d_model, 1e-5f);
    layer->ln2 = layer_norm_create(d_model, 1e-5f);
    return layer;
}

static void encoder_layer_free(EncoderLayer* layer) {
    if (!layer) return;
    mha_free(layer->mha);
    ffn_free(layer->ffn);
    layer_norm_free(layer->ln1);
    layer_norm_free(layer->ln2);
    free(layer);
}

static Tensor* encoder_layer_forward(EncoderLayer* layer, const Tensor* x) {
    // Self-attention + residual
    Tensor* attn_out = mha_forward(layer->mha, x, false);
    float* attn_data = (float*)attn_out->data;
    float* x_data = (float*)x->data;
    size_t size = x->size;
    for (size_t i = 0; i < size; i++) attn_data[i] += x_data[i];

    // Layer norm 1
    Tensor* ln1_out = layer_norm_forward(layer->ln1, attn_out);
    tensor_free(attn_out);

    // FFN + residual
    Tensor* ffn_out = ffn_forward(layer->ffn, ln1_out);
    float* ffn_data = (float*)ffn_out->data;
    float* ln1_data = (float*)ln1_out->data;
    for (size_t i = 0; i < size; i++) ffn_data[i] += ln1_data[i];
    tensor_free(ln1_out);

    // Layer norm 2
    Tensor* ln2_out = layer_norm_forward(layer->ln2, ffn_out);
    tensor_free(ffn_out);

    return ln2_out;
}

/* ============================================================================
 * Decoder Layer
 * ============================================================================ */

static DecoderLayer* decoder_layer_create(size_t d_model, size_t n_heads, size_t d_ff) {
    DecoderLayer* layer = (DecoderLayer*)malloc(sizeof(DecoderLayer));
    layer->self_attn = mha_create(d_model, n_heads);
    layer->cross_attn = mha_create(d_model, n_heads);
    layer->ffn = ffn_create(d_model, d_ff);
    layer->ln1 = layer_norm_create(d_model, 1e-5f);
    layer->ln2 = layer_norm_create(d_model, 1e-5f);
    layer->ln3 = layer_norm_create(d_model, 1e-5f);
    return layer;
}

static void decoder_layer_free(DecoderLayer* layer) {
    if (!layer) return;
    mha_free(layer->self_attn);
    mha_free(layer->cross_attn);
    ffn_free(layer->ffn);
    layer_norm_free(layer->ln1);
    layer_norm_free(layer->ln2);
    layer_norm_free(layer->ln3);
    free(layer);
}

static Tensor* decoder_layer_forward(DecoderLayer* layer, const Tensor* x, const Tensor* enc_output) {
    // x: [batch, seq_len, d_model], enc_output: [batch, enc_seq_len, d_model]
    size_t batch = x->shape[0];
    size_t seq_len = x->shape[1];
    size_t d_model = x->shape[2];
    size_t size = x->size;

    // Self-attention with causal mask + residual
    Tensor* self_attn_out = mha_forward(layer->self_attn, x, true);  // causal = true
    float* attn_data = (float*)self_attn_out->data;
    float* x_data = (float*)x->data;
    for (size_t i = 0; i < size; i++) attn_data[i] += x_data[i];

    // Layer norm 1
    Tensor* ln1_out = layer_norm_forward(layer->ln1, self_attn_out);
    tensor_free(self_attn_out);

    // Cross-attention with encoder output + residual
    Tensor* cross_attn_out = mha_forward(layer->cross_attn, ln1_out, false);  // encoder attend to all
    float* cross_data = (float*)cross_attn_out->data;
    float* ln1_data = (float*)ln1_out->data;
    for (size_t i = 0; i < size; i++) cross_data[i] += ln1_data[i];

    // Layer norm 2
    Tensor* ln2_out = layer_norm_forward(layer->ln2, cross_attn_out);
    tensor_free(cross_attn_out);

    // FFN + residual
    Tensor* ffn_out = ffn_forward(layer->ffn, ln2_out);
    float* ffn_data = (float*)ffn_out->data;
    float* ln2_data = (float*)ln2_out->data;
    for (size_t i = 0; i < size; i++) ffn_data[i] += ln2_data[i];

    // Layer norm 3
    Tensor* ln3_out = layer_norm_forward(layer->ln3, ffn_out);
    tensor_free(ffn_out);

    return ln3_out;
}

/* ============================================================================
 * Transformer Creation
 * ============================================================================ */

static Transformer* transformer_create(TransformerConfig config) {
    Transformer* tr = (Transformer*)malloc(sizeof(Transformer));
    tr->config = config;

    // Embeddings
    size_t emb_shape[] = {config.vocab_size, config.d_model};
    tr->token_embedding = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, emb_shape, 2);
    tensor_fill_randn(tr->token_embedding, 0.0f, sqrtf(1.0f / config.d_model));

    size_t pos_shape[] = {config.max_seq_len, config.d_model};
    tr->pos_embedding = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, pos_shape, 2);
    tensor_fill_randn(tr->pos_embedding, 0.0f, sqrtf(1.0f / config.d_model));

// Encoder layers
    tr->n_encoder_layers = config.n_encoder_layers;
    tr->enc_layers = (EncoderLayer**)malloc(sizeof(EncoderLayer*) * config.n_encoder_layers);
    for (size_t i = 0; i < config.n_encoder_layers; i++) {
        tr->enc_layers[i] = encoder_layer_create(config.d_model, config.n_heads, config.d_ff);
    }

// Decoder layers
    tr->n_decoder_layers = config.n_decoder_layers;
    tr->dec_layers = (DecoderLayer**)malloc(sizeof(DecoderLayer*) * config.n_decoder_layers);
    for (size_t i = 0; i < config.n_decoder_layers; i++) {
        tr->dec_layers[i] = decoder_layer_create(config.d_model, config.n_heads, config.d_ff);
    }

    // Output projection
    size_t out_shape[] = {config.vocab_size, config.d_model};
    tr->W_out = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, out_shape, 2);
    tensor_fill_randn(tr->W_out, 0.0f, sqrtf(1.0f / config.d_model));
    tr->b_out = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, (size_t[]){config.vocab_size}, 1);
    tensor_fill_f32(tr->b_out, 0.0f);

    tr->final_ln = layer_norm_create(config.d_model, 1e-5f);
    tr->training = true;

    return tr;
}

static void transformer_free(Transformer* tr) {
    if (!tr) return;
    tensor_free(tr->token_embedding);
    tensor_free(tr->pos_embedding);

    for (size_t i = 0; i < tr->n_encoder_layers; i++) {
        encoder_layer_free(tr->enc_layers[i]);
    }
    free(tr->enc_layers);

    for (size_t i = 0; i < tr->n_decoder_layers; i++) {
        decoder_layer_free(tr->dec_layers[i]);
    }
    free(tr->dec_layers);

    tensor_free(tr->W_out);
    tensor_free(tr->b_out);
    layer_norm_free(tr->final_ln);
    free(tr);
}

/* ============================================================================
 * Forward Pass (Causal Language Model)
 * ============================================================================ */

static Tensor* transformer_forward(Transformer* tr, const Tensor* input) {
    // input: [batch, seq_len] (token IDs)
    // Returns: [batch, seq_len, vocab_size] logits
    size_t batch = input->shape[0];
    size_t seq_len = input->shape[1];
    size_t d_model = tr->config.d_model;
    size_t vocab_size = tr->config.vocab_size;

    // Embedding + positional encoding
    size_t shape[] = {batch, seq_len, d_model};
    Tensor* x = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 3);
    float* x_data = (float*)x->data;
    uint32_t* input_data = (uint32_t*)input->data;
    float* emb = (float*)tr->token_embedding->data;
    float* pos = (float*)tr->pos_embedding->data;

    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            size_t token_id = input_data[b * seq_len + s];
            if (token_id >= vocab_size) token_id = 0;

            for (size_t d = 0; d < d_model; d++) {
                float emb_val = emb[token_id * d_model + d];
                float pos_val = pos[s * d_model + d];
                x_data[b * seq_len * d_model + s * d_model + d] = emb_val + pos_val;
            }
        }
    }

    // Decoder layers with causal masking
    for (size_t i = 0; i < tr->n_decoder_layers; i++) {
        Tensor* new_x = decoder_layer_forward(tr->dec_layers[i], x, NULL);
        tensor_free(x);
        x = new_x;
    }

    // If no decoder layers, just use embeddings through final_ln
    if (tr->n_decoder_layers == 0) {
        Tensor* ln_out = layer_norm_forward(tr->final_ln, x);
        tensor_free(x);
        x = ln_out;
    }

    // LM head: project to vocabulary [batch, seq_len, vocab_size]
    size_t logits_shape[] = {batch, seq_len, vocab_size};
    Tensor* logits = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, logits_shape, 3);
    float* logits_data = (float*)logits->data;
    float* w_out = (float*)tr->W_out->data;
    float* b_out = (float*)tr->b_out->data;

    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            for (size_t v = 0; v < vocab_size; v++) {
                float sum = b_out[v];
                for (size_t d = 0; d < d_model; d++) {
                    sum += x_data[b * seq_len * d_model + s * d_model + d] * w_out[v * d_model + d];
                }
                logits_data[b * seq_len * vocab_size + s * vocab_size + v] = sum;
            }
        }
    }

    tensor_free(x);
    return logits;
}

/* ============================================================================
 * Cross-Entropy Loss for Language Modeling (simplified: no decoder)
 * ============================================================================ */

static float transformer_compute_loss_and_grad_simple(Transformer* tr, const Tensor* input, const Tensor* targets,
                                                      float* grad_emb, float* grad_pos,
                                                      float* grad_W_out, float* grad_b_out) {
    // Simplified version: directly maps embeddings to logits, no decoder layers
    // This trains embeddings, positional encodings, and LM head
    size_t batch = input->shape[0];
    size_t seq_len = input->shape[1];
    size_t d_model = tr->config.d_model;
    size_t vocab_size = tr->config.vocab_size;
    size_t max_seq_len = tr->config.max_seq_len;

    uint32_t* input_data = (uint32_t*)input->data;
    float* emb = (float*)tr->token_embedding->data;
    float* pos = (float*)tr->pos_embedding->data;
    float* w_out = (float*)tr->W_out->data;
    float* b_out = (float*)tr->b_out->data;

    float total_loss = 0.0f;
    size_t count = 0;

    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            uint32_t target_token = ((uint32_t*)targets->data)[b * seq_len + s];
            if (target_token >= vocab_size) target_token = 0;
            uint32_t token_id = input_data[b * seq_len + s];
            if (token_id >= vocab_size) token_id = 0;

            // Compute hidden = embedding + positional
            float hidden[256];  // max d_model
            for (size_t d = 0; d < d_model; d++) {
                hidden[d] = emb[token_id * d_model + d] + pos[s * d_model + d];
            }

            // Compute logits
            float softmax[128];
            float max_val = -1e9f;
            for (size_t v = 0; v < vocab_size; v++) {
                float logit = b_out[v];
                for (size_t d = 0; d < d_model; d++) {
                    logit += hidden[d] * w_out[v * d_model + d];
                }
                softmax[v] = logit;
                if (logit > max_val) max_val = logit;
            }

            // Softmax
            float sum_exp = 0.0f;
            for (size_t v = 0; v < vocab_size; v++) {
                softmax[v] = expf(softmax[v] - max_val);
                sum_exp += softmax[v];
            }
            for (size_t v = 0; v < vocab_size; v++) {
                softmax[v] /= sum_exp;
            }

            // Loss
            if (softmax[target_token] > 0.0f) {
                total_loss -= logf(softmax[target_token] + 1e-10f);
            }
            count++;

            // Gradients
            // dL/dlogit_v = softmax[v] - 1(v==target)
            // dL/dhidden[d] = sum_v dL/dlogit_v * w_out[v,d]
            // dL/demb[token,d] = dL/dhidden[d] (ignoring decoder)
            // dL/dpos[s,d] = dL/dhidden[d]
            // dL/dW_out[v,d] = dL/dlogit_v * hidden[d]
            // dL/db_out[v] = dL/dlogit_v

            float d_hidden[256];
            for (size_t d = 0; d < d_model; d++) d_hidden[d] = 0.0f;

            for (size_t v = 0; v < vocab_size; v++) {
                float d_logit = softmax[v];
                if (v == target_token) d_logit -= 1.0f;

                // dL/dW_out and dL/db_out
                for (size_t d = 0; d < d_model; d++) {
                    grad_W_out[v * d_model + d] += d_logit * hidden[d];
                }
                grad_b_out[v] += d_logit;

                // dL/dhidden
                for (size_t d = 0; d < d_model; d++) {
                    d_hidden[d] += d_logit * w_out[v * d_model + d];
                }
            }

            // dL/demb and dL/dpos
            for (size_t d = 0; d < d_model; d++) {
                grad_emb[token_id * d_model + d] += d_hidden[d];
                grad_pos[s * d_model + d] += d_hidden[d];
            }
        }
    }

    return count > 0 ? total_loss / count : 0.0f;
}

static float transformer_compute_loss_and_grad(Transformer* tr, const Tensor* input, const Tensor* targets,
                                              float* grad_W_out, float* grad_b_out) {
    // Original version: full forward through decoder layers
    size_t batch = input->shape[0];
    size_t seq_len = input->shape[1];
    size_t d_model = tr->config.d_model;
    size_t vocab_size = tr->config.vocab_size;

    // Forward pass
    Tensor* x = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, (size_t[]){batch, seq_len, d_model}, 3);
    float* x_data = (float*)x->data;
    uint32_t* input_data = (uint32_t*)input->data;
    float* emb = (float*)tr->token_embedding->data;
    float* pos = (float*)tr->pos_embedding->data;

    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            size_t token_id = input_data[b * seq_len + s];
            if (token_id >= vocab_size) token_id = 0;

            for (size_t d = 0; d < d_model; d++) {
                float emb_val = emb[token_id * d_model + d];
                float pos_val = pos[s * d_model + d];
                x_data[b * seq_len * d_model + s * d_model + d] = emb_val + pos_val;
            }
        }
    }

    // Decoder layers
    for (size_t i = 0; i < tr->n_decoder_layers; i++) {
        Tensor* new_x = decoder_layer_forward(tr->dec_layers[i], x, NULL);
        tensor_free(x);
        x = new_x;
    }

    if (tr->n_decoder_layers == 0) {
        Tensor* ln_out = layer_norm_forward(tr->final_ln, x);
        tensor_free(x);
        x = ln_out;
    }

    // LM head
    float* w_out = (float*)tr->W_out->data;
    float* b_out = (float*)tr->b_out->data;

    float total_loss = 0.0f;
    size_t count = 0;

    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            uint32_t target_token = ((uint32_t*)targets->data)[b * seq_len + s];
            if (target_token >= vocab_size) target_token = 0;

            float* logits_row = &x_data[b * seq_len * d_model + s * d_model];
            float softmax[128];
            float max_val = -1e9f;
            for (size_t v = 0; v < vocab_size; v++) {
                float logit = 0.0f;
                for (size_t d = 0; d < d_model; d++) {
                    logit += logits_row[d] * w_out[v * d_model + d];
                }
                logit += b_out[v];
                softmax[v] = logit;
                if (logit > max_val) max_val = logit;
            }

            float sum_exp = 0.0f;
            for (size_t v = 0; v < vocab_size; v++) {
                softmax[v] = expf(softmax[v] - max_val);
                sum_exp += softmax[v];
            }
            for (size_t v = 0; v < vocab_size; v++) {
                softmax[v] /= sum_exp;
            }

            if (softmax[target_token] > 0.0f) {
                total_loss -= logf(softmax[target_token] + 1e-10f);
            }
            count++;

            for (size_t v = 0; v < vocab_size; v++) {
                float d_logit = softmax[v];
                if (v == target_token) d_logit -= 1.0f;

                for (size_t d = 0; d < d_model; d++) {
                    grad_W_out[v * d_model + d] += d_logit * x_data[b * seq_len * d_model + s * d_model + d];
                }
                grad_b_out[v] += d_logit;
            }
        }
    }

    tensor_free(x);
    return count > 0 ? total_loss / count : 0.0f;
}

#endif /* __C_TRANSFORMER_H__ */
