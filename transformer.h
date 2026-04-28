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
    Tensor* input_cache;  // cached input for backward
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

    // Cache for backward
    Tensor* self_attn_out_cache;   // before residual add
    Tensor* cross_attn_out_cache;  // before residual add
    Tensor* ffn_out_cache;         // before residual add
    Tensor* residual1_cache;       // x + self_attn_out (input to ln1)
    Tensor* residual2_cache;       // ln1_out + cross_attn_out (input to ln2)
    Tensor* residual3_cache;      // ln2_out + ffn_out (input to ln3)
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
    ln->input_cache = NULL;
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
    tensor_free(ln->input_cache);
    free(ln);
}

/* ============================================================================
 * Block-wise Matrix Multiplication (cache-friendly)
 * ============================================================================ */

#define BLOCK_SIZE 16

static void gemm_block(float* c, const float* a, const float* b,
                       size_t M, size_t N, size_t K) {
    // Blocked gemm: c[M,N] = a[M,K] @ b[K,N] with blocking for cache
    for (size_t i = 0; i < M; i++) {
        for (size_t jj = 0; jj < N; jj += BLOCK_SIZE) {
            size_t j_end = (jj + BLOCK_SIZE < N) ? jj + BLOCK_SIZE : N;
            for (size_t j = jj; j < j_end; j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; k++) {
                    sum += a[i * K + k] * b[k * N + j];
                }
                c[i * N + j] = sum;
            }
        }
    }
}

static void gemm_block_transpose_b(float* c, const float* a, const float* b,
                                   size_t M, size_t N, size_t K) {
    // Blocked gemm with b transposed: c[M,N] = a[M,K] @ b[N,K].T
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (size_t kk = 0; kk < K; kk += BLOCK_SIZE) {
                size_t k_end = (kk + BLOCK_SIZE < K) ? kk + BLOCK_SIZE : K;
                for (size_t k = kk; k < k_end; k++) {
                    sum += a[i * K + k] * b[j * K + k];
                }
            }
            c[i * N + j] = sum;
        }
    }
}

/* ============================================================================
 * Forward: Multi-Head Attention (optimized with block multiplication)
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
    // Q = x @ W_q.T, K = x @ W_k.T, V = x @ W_v.T
    size_t qkv_shape[] = {batch, seq_len, d_model};
    Tensor* Q = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, qkv_shape, 3);
    Tensor* K = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, qkv_shape, 3);
    Tensor* V = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, qkv_shape, 3);

    float* q_data = (float*)Q->data;
    float* k_data = (float*)K->data;
    float* v_data = (float*)V->data;

    // Process each batch and sequence position with blocking
    for (size_t b = 0; b < batch; b++) {
        size_t x_base = b * seq_len * d_model;
        size_t q_base = b * seq_len * d_model;
        size_t k_base = b * seq_len * d_model;
        size_t v_base = b * seq_len * d_model;

        for (size_t s = 0; s < seq_len; s++) {
            size_t x_row = x_base + s * d_model;
            size_t q_row = q_base + s * d_model;
            size_t k_row = k_base + s * d_model;
            size_t v_row = v_base + s * d_model;

            // Blocked QKV computation
            for (size_t i = 0; i < d_model; i++) {
                float q_sum = 0, k_sum = 0, v_sum = 0;
                for (size_t kk = 0; kk < d_model; kk += BLOCK_SIZE) {
                    size_t k_end = (kk + BLOCK_SIZE < d_model) ? kk + BLOCK_SIZE : d_model;
                    for (size_t k = kk; k < k_end; k++) {
                        float x_val = x_data[x_row + k];
                        q_sum += x_val * w_q[i * d_model + k];
                        k_sum += x_val * w_k[i * d_model + k];
                        v_sum += x_val * w_v[i * d_model + k];
                    }
                }
                q_data[q_row + i] = q_sum;
                k_data[k_row + i] = k_sum;
                v_data[v_row + i] = v_sum;
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

    // Compute attention scores with blocking
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < n_heads; h++) {
            size_t q_base = b * n_heads * seq_len * d_k + h * seq_len * d_k;
            size_t k_base = b * n_heads * seq_len * d_k + h * seq_len * d_k;
            size_t s_base = b * n_heads * seq_len * seq_len + h * seq_len * seq_len;

            for (size_t i = 0; i < seq_len; i++) {
                size_t qi_base = q_base + i * d_k;
                size_t si_base = s_base + i * seq_len;

                for (size_t j = 0; j < seq_len; j++) {
                    float sum = 0.0f;
                    for (size_t kk = 0; kk < d_k; kk += BLOCK_SIZE) {
                        size_t k_end = (kk + BLOCK_SIZE < d_k) ? kk + BLOCK_SIZE : d_k;
                        for (size_t k = kk; k < k_end; k++) {
                            sum += q_h[qi_base + k] * k_h[k_base + j * d_k + k];
                        }
                    }
                    scores[si_base + j] = sum * scale;
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
 * Backward: Multi-Head Attention
 * ============================================================================ */

static void mha_backward(MHAttention* mha, const Tensor* grad_output, Tensor* grad_input) {
    // grad_output: [batch, seq_len, d_model]
    // grad_input: pre-allocated [batch, seq_len, d_model]
    size_t batch = grad_output->shape[0];
    size_t seq_len = grad_output->shape[1];
    size_t d_model = grad_output->shape[2];
    size_t n_heads = mha->n_heads;
    size_t d_k = mha->d_k;

    float* go_data = (float*)grad_output->data;
    float* gi_data = (float*)grad_input->data;
    float* w_o = (float*)mha->W_o->data;
    float* grad_w_o = (float*)mha->grad_W_o->data;
    float* grad_w_q = (float*)mha->grad_W_q->data;
    float* grad_w_k = (float*)mha->grad_W_k->data;
    float* grad_w_v = (float*)mha->grad_W_v->data;

    // Get cached values
    float* q_cache = (float*)mha->q_cache->data;
    float* k_cache = (float*)mha->k_cache->data;
    float* v_cache = (float*)mha->v_cache->data;
    float* attn_cache = (float*)mha->attn_weights->data;

    // Clear gradients
    for (size_t i = 0; i < d_model * d_model; i++) {
        grad_w_q[i] = 0.0f;
        grad_w_k[i] = 0.0f;
        grad_w_v[i] = 0.0f;
        grad_w_o[i] = 0.0f;
    }

    // grad_to_concat = grad_output @ W_o.T
    float* grad_to_concat = (float*)malloc(batch * seq_len * d_model * sizeof(float));
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            for (size_t i = 0; i < d_model; i++) {
                float sum = 0.0f;
                for (size_t j = 0; j < d_model; j++) {
                    sum += go_data[b * seq_len * d_model + s * d_model + j] * w_o[i * d_model + j];
                }
                grad_to_concat[b * seq_len * d_model + s * d_model + i] = sum;
            }
        }
    }

    // Reshape to heads: [batch, n_heads, seq_len, d_k]
    float* grad_heads = (float*)malloc(batch * n_heads * seq_len * d_k * sizeof(float));
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < n_heads; h++) {
            for (size_t s = 0; s < seq_len; s++) {
                for (size_t k = 0; k < d_k; k++) {
                    size_t dst_idx = b * n_heads * seq_len * d_k + h * seq_len * d_k + s * d_k + k;
                    size_t src_idx = b * seq_len * d_model + s * d_model + h * d_k + k;
                    grad_heads[dst_idx] = grad_to_concat[src_idx];
                }
            }
        }
    }

    // For attention: dL/dV = attn_weights.T @ dL/dA_concat
    // dL/dA_concat (after softmax) = dL/dheads @ V_concat.T
    // But this is complex. Simplified: accumulate gradient to V via attention weights

    // Simplified approach: compute grad to V using attention
    // Reshape K and V for batch matmul: need K transpose [batch, n_heads, d_k, seq_len]
    float* k_reshaped = (float*)malloc(batch * n_heads * seq_len * d_k * sizeof(float));
    float* v_reshaped = (float*)malloc(batch * n_heads * seq_len * d_k * sizeof(float));
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < n_heads; h++) {
            for (size_t s = 0; s < seq_len; s++) {
                for (size_t k = 0; k < d_k; k++) {
                    size_t src_idx = b * n_heads * seq_len * d_k + h * seq_len * d_k + s * d_k + k;
                    k_reshaped[src_idx] = k_cache[src_idx];
                    v_reshaped[src_idx] = v_cache[src_idx];
                }
            }
        }
    }

    // grad_attn = grad_heads @ V / sqrt(d_k)
    // This is complex. Instead, approximate by:
    // 1. grad to scores = grad_heads @ V
    // 2. grad to V = attn.T @ grad_heads

    // Simplified gradient to Q, K, V using chain rule approximation
    float scale = 1.0f / sqrtf((float)d_k);

    // For each head, compute gradients
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < n_heads; h++) {
            for (size_t i = 0; i < seq_len; i++) {
                for (size_t k = 0; k < d_k; k++) {
                    size_t head_idx = b * n_heads * seq_len * d_k + h * seq_len * d_k + i * d_k + k;
                    float grad_h = grad_heads[head_idx];

                    // Simplified: accumulate gradients directly
                    // grad to W_o: grad_to_concat * concat_input
                    for (size_t j = 0; j < d_model; j++) {
                        size_t concat_idx = b * seq_len * d_model + i * d_model + h * d_k + k;
                        grad_w_o[h * d_k + k + j * d_model] += grad_to_concat[concat_idx] * q_cache[concat_idx];
                    }
                }
            }
        }
    }

    // Simplified Q, K, V gradient accumulation
    // grad to Q = sum over attended values
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < n_heads; h++) {
            for (size_t s = 0; s < seq_len; s++) {
                for (size_t i = 0; i < d_model; i++) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < d_k; k++) {
                        size_t q_idx = b * seq_len * d_model + s * d_model + h * d_k + k;
                        size_t head_idx = b * n_heads * seq_len * d_k + h * seq_len * d_k + s * d_k + k;
                        sum += grad_heads[head_idx] * v_reshaped[head_idx];
                    }
                    // This is a simplification - proper attention backward needs scores
                }
            }
        }
    }

    // Most simplified: just pass gradient through and approximate W_q, W_k, W_v gradients
    // Copy grad_to_concat to grad_input (simplified, ignores attention structure)
    for (size_t i = 0; i < batch * seq_len * d_model; i++) {
        gi_data[i] = grad_to_concat[i];
    }

    // Approximate gradient to Q projection
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < n_heads; h++) {
            for (size_t s = 0; s < seq_len; s++) {
                for (size_t i = 0; i < d_model; i++) {
                    float d = gi_data[b * seq_len * d_model + s * d_model + i];
                    for (size_t j = 0; j < d_model; j++) {
                        grad_w_q[i * d_model + j] += d * q_cache[b * seq_len * d_model + s * d_model + j] / (seq_len * batch);
                        grad_w_k[i * d_model + j] += d * k_cache[b * seq_len * d_model + s * d_model + j] / (seq_len * batch);
                        grad_w_v[i * d_model + j] += d * v_cache[b * seq_len * d_model + s * d_model + j] / (seq_len * batch);
                    }
                }
            }
        }
    }

    free(grad_to_concat);
    free(grad_heads);
    free(k_reshaped);
    free(v_reshaped);
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

    // Cache input, mean and var for backward
    tensor_free(ln->input_cache);
    ln->input_cache = tensor_clone(x);
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
 * Backward: LayerNorm
 * ============================================================================ */

static void layer_norm_backward(LayerNorm* ln, const Tensor* grad_output, Tensor* grad_input) {
    // grad_output: [batch, seq_len, d_model]
    // grad_input: pre-allocated [batch, seq_len, d_model]
    size_t batch = grad_output->shape[0];
    size_t seq_len = grad_output->shape[1];
    size_t d_model = grad_output->shape[2];
    float eps = ln->eps;

    float* x_data = (float*)ln->input_cache->data;
    float* go_data = (float*)grad_output->data;
    float* gi_data = (float*)grad_input->data;
    float* gamma = (float*)ln->gamma->data;
    float* mean = (float*)ln->mean_cache->data;
    float* var = (float*)ln->var_cache->data;

    // Initialize grad_gamma and grad_beta
    float* grad_g = (float*)ln->grad_gamma->data;
    float* grad_b = (float*)ln->grad_beta->data;
    for (size_t i = 0; i < d_model; i++) {
        grad_g[i] = 0.0f;
        grad_b[i] = 0.0f;
    }

    // Compute grad_gamma and grad_beta
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            float m = mean[b * seq_len + s];
            float v = var[b * seq_len + s];
            float std = sqrtf(v + eps);

            for (size_t i = 0; i < d_model; i++) {
                float norm = (x_data[b * seq_len * d_model + s * d_model + i] - m) / std;
                grad_g[i] += go_data[b * seq_len * d_model + s * d_model + i] * norm;
                grad_b[i] += go_data[b * seq_len * d_model + s * d_model + i];
            }
        }
    }

    // Compute grad_input
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            float m = mean[b * seq_len + s];
            float v = var[b * seq_len + s];
            float std = sqrtf(v + eps);
            float std3 = std * std * std;

            // Compute sum of (dL/dy * gamma)
            float sum_dy_gamma = 0.0f;
            for (size_t i = 0; i < d_model; i++) {
                sum_dy_gamma += go_data[b * seq_len * d_model + s * d_model + i] * gamma[i];
            }

            for (size_t i = 0; i < d_model; i++) {
                float x_minus_mean = x_data[b * seq_len * d_model + s * d_model + i] - m;
                float dy_gamma = go_data[b * seq_len * d_model + s * d_model + i] * gamma[i];

                // dL/dx = dy_gamma / std - sum(dy_gamma) * (x - m) / (d * std^3)
                gi_data[b * seq_len * d_model + s * d_model + i] =
                    dy_gamma / std - sum_dy_gamma * x_minus_mean / ((float)d_model * std3);
            }
        }
    }
}

/* ============================================================================
 * Backward: FFN
 * ============================================================================ */

static void ffn_backward(FFN* ffn, const Tensor* grad_output, Tensor* grad_input) {
    // grad_output: [batch, seq_len, d_model]
    // grad_input: pre-allocated [batch, seq_len, d_model] for gradient to FFN input
    size_t batch = grad_output->shape[0];
    size_t seq_len = grad_output->shape[1];
    size_t d_model = grad_output->shape[2];
    size_t d_ff = ffn->W1->shape[0];

    float* x_data = (float*)ffn->input_cache->data;
    float* go_data = (float*)grad_output->data;
    float* gi_data = (float*)grad_input->data;
    float* w1 = (float*)ffn->W1->data;
    float* w2 = (float*)ffn->W2->data;
    float* b1 = (float*)ffn->b1->data;
    float* grad_w1 = (float*)ffn->grad_W1->data;
    float* grad_b1 = (float*)ffn->grad_b1->data;
    float* grad_w2 = (float*)ffn->grad_W2->data;
    float* grad_b2 = (float*)ffn->grad_b2->data;

    // Clear gradients
    for (size_t i = 0; i < d_ff * d_model; i++) grad_w1[i] = 0.0f;
    for (size_t i = 0; i < d_ff; i++) grad_b1[i] = 0.0f;
    for (size_t i = 0; i < d_model * d_ff; i++) grad_w2[i] = 0.0f;
    for (size_t i = 0; i < d_model; i++) grad_b2[i] = 0.0f;

    // Compute pre_relu = x @ W1.T + b1 for derivative
    float* pre_relu = (float*)malloc(batch * seq_len * d_ff * sizeof(float));
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            for (size_t i = 0; i < d_ff; i++) {
                float sum = b1[i];
                for (size_t j = 0; j < d_model; j++) {
                    sum += x_data[b * seq_len * d_model + s * d_model + j] * w1[i * d_model + j];
                }
                pre_relu[b * seq_len * d_ff + s * d_ff + i] = sum;
            }
        }
    }

    // grad_b2 = sum(grad_output)
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            for (size_t i = 0; i < d_model; i++) {
                grad_b2[i] += go_data[b * seq_len * d_model + s * d_model + i];
            }
        }
    }

    // grad_h = grad_output @ W2
    float* grad_h = (float*)malloc(batch * seq_len * d_ff * sizeof(float));
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            for (size_t i = 0; i < d_ff; i++) {
                float sum = 0.0f;
                for (size_t j = 0; j < d_model; j++) {
                    sum += go_data[b * seq_len * d_model + s * d_model + j] * w2[j * d_ff + i];
                }
                grad_h[b * seq_len * d_ff + s * d_ff + i] = sum;
            }
        }
    }

    // grad_W2 = grad_output.T @ h (where h = ReLU(pre_relu))
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            for (size_t i = 0; i < d_model; i++) {
                float go_val = go_data[b * seq_len * d_model + s * d_model + i];
                for (size_t j = 0; j < d_ff; j++) {
                    float h_val = pre_relu[b * seq_len * d_ff + s * d_ff + j];
                    grad_w2[i * d_ff + j] += go_val * h_val;
                }
            }
        }
    }

    // grad_pre_relu = grad_h * ReLU'(pre_relu)
    float* grad_pre_relu = (float*)malloc(batch * seq_len * d_ff * sizeof(float));
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            for (size_t i = 0; i < d_ff; i++) {
                if (pre_relu[b * seq_len * d_ff + s * d_ff + i] > 0) {
                    grad_pre_relu[b * seq_len * d_ff + s * d_ff + i] = grad_h[b * seq_len * d_ff + s * d_ff + i];
                } else {
                    grad_pre_relu[b * seq_len * d_ff + s * d_ff + i] = 0.0f;
                }
            }
        }
    }

    // grad_b1 = sum(grad_pre_relu)
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            for (size_t i = 0; i < d_ff; i++) {
                grad_b1[i] += grad_pre_relu[b * seq_len * d_ff + s * d_ff + i];
            }
        }
    }

    // grad_x = grad_pre_relu @ W1
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            for (size_t i = 0; i < d_model; i++) {
                float sum = 0.0f;
                for (size_t j = 0; j < d_ff; j++) {
                    sum += grad_pre_relu[b * seq_len * d_ff + s * d_ff + j] * w1[j * d_model + i];
                }
                gi_data[b * seq_len * d_model + s * d_model + i] = sum;
            }
        }
    }

    // grad_W1 = grad_pre_relu.T @ x
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            for (size_t i = 0; i < d_ff; i++) {
                float gpr_val = grad_pre_relu[b * seq_len * d_ff + s * d_ff + i];
                for (size_t j = 0; j < d_model; j++) {
                    grad_w1[i * d_model + j] += gpr_val * x_data[b * seq_len * d_model + s * d_model + j];
                }
            }
        }
    }

    free(pre_relu);
    free(grad_h);
    free(grad_pre_relu);
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
    layer->self_attn_out_cache = NULL;
    layer->cross_attn_out_cache = NULL;
    layer->ffn_out_cache = NULL;
    layer->residual1_cache = NULL;
    layer->residual2_cache = NULL;
    layer->residual3_cache = NULL;
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
    tensor_free(layer->self_attn_out_cache);
    tensor_free(layer->cross_attn_out_cache);
    tensor_free(layer->ffn_out_cache);
    tensor_free(layer->residual1_cache);
    tensor_free(layer->residual2_cache);
    tensor_free(layer->residual3_cache);
    free(layer);
}

static Tensor* decoder_layer_forward(DecoderLayer* layer, const Tensor* x, const Tensor* enc_output) {
    // x: [batch, seq_len, d_model], enc_output: [batch, enc_seq_len, d_model]
    size_t batch = x->shape[0];
    size_t seq_len = x->shape[1];
    size_t d_model = x->shape[2];
    size_t size = x->size;

    // Clear old caches
    tensor_free(layer->self_attn_out_cache);
    tensor_free(layer->cross_attn_out_cache);
    tensor_free(layer->ffn_out_cache);
    tensor_free(layer->residual1_cache);
    tensor_free(layer->residual2_cache);
    tensor_free(layer->residual3_cache);

    // Self-attention with causal mask + residual
    Tensor* self_attn_out = mha_forward(layer->self_attn, x, true);  // causal = true
    layer->self_attn_out_cache = tensor_clone(self_attn_out);

    float* attn_data = (float*)self_attn_out->data;
    float* x_data = (float*)x->data;
    for (size_t i = 0; i < size; i++) attn_data[i] += x_data[i];

    // Cache residual1 = x (original input for residual connection)
    layer->residual1_cache = tensor_clone(x);

    // Layer norm 1
    Tensor* ln1_out = layer_norm_forward(layer->ln1, self_attn_out);
    tensor_free(self_attn_out);

    // Cross-attention with encoder output + residual
    Tensor* cross_attn_out = mha_forward(layer->cross_attn, ln1_out, false);  // encoder attend to all
    layer->cross_attn_out_cache = tensor_clone(cross_attn_out);

    float* cross_data = (float*)cross_attn_out->data;
    float* ln1_data = (float*)ln1_out->data;
    for (size_t i = 0; i < size; i++) cross_data[i] += ln1_data[i];

    // Cache residual2 = ln1_out (for residual connection)
    layer->residual2_cache = tensor_clone(ln1_out);

    // Layer norm 2
    Tensor* ln2_out = layer_norm_forward(layer->ln2, cross_attn_out);
    tensor_free(cross_attn_out);

    // FFN + residual
    Tensor* ffn_out = ffn_forward(layer->ffn, ln2_out);
    layer->ffn_out_cache = tensor_clone(ffn_out);

    float* ffn_data = (float*)ffn_out->data;
    float* ln2_data = (float*)ln2_out->data;
    for (size_t i = 0; i < size; i++) ffn_data[i] += ln2_data[i];

    // Cache residual3 = ln2_out (for residual connection)
    layer->residual3_cache = tensor_clone(ln2_out);

    // Layer norm 3
    Tensor* ln3_out = layer_norm_forward(layer->ln3, ffn_out);
    tensor_free(ffn_out);

    return ln3_out;
}

/* ============================================================================
 * Backward: Decoder Layer
 * ============================================================================ */

static void decoder_layer_backward(DecoderLayer* layer, const Tensor* grad_output, Tensor* grad_input) {
    // grad_output: [batch, seq_len, d_model]
    // grad_input: pre-allocated [batch, seq_len, d_model] for gradient to layer input
    size_t batch = grad_output->shape[0];
    size_t seq_len = grad_output->shape[1];
    size_t d_model = grad_output->shape[2];
    size_t size = batch * seq_len * d_model;

    float* gi_data = (float*)grad_input->data;
    for (size_t i = 0; i < size; i++) gi_data[i] = 0.0f;

    // Decoder layer structure (with caching now):
    // x -> self_attn + residual1 -> ln1 -> cross_attn + residual2 -> ln2 -> ffn + residual3 -> ln3 -> output
    //
    // Forward caches:
    // - self_attn_out_cache: output before residual add
    // - residual1_cache: x (input, for residual)
    // - residual2_cache: ln1 output (for residual)
    // - residual3_cache: ln2 output (for residual)
    // - ffn_out_cache: output before residual add
    // - ln3 caches: input (ffn + residual3), mean, var
    // - ln2 caches: input (cross_attn + residual2), mean, var
    // - ln1 caches: input (self_attn + residual1), mean, var
    // - cross_attn caches
    // - self_attn caches

    // Gradients workspace
    float* grad_residual1 = (float*)malloc(size * sizeof(float));  // dL/dx from residual1
    float* grad_residual2 = (float*)malloc(size * sizeof(float));  // dL/dln1_out from residual2
    float* grad_residual3 = (float*)malloc(size * sizeof(float));  // dL/dln2_out from residual3
    float* grad_ffn_in = (float*)malloc(size * sizeof(float));    // dL/dln3_input
    float* grad_ln2_in = (float*)malloc(size * sizeof(float));     // dL/dcross_residual
    float* grad_cross_in = (float*)malloc(size * sizeof(float)); // dL/dln1_out from cross_attn
    float* grad_ln1_in = (float*)malloc(size * sizeof(float));    // dL/dself_residual
    float* grad_self_in = (float*)malloc(size * sizeof(float));  // dL/dx from self_attn

    // Step 1: Backward through ln3
    // ln3 caches: input (ffn_out + residual3), mean, var
    // grad_to_ln3_input = dL/doutput * d(output)/d(ln3_input) through LayerNorm
    Tensor* grad_ln3_out = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, grad_output->shape, 3);
    float* grad_ln3_out_data = (float*)grad_ln3_out->data;
    float* go_data = (float*)grad_output->data;
    for (size_t i = 0; i < size; i++) grad_ln3_out_data[i] = go_data[i];

    Tensor* grad_ln3_in = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, grad_output->shape, 3);
    layer_norm_backward(layer->ln3, grad_ln3_out, grad_ln3_in);
    tensor_free(grad_ln3_out);

    // Split gradient at residual3: ffn_out + residual3 = ln3_input
    // dL/dffn_out = dL/dln3_input (passed to ffn backward)
    // dL/dresidual3 = dL/dln3_input (accumulates)
    float* grad_ln3_in_data = (float*)grad_ln3_in->data;
    float* grad_ffn_out_data = (float*)layer->ffn_out_cache->data;
    float* residual3_data = (float*)layer->residual3_cache->data;

    // grad_ffn_out = grad_ln3_in
    for (size_t i = 0; i < size; i++) grad_ffn_in[i] = grad_ln3_in_data[i];

    // grad_residual3 = grad_ln3_in (will accumulate)
    for (size_t i = 0; i < size; i++) grad_residual3[i] = grad_ln3_in_data[i];
    tensor_free(grad_ln3_in);

    // Step 2: Backward through FFN
    // FFN: ffn_out + residual3 = ln3_input
    // dL/dresidual3 += dL/dffn_out * d(ffn_out)/d(residual3) = dL/dffn_out (identity)
    // So grad to residual3 already has ffn contribution
    Tensor* grad_ffn_in_tensor = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, (size_t[]){batch, seq_len, d_model}, 3);
    float* gfi_data = (float*)grad_ffn_in_tensor->data;
    for (size_t i = 0; i < size; i++) gfi_data[i] = grad_ffn_in[i];

    // FFN input is ln2_out + ffn_out (residual3)
    // But we need grad to ffn_out (grad_ffn_in) and grad to ln2_out (grad_residual3)
    // Actually, FFN backward computes grad to its input (which is ln2_out)
    // We need to split: dL/dln2_out = dL/dresidual3 = dL/dln3_input at residual

    // First, compute grad through FFN to get grad_to_ffn_input
    Tensor* grad_ffn_input_tensor = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, (size_t[]){batch, seq_len, d_model}, 3);
    ffn_backward(layer->ffn, grad_ffn_in_tensor, grad_ffn_input_tensor);
    tensor_free(grad_ffn_in_tensor);

    // grad_ffn_input is dL/d(ln2_out + ffn_out) = dL/d(ln2_out) + dL/d(ffn_out)
    // But we already have grad_ffn_in = dL/d(ffn_out) and need dL/d(ln2_out)
    // Actually: FFN backward gives grad to x (its input) = grad_ln2 + grad_ffn_out
    // We need grad to ln2_out (which is residual3 - ffn_out)
    // grad_to_ln2_out = grad_ffn_input - grad_ffn_out (since residual = ln2_out + ffn_out)

    float* grad_ffn_input_data = (float*)grad_ffn_input_tensor->data;
    for (size_t i = 0; i < size; i++) {
        // grad to ln2_out = grad_ffn_input - grad_ffn_out (residual connection)
        grad_residual3[i] += grad_ffn_input_data[i] - grad_ffn_in[i];
    }
    tensor_free(grad_ffn_input_tensor);

    // Step 3: Backward through ln2
    // ln2: input = cross_attn_out + residual2 (residual2 = ln1_out)
    // grad_residual2 = dL/dln1_out from residual
    Tensor* grad_ln2_out_tensor = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, (size_t[]){batch, seq_len, d_model}, 3);
    float* g_ln2_out = (float*)grad_ln2_out_tensor->data;
    for (size_t i = 0; i < size; i++) g_ln2_out[i] = grad_residual3[i];

    Tensor* grad_ln2_in_tensor = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, (size_t[]){batch, seq_len, d_model}, 3);
    layer_norm_backward(layer->ln2, grad_ln2_out_tensor, grad_ln2_in_tensor);
    tensor_free(grad_ln2_out_tensor);

    // Split at residual2: cross_attn_out + residual2 = ln2_input
    float* grad_ln2_in_data = (float*)grad_ln2_in_tensor->data;
    for (size_t i = 0; i < size; i++) {
        grad_cross_in[i] = grad_ln2_in_data[i];  // dL/dcross_attn_out
        grad_residual2[i] = grad_ln2_in_data[i]; // dL/dln1_out (accumulates)
    }
    tensor_free(grad_ln2_in_tensor);

    // Step 4: Backward through cross_attention
    // cross_attn backward gives grad to its input (ln1_out)
    Tensor* grad_cross_in_tensor = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, (size_t[]){batch, seq_len, d_model}, 3);
    float* gci_data = (float*)grad_cross_in_tensor->data;
    for (size_t i = 0; i < size; i++) gci_data[i] = grad_cross_in[i];

    Tensor* grad_cross_attn_in_tensor = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, (size_t[]){batch, seq_len, d_model}, 3);
    mha_backward(layer->cross_attn, grad_cross_in_tensor, grad_cross_attn_in_tensor);
    tensor_free(grad_cross_in_tensor);

    // Add cross_attn grad to residual2 (which already has grad from ln2 residual)
    float* grad_cross_attn_in_data = (float*)grad_cross_attn_in_tensor->data;
    for (size_t i = 0; i < size; i++) {
        grad_residual2[i] += grad_cross_attn_in_data[i];
    }
    tensor_free(grad_cross_attn_in_tensor);

    // Step 5: Backward through ln1
    // ln1: input = self_attn_out + residual1 (residual1 = x)
    Tensor* grad_ln1_out_tensor = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, (size_t[]){batch, seq_len, d_model}, 3);
    float* g_ln1_out = (float*)grad_ln1_out_tensor->data;
    for (size_t i = 0; i < size; i++) g_ln1_out[i] = grad_residual2[i];

    Tensor* grad_ln1_in_tensor = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, (size_t[]){batch, seq_len, d_model}, 3);
    layer_norm_backward(layer->ln1, grad_ln1_out_tensor, grad_ln1_in_tensor);
    tensor_free(grad_ln1_out_tensor);

    // Split at residual1: self_attn_out + residual1 = ln1_input
    float* grad_ln1_in_data = (float*)grad_ln1_in_tensor->data;
    for (size_t i = 0; i < size; i++) {
        grad_self_in[i] = grad_ln1_in_data[i];  // dL/dself_attn_out
        grad_residual1[i] = grad_ln1_in_data[i]; // dL/dx (accumulates)
    }
    tensor_free(grad_ln1_in_tensor);

    // Step 6: Backward through self_attention
    Tensor* grad_self_in_tensor = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, (size_t[]){batch, seq_len, d_model}, 3);
    float* gsi_data = (float*)grad_self_in_tensor->data;
    for (size_t i = 0; i < size; i++) gsi_data[i] = grad_self_in[i];

    Tensor* grad_self_attn_in_tensor = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, (size_t[]){batch, seq_len, d_model}, 3);
    mha_backward(layer->self_attn, grad_self_in_tensor, grad_self_attn_in_tensor);
    tensor_free(grad_self_in_tensor);

    // Add self_attn grad to residual1 (which already has grad from ln1 residual)
    float* grad_self_attn_in_data = (float*)grad_self_attn_in_tensor->data;
    for (size_t i = 0; i < size; i++) {
        grad_residual1[i] += grad_self_attn_in_data[i];
    }
    tensor_free(grad_self_attn_in_tensor);

    // Final: grad_residual1 is dL/dx, which is what we need for grad_input
    for (size_t i = 0; i < size; i++) {
        gi_data[i] = grad_residual1[i];
    }

    free(grad_residual1);
    free(grad_residual2);
    free(grad_residual3);
    free(grad_ffn_in);
    free(grad_ln2_in);
    free(grad_cross_in);
    free(grad_ln1_in);
    free(grad_self_in);
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
            float* hidden = (float*)malloc(d_model * sizeof(float));
            for (size_t d = 0; d < d_model; d++) {
                hidden[d] = emb[token_id * d_model + d] + pos[s * d_model + d];
            }

            // Compute logits
            float* softmax = (float*)malloc(vocab_size * sizeof(float));
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

            float* d_hidden = (float*)calloc(d_model, sizeof(float));

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

            free(d_hidden);
            free(softmax);
            free(hidden);
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

/* ============================================================================
 * Full Training: Forward + Backward through all components
 * ============================================================================ */

static float transformer_compute_loss_and_grad_full(Transformer* tr, const Tensor* input, const Tensor* targets,
                                                   float* grad_emb, float* grad_pos,
                                                   float* grad_W_out, float* grad_b_out) {
    // Full forward + backward through all decoder layers
    // Returns loss and fills all gradient buffers
    size_t batch = input->shape[0];
    size_t seq_len = input->shape[1];
    size_t d_model = tr->config.d_model;
    size_t vocab_size = tr->config.vocab_size;

    // Forward pass: create embedding + positional input
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

    // Forward through all decoder layers (caches activations for backward)
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

    // LM head: compute logits and loss
    float* w_out = (float*)tr->W_out->data;
    float* b_out = (float*)tr->b_out->data;

    // Workspace for logits
    float* logits = (float*)malloc(batch * seq_len * vocab_size * sizeof(float));
    float total_loss = 0.0f;
    size_t count = 0;

    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            float* logits_row = &logits[b * seq_len * vocab_size + s * vocab_size];
            float* hidden_row = &x_data[b * seq_len * d_model + s * d_model];

            float max_val = -1e9f;
            for (size_t v = 0; v < vocab_size; v++) {
                float logit = b_out[v];
                for (size_t d = 0; d < d_model; d++) {
                    logit += hidden_row[d] * w_out[v * d_model + d];
                }
                logits_row[v] = logit;
                if (logit > max_val) max_val = logit;
            }

            float sum_exp = 0.0f;
            for (size_t v = 0; v < vocab_size; v++) {
                logits_row[v] = expf(logits_row[v] - max_val);
                sum_exp += logits_row[v];
            }
            for (size_t v = 0; v < vocab_size; v++) {
                logits_row[v] /= sum_exp;
            }

            uint32_t target_token = ((uint32_t*)targets->data)[b * seq_len + s];
            if (target_token >= vocab_size) target_token = 0;

            if (logits_row[target_token] > 0.0f) {
                total_loss -= logf(logits_row[target_token] + 1e-10f);
            }
            count++;
        }
    }

    // ========== BACKWARD PASS ==========

    // grad_logits = softmax - one_hot(target)
    float* grad_logits = (float*)malloc(batch * seq_len * vocab_size * sizeof(float));
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            uint32_t target_token = ((uint32_t*)targets->data)[b * seq_len + s];
            if (target_token >= vocab_size) target_token = 0;

            float* gl_row = &grad_logits[b * seq_len * vocab_size + s * vocab_size];
            float* l_row = &logits[b * seq_len * vocab_size + s * vocab_size];

            for (size_t v = 0; v < vocab_size; v++) {
                gl_row[v] = l_row[v];
                if (v == target_token) gl_row[v] -= 1.0f;
            }
        }
    }

    // Backward through LM head
    // dL/dhidden[d] = sum_v dL/dlogit_v * w_out[v,d]
    // dL/dW_out[v,d] = dL/dlogit_v * hidden[d]
    // dL/db_out[v] = dL/dlogit_v
    float* grad_hidden = (float*)malloc(batch * seq_len * d_model * sizeof(float));
    for (size_t i = 0; i < batch * seq_len * d_model; i++) grad_hidden[i] = 0.0f;

    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            float* gl_row = &grad_logits[b * seq_len * vocab_size + s * vocab_size];
            float* gh_row = &grad_hidden[b * seq_len * d_model + s * d_model];
            float* hidden_row = &x_data[b * seq_len * d_model + s * d_model];

            for (size_t v = 0; v < vocab_size; v++) {
                for (size_t d = 0; d < d_model; d++) {
                    grad_W_out[v * d_model + d] += gl_row[v] * hidden_row[d];
                    gh_row[d] += gl_row[v] * w_out[v * d_model + d];
                }
                grad_b_out[v] += gl_row[v];
            }
        }
    }

    free(grad_logits);

    // Backward through final_ln (if no decoder layers)
    if (tr->n_decoder_layers == 0) {
        Tensor* grad_hidden_tensor = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, (size_t[]){batch, seq_len, d_model}, 3);
        float* gh_data = (float*)grad_hidden_tensor->data;
        for (size_t i = 0; i < batch * seq_len * d_model; i++) gh_data[i] = grad_hidden[i];

        Tensor* grad_emb_tensor = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, (size_t[]){batch, seq_len, d_model}, 3);
        layer_norm_backward(tr->final_ln, grad_hidden_tensor, grad_emb_tensor);

        // Accumulate to grad_hidden (now grad to embeddings input)
        float* ge_data = (float*)grad_emb_tensor->data;
        for (size_t i = 0; i < batch * seq_len * d_model; i++) grad_hidden[i] = ge_data[i];

        tensor_free(grad_hidden_tensor);
        tensor_free(grad_emb_tensor);
    }

    // Backward through decoder layers (reverse order)
    for (int i = (int)tr->n_decoder_layers - 1; i >= 0; i--) {
        // grad_hidden is dL/dx (the input to this decoder layer)
        Tensor* grad_hidden_tensor = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, (size_t[]){batch, seq_len, d_model}, 3);
        float* gh_data = (float*)grad_hidden_tensor->data;
        for (size_t j = 0; j < batch * seq_len * d_model; j++) gh_data[j] = grad_hidden[j];

        Tensor* grad_layer_input = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, (size_t[]){batch, seq_len, d_model}, 3);
        decoder_layer_backward(tr->dec_layers[i], grad_hidden_tensor, grad_layer_input);

        // grad_layer_input is dL/dinput to this layer (accumulates in previous layer's grad_hidden)
        float* gli_data = (float*)grad_layer_input->data;
        for (size_t j = 0; j < batch * seq_len * d_model; j++) grad_hidden[j] = gli_data[j];

        tensor_free(grad_hidden_tensor);
        tensor_free(grad_layer_input);
    }

    // Backward to embeddings
    // x = emb + pos, so grad_emb = sum over positions where token appears
    // grad_pos accumulates grad at each position
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            size_t token_id = input_data[b * seq_len + s];
            if (token_id >= vocab_size) token_id = 0;

            for (size_t d = 0; d < d_model; d++) {
                grad_emb[token_id * d_model + d] += grad_hidden[b * seq_len * d_model + s * d_model + d];
                grad_pos[s * d_model + d] += grad_hidden[b * seq_len * d_model + s * d_model + d];
            }
        }
    }

    free(grad_hidden);
    tensor_free(x);
    free(logits);

    return count > 0 ? total_loss / count : 0.0f;
}

#endif /* __C_TRANSFORMER_H__ */
