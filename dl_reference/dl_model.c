/**
 * @file dl_model.c
 * @brief Model implementations
 */

#include "dl_model.h"
#include "dl_layers.h"
#include "dl_loss.h"
#include "dl_tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * MLP (MULTI-LAYER PERCEPTRON)
 * ============================================================================ */

DL_MLP *dl_mlp_create(const DL_MLPConfig *config) {
    if (!config) return NULL;

    DL_MLP *mlp = (DL_MLP *)calloc(1, sizeof(DL_MLP));
    if (!mlp) return NULL;

    mlp->type = DL_MODEL_MLP;
    mlp->name = "MLP";
    mlp->training = true;
    mlp->input_size = config->input_size;
    mlp->output_size = config->output_size;
    mlp->activation = config->activation;
    mlp->use_dropout = false;
    mlp->dropout_p = 0.0;

    /* Calculate total layers: n_hidden + 1 output */
    mlp->n_layers = config->n_hidden + 1;
    mlp->layers = (DenseParams **)calloc(mlp->n_layers, sizeof(DenseParams *));

    /* Create hidden layers */
    size_t prev_size = config->input_size;
    for (size_t i = 0; i < config->n_hidden; i++) {
        mlp->layers[i] = dense_create(prev_size, config->hidden_sizes[i],
                                      config->use_bias, true);
        prev_size = config->hidden_sizes[i];
    }

    /* Create output layer */
    mlp->layers[config->n_hidden] = dense_create(prev_size, config->output_size,
                                                  config->use_bias, true);

    return mlp;
}

static Tensor *apply_activation(const Tensor *x, DL_ActivationType act) {
    switch (act) {
        case DL_ACTIVATION_RELU:
            return tensor_relu(x);
        case DL_ACTIVATION_SIGMOID:
            return tensor_sigmoid(x);
        case DL_ACTIVATION_TANH:
            return tensor_tanh_activation(x);
        case DL_ACTIVATION_GELU:
            return tensor_gelu(x);
        default:
            return tensor_clone(x);
    }
}

Tensor *dl_mlp_forward(DL_MLP *mlp, const Tensor *input) {
    if (!mlp || !input) return NULL;

    Tensor *x = (Tensor *)input;

    for (size_t i = 0; i < mlp->n_layers; i++) {
        x = dense_forward(mlp->layers[i], x);

        /* Apply activation for all but last layer */
        if (i < mlp->n_layers - 1) {
            Tensor *x_act = apply_activation(x, mlp->activation);
            tensor_free(x);
            x = x_act;
        }

        /* Apply dropout during training */
        if (mlp->use_dropout && mlp->training && i < mlp->n_layers - 1) {
            Tensor *dropout_mask = NULL;
            Tensor *x_drop = tensor_dropout(x, mlp->dropout_p, true, &dropout_mask);
            tensor_free(x);
            x = x_drop;
            if (dropout_mask) tensor_free(dropout_mask);
        }
    }

    return x;
}

void dl_mlp_backward(DL_MLP *mlp, const Tensor *grad_output) {
    (void)mlp;
    (void)grad_output;
    /* Backward pass implementation would go here */
}

Tensor **dl_mlp_parameters(DL_MLP *mlp, size_t *n_params) {
    if (!mlp || !n_params) return NULL;

    *n_params = mlp->n_layers * 2;  /* weight + bias per layer */
    Tensor **params = (Tensor **)malloc(sizeof(Tensor *) * (*n_params));

    size_t idx = 0;
    for (size_t i = 0; i < mlp->n_layers; i++) {
        params[idx++] = mlp->layers[i]->weight;
        if (mlp->layers[i]->bias) {
            params[idx++] = mlp->layers[i]->bias;
        }
    }

    return params;
}

void dl_mlp_free(DL_MLP *mlp) {
    if (!mlp) return;

    for (size_t i = 0; i < mlp->n_layers; i++) {
        if (mlp->layers[i]) {
            dense_free(mlp->layers[i]);
        }
    }

    free(mlp->layers);
    free(mlp);
}

/* ============================================================================
 * CNN (CONVOLUTIONAL NEURAL NETWORK)
 * ============================================================================ */

DL_CNN *dl_cnn_create(size_t in_channels, size_t num_classes) {
    DL_CNN *cnn = (DL_CNN *)calloc(1, sizeof(DL_CNN));
    if (!cnn) return NULL;

    cnn->type = DL_MODEL_CNN;
    cnn->name = "CNN";
    cnn->training = true;
    cnn->num_classes = num_classes;

    /* Simple CNN: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> FC -> FC */
    cnn->n_conv_layers = 2;
    cnn->conv_layers = (Conv2DParams **)calloc(2, sizeof(Conv2DParams *));

    cnn->conv_layers[0] = conv2d_create(in_channels, 32, 3, 3, 1, 1, 1, 1, true, true);
    cnn->conv_layers[1] = conv2d_create(32, 64, 3, 3, 1, 1, 1, 1, true, true);

    cnn->n_pool_layers = 2;
    cnn->pool_layers = (MaxPool2DParams **)calloc(2, sizeof(MaxPool2DParams *));
    cnn->pool_layers[0] = maxpool2d_create(2, 2, 2, 2);
    cnn->pool_layers[1] = maxpool2d_create(2, 2, 2, 2);

    /* FC layers: after 2 pools, 8x8 -> 64 channels */
    cnn->n_fc_layers = 2;
    cnn->fc_layers = (DenseParams **)calloc(2, sizeof(DenseParams *));
    cnn->fc_layers[0] = dense_create(64 * 8 * 8, 512, true, true);
    cnn->fc_layers[1] = dense_create(512, num_classes, true, true);

    return cnn;
}

Tensor *dl_cnn_forward(DL_CNN *cnn, const Tensor *input) {
    if (!cnn || !input) return NULL;

    /* Assuming input shape: [batch, channels, height, width] */
    Tensor *x = (Tensor *)input;

    /* Conv1 -> ReLU -> Pool1 */
    x = conv2d_forward(cnn->conv_layers[0], x);
    x = tensor_relu(x);
    tensor_free(x);
    x = maxpool2d_forward(cnn->pool_layers[0], x);

    /* Conv2 -> ReLU -> Pool2 */
    x = conv2d_forward(cnn->conv_layers[1], x);
    x = tensor_relu(x);
    tensor_free(x);
    x = maxpool2d_forward(cnn->pool_layers[1], x);

    /* Flatten */
    size_t batch = input->shape[0];
    size_t flat_size = 64 * 8 * 8;
    size_t flat_shape[] = {batch, flat_size};
    Tensor *x_flat = tensor_reshape(x, 2, flat_shape);
    tensor_free(x);
    x = x_flat;

    /* FC1 -> ReLU */
    x = dense_forward(cnn->fc_layers[0], x);
    x = tensor_relu(x);

    /* FC2 (output) */
    x = dense_forward(cnn->fc_layers[1], x);

    return x;
}

void dl_cnn_backward(DL_CNN *cnn, const Tensor *grad_output) {
    (void)cnn;
    (void)grad_output;
}

Tensor **dl_cnn_parameters(DL_CNN *cnn, size_t *n_params) {
    if (!cnn || !n_params) return NULL;

    size_t n = 0;
    for (size_t i = 0; i < cnn->n_conv_layers; i++) {
        n += 2;  /* weight + bias */
    }
    for (size_t i = 0; i < cnn->n_fc_layers; i++) {
        n += 2;  /* weight + bias */
    }

    *n_params = n;
    Tensor **params = (Tensor **)malloc(sizeof(Tensor *) * n);

    size_t idx = 0;
    for (size_t i = 0; i < cnn->n_conv_layers; i++) {
        params[idx++] = cnn->conv_layers[i]->weight;
        params[idx++] = cnn->conv_layers[i]->bias;
    }
    for (size_t i = 0; i < cnn->n_fc_layers; i++) {
        params[idx++] = cnn->fc_layers[i]->weight;
        params[idx++] = cnn->fc_layers[i]->bias;
    }

    return params;
}

void dl_cnn_free(DL_CNN *cnn) {
    if (!cnn) return;

    for (size_t i = 0; i < cnn->n_conv_layers; i++) {
        if (cnn->conv_layers[i]) conv2d_free(cnn->conv_layers[i]);
    }
    for (size_t i = 0; i < cnn->n_pool_layers; i++) {
        if (cnn->pool_layers[i]) maxpool2d_free(cnn->pool_layers[i]);
    }
    for (size_t i = 0; i < cnn->n_fc_layers; i++) {
        if (cnn->fc_layers[i]) dense_free(cnn->fc_layers[i]);
    }

    free(cnn->conv_layers);
    free(cnn->pool_layers);
    free(cnn->fc_layers);
    free(cnn);
}

/* ============================================================================
 * TRANSFORMER
 * ============================================================================ */

static Tensor *scaled_dot_product_attention(const Tensor *q, const Tensor *k,
                                              const Tensor *v, double scale) {
    /* Compute Q @ K^T */
    Tensor *k_t = tensor_transpose(k);
    Tensor *qk = tensor_matmul(q, k_t);
    tensor_free(k_t);

    /* Scale */
    Tensor *qk_scaled = tensor_mul_scalar(qk, scale);
    tensor_free(qk);

    /* Softmax */
    Tensor *attn_weights = tensor_softmax(qk_scaled, -1);
    tensor_free(qk_scaled);

    /* Attention output: softmax(QK^T/sqrt(d_k)) @ V */
    Tensor *attn_output = tensor_matmul(attn_weights, v);
    tensor_free(attn_weights);

    return attn_output;
}

static Tensor *multihead_attention(const Tensor *x, DenseParams *wq, DenseParams *wk,
                                   DenseParams *wv, DenseParams *wo,
                                   size_t n_heads, size_t d_model) {
    size_t d_k = d_model / n_heads;

    /* Project x to Q, K, V */
    Tensor *q = dense_forward(wq, x);
    Tensor *k = dense_forward(wk, x);
    Tensor *v = dense_forward(wv, x);

    /* Reshape for multi-head: [batch, seq, n_heads, d_k] -> [batch, n_heads, seq, d_k] */
    /* Simplified: just use single-head attention for now */
    Tensor *attn_out = scaled_dot_product_attention(q, k, v, 1.0 / sqrt((double)d_k));

    tensor_free(q);
    tensor_free(k);
    tensor_free(v);

    /* Output projection */
    attn_out = dense_forward(wo, attn_out);

    return attn_out;
}

static Tensor *feed_forward(const Tensor *x, DenseParams *w1, DenseParams *b1,
                            DenseParams *w2, DenseParams *b2) {
    Tensor *h = dense_forward(w1, x);
    tensor_free(h);
    h = tensor_relu(x);

    Tensor *h2 = dense_forward(w2, h);
    tensor_free(h);

    return h2;
}

DL_Transformer *dl_transformer_create(size_t vocab_size, size_t d_model,
                                      size_t n_heads, size_t n_encoder_layers,
                                      size_t n_decoder_layers, size_t d_ff,
                                      size_t max_seq_len, double dropout_p) {
    DL_Transformer *tr = (DL_Transformer *)calloc(1, sizeof(DL_Transformer));
    if (!tr) return NULL;

    tr->type = DL_MODEL_TRANSFORMER;
    tr->name = "Transformer";
    tr->training = true;
    tr->vocab_size = vocab_size;
    tr->d_model = d_model;
    tr->n_heads = n_heads;
    tr->d_ff = d_ff;
    tr->max_seq_len = max_seq_len;
    tr->dropout_p = dropout_p;
    tr->n_encoder_layers = n_encoder_layers;
    tr->n_decoder_layers = n_decoder_layers;

    /* Token embedding */
    size_t emb_shape[] = {vocab_size, d_model};
    tr->token_embedding = tensor_create(2, emb_shape, DL_DEVICE_CPU, true);
    for (size_t i = 0; i < tr->token_embedding->size; i++) {
        tr->token_embedding->data[i] = ((double)rand() / RAND_MAX * 2 - 1) * 0.01;
    }

    /* Positional embedding */
    size_t pos_shape[] = {max_seq_len, d_model};
    tr->pos_embedding = tensor_create(2, pos_shape, DL_DEVICE_CPU, true);
    for (size_t i = 0; i < tr->pos_embedding->size; i++) {
        tr->pos_embedding->data[i] = ((double)rand() / RAND_MAX * 2 - 1) * 0.01;
    }

    /* Allocate encoder layers */
    if (n_encoder_layers > 0) {
        tr->enc_layers = (DL_TransformerEncoderLayer *)calloc(n_encoder_layers,
                                                               sizeof(DL_TransformerEncoderLayer));
        for (size_t i = 0; i < n_encoder_layers; i++) {
            DL_TransformerEncoderLayer *layer = &tr->enc_layers[i];
            layer->d_model = d_model;
            layer->n_heads = n_heads;
            layer->d_ff = d_ff;

            layer->wq = dense_create(d_model, d_model, true, true);
            layer->wk = dense_create(d_model, d_model, true, true);
            layer->wv = dense_create(d_model, d_model, true, true);
            layer->wo = dense_create(d_model, d_model, true, true);

            layer->ff_w1 = dense_create(d_model, d_ff, true, true);
            layer->ff_b1 = dense_create(d_ff, 1, true, true);
            layer->ff_w2 = dense_create(d_ff, d_model, true, true);
            layer->ff_b2 = dense_create(d_model, 1, true, true);

            layer->ln1 = layer_norm_create(d_model, 1e-5, true);
            layer->ln2 = layer_norm_create(d_model, 1e-5, true);
        }
    }

    /* Output projection */
    tr->output_projection = dense_create(d_model, vocab_size, true, true);

    return tr;
}

Tensor *dl_transformer_forward(DL_Transformer *tr, const Tensor *input,
                               const Tensor *target, bool training) {
    (void)target;
    if (!tr || !input) return NULL;

    size_t batch_size = input->shape[0];
    size_t seq_len = input->shape[1];

    /* Embed tokens: token_embedding[input] + pos_embedding */
    /* Simplified: just use token embedding for now */
    Tensor *x = (Tensor *)input;  /* input should be token indices */

    /* Encode positions */
    /* For simplicity, we'll just pass through the encoder directly */

    if (tr->n_encoder_layers > 0) {
        /* Multi-head self-attention */
        DL_TransformerEncoderLayer *enc = &tr->enc_layers[0];
        Tensor *attn_out = multihead_attention(x, enc->wq, enc->wk, enc->wv, enc->wo,
                                                tr->n_heads, tr->d_model);

        /* Add & norm */
        Tensor *attn_plus_x = tensor_add(attn_out, x);
        tensor_free(attn_out);
        tensor_free(x);
        x = layer_norm_forward(enc->ln1, attn_plus_x);
        tensor_free(attn_plus_x);

        /* Feed forward */
        Tensor *ff_out = feed_forward(x, enc->ff_w1, enc->ff_b1, enc->ff_w2, enc->ff_b2);
        tensor_free(x);

        /* Add & norm */
        Tensor *ff_plus_x = tensor_add(ff_out, x);
        tensor_free(ff_out);
        x = layer_norm_forward(enc->ln2, ff_plus_x);
        tensor_free(ff_plus_x);
    }

    /* Output projection to vocabulary */
    Tensor *logits = dense_forward(tr->output_projection, x);
    tensor_free(x);

    return logits;
}

Tensor **dl_transformer_parameters(DL_Transformer *tr, size_t *n_params) {
    if (!tr || !n_params) return NULL;

    /* Count parameters */
    size_t n = 0;
    n++;  /* token_embedding */
    n++;  /* pos_embedding */
    n += 6 * tr->n_encoder_layers;  /* wq, wk, wv, wo, ff_w1, ff_w2 + norms */
    n += 2;  /* output_projection */

    *n_params = n;
    Tensor **params = (Tensor **)malloc(sizeof(Tensor *) * n);

    size_t idx = 0;
    params[idx++] = tr->token_embedding;
    params[idx++] = tr->pos_embedding;

    for (size_t i = 0; i < tr->n_encoder_layers; i++) {
        DL_TransformerEncoderLayer *enc = &tr->enc_layers[i];
        params[idx++] = enc->wq->weight;
        params[idx++] = enc->wk->weight;
        params[idx++] = enc->wv->weight;
        params[idx++] = enc->wo->weight;
        params[idx++] = enc->ff_w1->weight;
        params[idx++] = enc->ff_w2->weight;
    }

    params[idx++] = tr->output_projection->weight;

    return params;
}

void dl_transformer_free(DL_Transformer *transformer) {
    if (!transformer) return;

    if (transformer->token_embedding) tensor_free(transformer->token_embedding);
    if (transformer->pos_embedding) tensor_free(transformer->pos_embedding);

    for (size_t i = 0; i < transformer->n_encoder_layers; i++) {
        DL_TransformerEncoderLayer *enc = &transformer->enc_layers[i];
        dense_free(enc->wq);
        dense_free(enc->wk);
        dense_free(enc->wv);
        dense_free(enc->wo);
        dense_free(enc->ff_w1);
        dense_free(enc->ff_w2);
        layer_norm_free(enc->ln1);
        layer_norm_free(enc->ln2);
    }

    free(transformer->enc_layers);
    dense_free(transformer->output_projection);
    free(transformer);
}

/* ============================================================================
 * TRAINING UTILITIES
 * ============================================================================ */

DL_TrainingResult dl_train_epoch(DL_MLP *model, const Tensor *X, const Tensor *y,
                                DL_Optimizer *optimizer, double (*loss_fn)(Tensor *, Tensor *)) {
    (void)model;
    (void)X;
    (void)y;
    (void)optimizer;
    (void)loss_fn;

    DL_TrainingResult result = {0};
    return result;
}

DL_TrainingResult dl_evaluate(DL_MLP *model, const Tensor *X, const Tensor *y,
                              double (*loss_fn)(Tensor *, Tensor *)) {
    (void)model;
    (void)X;
    (void)y;
    (void)loss_fn;

    DL_TrainingResult result = {0};
    return result;
}

Tensor *dl_predict(DL_MLP *model, const Tensor *input) {
    return dl_mlp_forward(model, input);
}
