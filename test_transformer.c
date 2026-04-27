/**
 * @file test_transformer.c
 * @brief Test Transformer model
 */

#include "transformer.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int nearly_equal(float a, float b, float eps) {
    return fabsf(a - b) < eps;
}

static int test_transformer_create(void) {
    printf("=== test_transformer_create ===\n");

    TransformerConfig config = {
        .vocab_size = 100,
        .d_model = 64,
        .n_heads = 4,
        .n_encoder_layers = 2,
        .n_decoder_layers = 2,
        .d_ff = 256,
        .max_seq_len = 50,
        .dropout_p = 0.1f
    };

    Transformer* model = transformer_create(config);
    if (!model) {
        printf("FAIL: model is NULL\n");
        return 1;
    }

    if (model->config.vocab_size != 100 || model->config.d_model != 64) {
        printf("FAIL: wrong config\n");
        transformer_free(model);
        return 1;
    }

    printf("PASS\n");
    transformer_free(model);
    return 0;
}

static int test_transformer_forward(void) {
    printf("=== test_transformer_forward ===\n");

    TransformerConfig config = {
        .vocab_size = 100,
        .d_model = 64,
        .n_heads = 4,
        .n_encoder_layers = 2,
        .n_decoder_layers = 2,
        .d_ff = 256,
        .max_seq_len = 50,
        .dropout_p = 0.0f
    };

    Transformer* model = transformer_create(config);

    // Input: [batch=2, seq_len=8] (token IDs)
    size_t shape[] = {2, 8};
    Tensor* input = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 2);
    uint32_t* data = (uint32_t*)input->data;
    for (size_t i = 0; i < input->size; i++) {
        data[i] = rand() % 100;
    }

    Tensor* output = transformer_forward(model, input);

    if (!output) {
        printf("FAIL: output is NULL\n");
        transformer_free(model);
        tensor_free(input);
        return 1;
    }

    // Check shape: [batch=2, seq_len=8, d_model=64]
    if (output->ndim != 3 ||
        output->shape[0] != 2 ||
        output->shape[1] != 8 ||
        output->shape[2] != 64) {
        printf("FAIL: expected shape [2, 8, 64], got [%zu, %zu, %zu]\n",
               output->shape[0], output->shape[1], output->shape[2]);
        transformer_free(model);
        tensor_free(input);
        tensor_free(output);
        return 1;
    }

    // Check for NaN/Inf
    float* out = (float*)output->data;
    int valid = 1;
    for (size_t i = 0; i < output->size; i++) {
        if (isnan(out[i]) || isinf(out[i])) {
            valid = 0;
            break;
        }
    }

    if (!valid) {
        printf("FAIL: output contains NaN or Inf\n");
        transformer_free(model);
        tensor_free(input);
        tensor_free(output);
        return 1;
    }

    printf("PASS\n");
    transformer_free(model);
    tensor_free(input);
    tensor_free(output);
    return 0;
}

static int test_transformer_multi_head_attention(void) {
    printf("=== test_transformer_multi_head_attention ===\n");

    MHAttention* mha = mha_create(64, 4);

    // Input: [batch=2, seq_len=8, d_model=64]
    size_t shape[] = {2, 8, 64};
    Tensor* input = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 3);
    float* data = (float*)input->data;
    for (size_t i = 0; i < input->size; i++) {
        data[i] = (float)(rand() % 100) / 100.0f;
    }

    Tensor* output = mha_forward(mha, input);

    if (!output) {
        printf("FAIL: output is NULL\n");
        mha_free(mha);
        tensor_free(input);
        return 1;
    }

    // Check shape
    if (output->ndim != 3 ||
        output->shape[0] != 2 ||
        output->shape[1] != 8 ||
        output->shape[2] != 64) {
        printf("FAIL: expected shape [2, 8, 64]\n");
        mha_free(mha);
        tensor_free(input);
        tensor_free(output);
        return 1;
    }

    printf("PASS\n");
    mha_free(mha);
    tensor_free(input);
    tensor_free(output);
    return 0;
}

static int test_transformer_ffn(void) {
    printf("=== test_transformer_ffn ===\n");

    FFN* ffn = ffn_create(64, 256);

    // Input: [batch=2, seq_len=8, d_model=64]
    size_t shape[] = {2, 8, 64};
    Tensor* input = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 3);
    float* data = (float*)input->data;
    for (size_t i = 0; i < input->size; i++) {
        data[i] = (float)(rand() % 100) / 100.0f;
    }

    Tensor* output = ffn_forward(ffn, input);

    if (!output) {
        printf("FAIL: output is NULL\n");
        ffn_free(ffn);
        tensor_free(input);
        return 1;
    }

    // Check shape
    if (output->ndim != 3 ||
        output->shape[0] != 2 ||
        output->shape[1] != 8 ||
        output->shape[2] != 64) {
        printf("FAIL: expected shape [2, 8, 64]\n");
        ffn_free(ffn);
        tensor_free(input);
        tensor_free(output);
        return 1;
    }

    // Check for NaN/Inf
    float* out = (float*)output->data;
    int valid = 1;
    for (size_t i = 0; i < output->size; i++) {
        if (isnan(out[i]) || isinf(out[i])) {
            valid = 0;
            break;
        }
    }

    if (!valid) {
        printf("FAIL: output contains NaN or Inf\n");
        ffn_free(ffn);
        tensor_free(input);
        tensor_free(output);
        return 1;
    }

    printf("PASS\n");
    ffn_free(ffn);
    tensor_free(input);
    tensor_free(output);
    return 0;
}

static int test_transformer_layer_norm(void) {
    printf("=== test_transformer_layer_norm ===\n");

    LayerNorm* ln = layer_norm_create(64, 1e-5f);

    // Input: [batch=2, seq_len=8, d_model=64]
    size_t shape[] = {2, 8, 64};
    Tensor* input = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 3);
    float* data = (float*)input->data;
    for (size_t i = 0; i < input->size; i++) {
        data[i] = (float)(rand() % 100) / 100.0f;
    }

    Tensor* output = layer_norm_forward(ln, input);

    if (!output) {
        printf("FAIL: output is NULL\n");
        layer_norm_free(ln);
        tensor_free(input);
        return 1;
    }

    // Check shape
    if (output->ndim != 3 ||
        output->shape[0] != 2 ||
        output->shape[1] != 8 ||
        output->shape[2] != 64) {
        printf("FAIL: expected shape [2, 8, 64]\n");
        layer_norm_free(ln);
        tensor_free(input);
        tensor_free(output);
        return 1;
    }

    // Check mean ~0, var ~1 for each sequence position
    float* out = (float*)output->data;
    int valid = 1;
    for (size_t b = 0; b < 2 && valid; b++) {
        for (size_t s = 0; s < 8 && valid; s++) {
            float sum = 0.0f;
            for (size_t d = 0; d < 64; d++) {
                sum += out[b * 8 * 64 + s * 64 + d];
            }
            float mean = sum / 64.0f;
            if (fabsf(mean) > 0.1f) {
                printf("FAIL: mean not close to 0 (mean=%.4f)\n", mean);
                valid = 0;
                break;
            }
        }
    }

    if (!valid) {
        layer_norm_free(ln);
        tensor_free(input);
        tensor_free(output);
        return 1;
    }

    printf("PASS\n");
    layer_norm_free(ln);
    tensor_free(input);
    tensor_free(output);
    return 0;
}

int main(void) {
    printf("\n=== Transformer Tests ===\n\n");
    srand((unsigned int)time(NULL));

    int failures = 0;
    failures += test_transformer_create();
    failures += test_transformer_forward();
    failures += test_transformer_multi_head_attention();
    failures += test_transformer_ffn();
    failures += test_transformer_layer_norm();

    printf("\n=== Summary: %d failures ===\n", failures);
    return failures ? 1 : 0;
}
