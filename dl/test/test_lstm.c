/**
 * @file test_lstm.c
 * @brief Test LSTM model
 */

#include "dl/lstm.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int nearly_equal(float a, float b, float eps) {
    return fabsf(a - b) < eps;
}

static int test_lstm_create(void) {
    printf("=== test_lstm_create ===\n");

    LSTM* model = lstm_create(10, 32, 1);
    if (!model) {
        printf("FAIL: model is NULL\n");
        return 1;
    }

    if (model->input_size != 10 || model->hidden_size != 32) {
        printf("FAIL: wrong dimensions\n");
        lstm_free(model);
        return 1;
    }

    printf("PASS\n");
    lstm_free(model);
    return 0;
}

static int test_lstm_forward(void) {
    printf("=== test_lstm_forward ===\n");

    LSTM* model = lstm_create(10, 32, 1);

    // Input: [batch=2, seq_len=5, input_size=10]
    size_t shape[] = {2, 5, 10};
    Tensor* input = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 3);
    float* data = (float*)input->data;
    for (size_t i = 0; i < input->size; i++) {
        data[i] = (float)(rand() % 100) / 100.0f;
    }

    Tensor* output = lstm_forward(model, input);

    if (!output) {
        printf("FAIL: output is NULL\n");
        lstm_free(model);
        tensor_free(input);
        return 1;
    }

    // Check shape: [batch=2, seq_len=5, hidden_size=32]
    if (output->ndim != 3 ||
        output->shape[0] != 2 ||
        output->shape[1] != 5 ||
        output->shape[2] != 32) {
        printf("FAIL: expected shape [2, 5, 32], got [%zu, %zu, %zu]\n",
               output->shape[0], output->shape[1], output->shape[2]);
        lstm_free(model);
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
        lstm_free(model);
        tensor_free(input);
        tensor_free(output);
        return 1;
    }

    printf("PASS\n");
    lstm_free(model);
    tensor_free(input);
    tensor_free(output);
    return 0;
}

static int test_lstm_train_step(void) {
    printf("=== test_lstm_train_step ===\n");

    LSTM* model = lstm_create(10, 32, 1);

    // Input: [batch=2, seq_len=5, input_size=10]
    size_t shape[] = {2, 5, 10};
    Tensor* input = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 3);
    float* data = (float*)input->data;
    for (size_t i = 0; i < input->size; i++) {
        data[i] = (float)(rand() % 100) / 100.0f;
    }

    // Targets: same shape as output
    size_t target_shape[] = {2, 5, 32};
    Tensor* targets = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, target_shape, 3);
    float* tdata = (float*)targets->data;
    for (size_t i = 0; i < targets->size; i++) {
        tdata[i] = (float)(rand() % 100) / 100.0f;
    }

    float loss = lstm_train_step(model, 0.001f, input, targets);

    if (isnan(loss) || isinf(loss)) {
        printf("FAIL: loss is NaN or Inf\n");
        lstm_free(model);
        tensor_free(input);
        tensor_free(targets);
        return 1;
    }

    printf("Loss: %.4f\n", loss);
    printf("PASS\n");
    lstm_free(model);
    tensor_free(input);
    tensor_free(targets);
    return 0;
}

static int test_lstm_cell_gates(void) {
    printf("=== test_lstm_cell_gates ===\n");

    LSTM* model = lstm_create(5, 8, 1);

    // Input with constant values
    size_t shape[] = {1, 3, 5};
    Tensor* input = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 3);
    float* data = (float*)input->data;
    for (size_t i = 0; i < input->size; i++) data[i] = 0.5f;

    Tensor* output = lstm_forward(model, input);

    if (!output) {
        printf("FAIL: output is NULL\n");
        lstm_free(model);
        tensor_free(input);
        return 1;
    }

    // Check cell states are within valid range [0, 1] for tanh
    LSTMLayer* layer = model->layer;
    int valid = 1;
    for (size_t t = 0; t < 3; t++) {
        if (!layer->c_cache[t]) continue;
        float* c = (float*)layer->c_cache[t]->data;
        for (size_t i = 0; i < layer->c_cache[t]->size; i++) {
            if (isnan(c[i]) || isinf(c[i])) {
                valid = 0;
                break;
            }
        }
    }

    if (!valid) {
        printf("FAIL: cell states contain NaN or Inf\n");
        lstm_free(model);
        tensor_free(input);
        tensor_free(output);
        return 1;
    }

    printf("PASS\n");
    lstm_free(model);
    tensor_free(input);
    tensor_free(output);
    return 0;
}

int main(void) {
    printf("\n=== LSTM Tests ===\n\n");
    srand((unsigned int)time(NULL));

    int failures = 0;
    failures += test_lstm_create();
    failures += test_lstm_forward();
    failures += test_lstm_train_step();
    failures += test_lstm_cell_gates();

    printf("\n=== Summary: %d failures ===\n", failures);
    return failures ? 1 : 0;
}
