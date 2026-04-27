/**
 * @file test_rnn.c
 * @brief Test RNN model
 */

#include "rnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int nearly_equal(float a, float b, float eps) {
    return fabsf(a - b) < eps;
}

static int test_rnn_create(void) {
    printf("=== test_rnn_create ===\n");

    RNN* model = rnn_create(10, 32, 1);
    if (!model) {
        printf("FAIL: model is NULL\n");
        return 1;
    }

    if (model->input_size != 10 || model->hidden_size != 32) {
        printf("FAIL: wrong dimensions\n");
        rnn_free(model);
        return 1;
    }

    printf("PASS\n");
    rnn_free(model);
    return 0;
}

static int test_rnn_forward(void) {
    printf("=== test_rnn_forward ===\n");

    RNN* model = rnn_create(10, 32, 1);

    // Input: [batch=2, seq_len=5, input_size=10]
    size_t shape[] = {2, 5, 10};
    Tensor* input = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 3);
    float* data = (float*)input->data;
    for (size_t i = 0; i < input->size; i++) {
        data[i] = (float)(rand() % 100) / 100.0f;
    }

    Tensor* output = rnn_forward(model, input);

    if (!output) {
        printf("FAIL: output is NULL\n");
        rnn_free(model);
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
        rnn_free(model);
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
        rnn_free(model);
        tensor_free(input);
        tensor_free(output);
        return 1;
    }

    printf("PASS\n");
    tensor_free(output);
    tensor_free(input);
    rnn_free(model);
    return 0;
}

static int test_rnn_backward_only(void) {
    printf("=== test_rnn_backward_only ===\n");

    RNN* model = rnn_create(10, 32, 1);

    // Input
    size_t shape[] = {2, 5, 10};
    Tensor* input = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 3);
    float* data = (float*)input->data;
    for (size_t i = 0; i < input->size; i++) {
        data[i] = (float)(rand() % 100) / 100.0f;
    }

    // Forward
    Tensor* output = rnn_forward(model, input);

    // Create gradient output
    size_t grad_shape[] = {2, 5, 32};
    Tensor* grad = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, grad_shape, 3);
    float* grad_data = (float*)grad->data;
    for (size_t i = 0; i < grad->size; i++) {
        grad_data[i] = 0.1f;
    }

    // Backward
    rnn_backward(model, grad);

    printf("PASS\n");

    tensor_free(output);
    tensor_free(grad);
    tensor_free(input);
    rnn_free(model);
    return 0;
}

static int test_rnn_train_step_simple(void) {
    printf("=== test_rnn_train_step_simple ===\n");

    RNN* model = rnn_create(5, 8, 1);

    // Input: [batch=1, seq_len=3, input_size=5]
    size_t shape[] = {1, 3, 5};
    Tensor* input = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 3);
    float* data = (float*)input->data;
    for (size_t i = 0; i < input->size; i++) {
        data[i] = 0.5f;
    }

    // Targets: same shape
    size_t target_shape[] = {1, 3, 8};
    Tensor* targets = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, target_shape, 3);
    float* tdata = (float*)targets->data;
    for (size_t i = 0; i < targets->size; i++) {
        tdata[i] = 0.3f;
    }

    float loss = rnn_train_step(model, 0.001f, input, targets);

    if (isnan(loss) || isinf(loss)) {
        printf("FAIL: loss is NaN or Inf\n");
        tensor_free(input);
        tensor_free(targets);
        rnn_free(model);
        return 1;
    }

    printf("Loss: %.4f\n", loss);
    printf("PASS\n");

    tensor_free(input);
    tensor_free(targets);
    rnn_free(model);
    return 0;
}

int main(void) {
    printf("\n=== RNN Tests ===\n\n");
    srand((unsigned int)time(NULL));

    int failures = 0;
    failures += test_rnn_create();
    failures += test_rnn_forward();
    failures += test_rnn_backward_only();
    failures += test_rnn_train_step_simple();

    printf("\n=== Summary: %d failures ===\n", failures);
    return failures ? 1 : 0;
}
