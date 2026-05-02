/**
 * @file test_lenet5.c
 * @brief Test LeNet-5 model
 */

#include "dl/lenet5.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int nearly_equal(float a, float b, float eps) {
    return fabsf(a - b) < eps;
}

static int test_lenet5_create(void) {
    printf("=== test_lenet5_create ===\n");

    LeNet5* model = lenet5_create();
    if (!model) {
        printf("FAIL: model is NULL\n");
        return 1;
    }

    if (!model->conv1 || !model->conv2 || !model->conv3) {
        printf("FAIL: conv layers not created\n");
        return 1;
    }

    if (!model->fc1 || !model->fc2) {
        printf("FAIL: fc layers not created\n");
        return 1;
    }

    printf("PASS\n");
    lenet5_free(model);
    return 0;
}

static int test_lenet5_forward(void) {
    printf("=== test_lenet5_forward ===\n");

    LeNet5* model = lenet5_create();

    // Create input: [N=2, C=1, H=32, W=32]
    size_t shape[] = {2, 1, 32, 32};
    Tensor* input = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 4);
    float* data = (float*)input->data;
    for (size_t i = 0; i < input->size; i++) data[i] = (float)(rand() % 100) / 100.0f;

    // Forward pass
    Tensor* output = lenet5_forward(model, input);

    if (!output) {
        printf("FAIL: output is NULL\n");
        lenet5_free(model);
        tensor_free(input);
        return 1;
    }

    // Check output shape: [N=2, 10]
    if (output->ndim != 2 || output->shape[0] != 2 || output->shape[1] != 10) {
        printf("FAIL: expected shape [2, 10], got [%zu, %zu]\n",
               output->shape[0], output->shape[1]);
        lenet5_free(model);
        tensor_free(input);
        tensor_free(output);
        return 1;
    }

    // Check output values are valid (not NaN or Inf)
    float* out_data = (float*)output->data;
    int valid = 1;
    for (size_t i = 0; i < output->size; i++) {
        if (isnan(out_data[i]) || isinf(out_data[i])) {
            valid = 0;
            break;
        }
    }

    if (!valid) {
        printf("FAIL: output contains NaN or Inf\n");
        lenet5_free(model);
        tensor_free(input);
        tensor_free(output);
        return 1;
    }

    printf("PASS\n");
    lenet5_free(model);
    tensor_free(input);
    tensor_free(output);
    return 0;
}

static int test_lenet5_predict(void) {
    printf("=== test_lenet5_predict ===\n");

    LeNet5* model = lenet5_create();

    // Create input
    size_t shape[] = {4, 1, 32, 32};
    Tensor* input = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 4);
    float* data = (float*)input->data;
    for (size_t i = 0; i < input->size; i++) data[i] = (float)(rand() % 100) / 100.0f;

    // Predict (includes softmax)
    Tensor* pred = lenet5_predict(model, input);

    if (!pred) {
        printf("FAIL: prediction is NULL\n");
        lenet5_free(model);
        tensor_free(input);
        return 1;
    }

    // Check probabilities sum to ~1
    float* pred_data = (float*)pred->data;
    for (size_t n = 0; n < 4; n++) {
        float sum = 0.0f;
        for (size_t c = 0; c < 10; c++) {
            sum += pred_data[n * 10 + c];
        }
        if (!nearly_equal(sum, 1.0f, 0.01f)) {
            printf("FAIL: probabilities don't sum to 1 (sum=%.4f)\n", sum);
            lenet5_free(model);
            tensor_free(input);
            tensor_free(pred);
            return 1;
        }
    }

    printf("PASS\n");
    lenet5_free(model);
    tensor_free(input);
    tensor_free(pred);
    return 0;
}

static int test_lenet5_train_step(void) {
    printf("=== test_lenet5_train_step ===\n");

    LeNet5* model = lenet5_create();

    // Create input and targets
    size_t shape[] = {2, 1, 32, 32};
    Tensor* input = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 4);
    float* in_data = (float*)input->data;
    for (size_t i = 0; i < input->size; i++) in_data[i] = (float)(rand() % 100) / 100.0f;

    size_t target_shape[] = {2, 10};
    Tensor* targets = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, target_shape, 2);
    float* target_data = (float*)targets->data;
    for (size_t i = 0; i < 20; i++) target_data[i] = 0.0f;
    target_data[0] = 1.0f; target_data[10 + 5] = 1.0f; // One-hot targets

    // Training step
    float loss = lenet5_train_step(model, 0.01f, input, targets);

    if (isnan(loss) || isinf(loss)) {
        printf("FAIL: loss is NaN or Inf\n");
        lenet5_free(model);
        tensor_free(input);
        tensor_free(targets);
        return 1;
    }

    printf("Loss: %.4f\n", loss);
    printf("PASS\n");
    lenet5_free(model);
    tensor_free(input);
    tensor_free(targets);
    return 0;
}

int main(void) {
    printf("\n=== LeNet-5 Tests ===\n\n");
    srand((unsigned int)time(NULL));

    int failures = 0;
    failures += test_lenet5_create();
    failures += test_lenet5_forward();
    failures += test_lenet5_predict();
    failures += test_lenet5_train_step();

    printf("\n=== Summary: %d failures ===\n", failures);
    return failures ? 1 : 0;
}
