#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../opencl_mlp.h"

/* ============================================================================
 * Simple Synthetic Data Generation for Testing
 * ============================================================================ */

static void generate_xor_data(float* X, float* y, size_t n_samples) {
    srand(42);
    for (size_t i = 0; i < n_samples; i++) {
        float x1 = (rand() % 2) ? 1.0f : 0.0f;
        float x2 = (rand() % 2) ? 1.0f : 0.0f;
        X[i * 2] = x1;
        X[i * 2 + 1] = x2;
        // XOR: 1 if exactly one input is 1
        y[i] = (x1 != x2) ? 1.0f : 0.0f;
    }
}

static int test_xor(CLOpenCL* cl) {
    printf("=== Test: XOR Classification ===\n");

    size_t input_dim = 2;
    size_t hidden_dim = 8;
    size_t output_dim = 2;  // binary classification
    size_t num_layers = 2;
    size_t n_samples = 100;
    size_t epochs = 500;
    float lr = 0.5f;

    // Generate XOR data
    float* h_X = (float*)malloc(n_samples * input_dim * sizeof(float));
    float* h_y = (float*)malloc(n_samples * sizeof(float));
    generate_xor_data(h_X, h_y, n_samples);

    // Convert to CLTensor
    size_t X_shape[] = {n_samples, input_dim};
    size_t y_shape[] = {n_samples};
    CLTensor* X = cl_tensor_create_from_host(cl, CL_TENSOR_DTYPE_F32,
                                               CL_TENSOR_LAYOUT_NCHW, X_shape, 2, h_X);
    CLTensor* y = cl_tensor_create_from_host(cl, CL_TENSOR_DTYPE_F32,
                                               CL_TENSOR_LAYOUT_NCHW, y_shape, 1, h_y);

    // Create MLP
    CLOpenCLMLP* mlp = cl_mlp_create(cl, input_dim, hidden_dim, output_dim, num_layers);
    CLSGDOptimizer* opt = cl_sgd_create(cl, mlp, lr, 0.0f, 0.0f);

    printf("  Training XOR problem...\n");
    for (size_t epoch = 0; epoch < epochs; epoch++) {
        float loss = cl_mlp_train_step(cl, mlp, opt, X, y);
        if (epoch % 100 == 0) {
            float acc = cl_mlp_accuracy(cl, mlp, X, y);
            printf("  Epoch %zu: loss=%.4f, acc=%.2f%%\n", epoch, loss, acc * 100);
        }
    }

    float final_acc = cl_mlp_accuracy(cl, mlp, X, y);
    printf("  Final accuracy: %.2f%%\n", final_acc * 100);

    // Test predictions on specific cases
    printf("  Testing specific cases:\n");
    float test_cases[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    int expected[4] = {0, 1, 1, 0};

    for (int i = 0; i < 4; i++) {
        size_t tc_shape[] = {1, 2};
        CLTensor* tc = cl_tensor_create_from_host(cl, CL_TENSOR_DTYPE_F32,
                                                    CL_TENSOR_LAYOUT_NCHW, tc_shape, 2, test_cases[i]);
        CLTensor* pred = cl_mlp_predict(cl, mlp, tc);

        float* h_pred = (float*)malloc(pred->nbytes);
        cl_tensor_download(pred, h_pred);

        int pred_class = (h_pred[1] > h_pred[0]) ? 1 : 0;
        printf("    (%d, %d) -> %d (expected %d) %s\n",
               (int)test_cases[i][0], (int)test_cases[i][1],
               pred_class, expected[i],
               pred_class == expected[i] ? "✓" : "✗");

        free(h_pred);
        cl_tensor_free(pred);
        cl_tensor_free(tc);
    }

    cl_mlp_free(mlp);
    cl_sgd_free(opt);
    cl_tensor_free(X);
    cl_tensor_free(y);
    free(h_X);
    free(h_y);

    if (final_acc > 0.9f) {
        printf("  PASSED\n");
        return 0;
    } else {
        printf("  FAILED (accuracy too low)\n");
        return 1;
    }
}

static int test_simple_classification(CLOpenCL* cl) {
    printf("=== Test: Simple 2-Class Classification ===\n");

    size_t input_dim = 2;
    size_t hidden_dim = 8;
    size_t output_dim = 2;
    size_t num_layers = 2;
    size_t n_samples = 80;
    size_t epochs = 400;
    float lr = 0.5f;

    // Generate well-separated data: two clusters at (0,0) and (5,5)
    float* h_X = (float*)malloc(n_samples * input_dim * sizeof(float));
    float* h_y = (float*)malloc(n_samples * sizeof(float));

    srand(42);
    for (size_t i = 0; i < n_samples; i++) {
        int cls = (i < n_samples / 2) ? 0 : 1;
        h_y[i] = (float)cls;
        if (cls == 0) {
            h_X[i * 2] = (rand() % 100) / 50.0f;       // 0-2
            h_X[i * 2 + 1] = (rand() % 100) / 50.0f;   // 0-2
        } else {
            h_X[i * 2] = 3.0f + (rand() % 100) / 50.0f;  // 3-5
            h_X[i * 2 + 1] = 3.0f + (rand() % 100) / 50.0f;
        }
    }

    size_t X_shape[] = {n_samples, input_dim};
    size_t y_shape[] = {n_samples};
    CLTensor* X = cl_tensor_create_from_host(cl, CL_TENSOR_DTYPE_F32,
                                               CL_TENSOR_LAYOUT_NCHW, X_shape, 2, h_X);
    CLTensor* y = cl_tensor_create_from_host(cl, CL_TENSOR_DTYPE_F32,
                                               CL_TENSOR_LAYOUT_NCHW, y_shape, 1, h_y);

    CLOpenCLMLP* mlp = cl_mlp_create(cl, input_dim, hidden_dim, output_dim, num_layers);
    CLSGDOptimizer* opt = cl_sgd_create(cl, mlp, lr, 0.0f, 0.0f);

    printf("  Training...\n");
    for (size_t epoch = 0; epoch < epochs; epoch++) {
        float loss = cl_mlp_train_step(cl, mlp, opt, X, y);
        if (epoch % 100 == 0) {
            float acc = cl_mlp_accuracy(cl, mlp, X, y);
            printf("  Epoch %zu: loss=%.4f, acc=%.2f%%\n", epoch, loss, acc * 100);
        }
    }

    float final_acc = cl_mlp_accuracy(cl, mlp, X, y);
    printf("  Final accuracy: %.2f%%\n", final_acc * 100);

    cl_mlp_free(mlp);
    cl_sgd_free(opt);
    cl_tensor_free(X);
    cl_tensor_free(y);
    free(h_X);
    free(h_y);

    if (final_acc > 0.9f) {
        printf("  PASSED\n");
        return 0;
    } else {
        printf("  FAILED\n");
        return 1;
    }
}

static int test_mlp_forward_only(CLOpenCL* cl) {
    printf("=== Test: Forward Pass Only ===\n");

    size_t input_dim = 4;
    size_t hidden_dim = 8;
    size_t output_dim = 3;

    CLOpenCLMLP* mlp = cl_mlp_create(cl, input_dim, hidden_dim, output_dim, 2);

    // Simple input
    float h_X[] = {1.0f, 2.0f, 3.0f, 4.0f};
    size_t shape[] = {1, 4};
    CLTensor* X = cl_tensor_create_from_host(cl, CL_TENSOR_DTYPE_F32,
                                              CL_TENSOR_LAYOUT_NCHW, shape, 2, h_X);

    CLTensor* output = cl_mlp_predict(cl, mlp, X);

    float* h_out = (float*)malloc(output->nbytes);
    cl_tensor_download(output, h_out);

    printf("  Input: [1.0, 2.0, 3.0, 4.0]\n");
    printf("  Output probabilities: [%.4f, %.4f, %.4f]\n",
           h_out[0], h_out[1], h_out[2]);

    // Check sum to 1 (softmax)
    float sum = h_out[0] + h_out[1] + h_out[2];
    printf("  Sum: %.4f %s\n", sum, fabsf(sum - 1.0f) < 0.01f ? "✓" : "✗");

    free(h_out);
    cl_tensor_free(output);
    cl_tensor_free(X);
    cl_mlp_free(mlp);

    if (fabsf(sum - 1.0f) < 0.01f) {
        printf("  PASSED\n");
        return 0;
    }
    printf("  FAILED\n");
    return 1;
}

static int test_layer_creation(CLOpenCL* cl) {
    printf("=== Test: Layer Creation ===\n");

    CLFCLayer* layer = cl_fc_layer_create(cl, 10, 5);
    if (!layer) {
        printf("  FAILED: Could not create layer\n");
        return 1;
    }

    if (layer->weight->shape[0] != 5 || layer->weight->shape[1] != 10) {
        printf("  FAILED: Wrong weight shape\n");
        cl_fc_layer_free(layer);
        return 1;
    }

    if (layer->bias->shape[0] != 5) {
        printf("  FAILED: Wrong bias shape\n");
        cl_fc_layer_free(layer);
        return 1;
    }

    printf("  Weight shape: [%zu, %zu]\n", layer->weight->shape[0], layer->weight->shape[1]);
    printf("  Bias shape: [%zu]\n", layer->bias->shape[0]);

    cl_fc_layer_free(layer);
    printf("  PASSED\n");
    return 0;
}

static int test_fc_forward(CLOpenCL* cl) {
    printf("=== Test: FC Layer Forward ===\n");

    CLFCLayer* layer = cl_fc_layer_create(cl, 3, 2);

    // Set weights to identity-like values
    float* h_w = (float*)malloc(layer->weight->nbytes);
    h_w[0] = 1.0f; h_w[1] = 0.0f; h_w[2] = 0.0f;
    h_w[3] = 0.0f; h_w[4] = 1.0f; h_w[5] = 0.0f;
    cl_tensor_upload(layer->weight, h_w);

    float* h_b = (float*)malloc(layer->bias->nbytes);
    h_b[0] = 0.0f; h_b[1] = 0.0f;
    cl_tensor_upload(layer->bias, h_b);

    // Input [1, 2, 3] -> output should be [1, 2] (with identity weights)
    float h_x[] = {1.0f, 2.0f, 3.0f};
    size_t shape[] = {1, 3};
    CLTensor* X = cl_tensor_create_from_host(cl, CL_TENSOR_DTYPE_F32,
                                              CL_TENSOR_LAYOUT_NCHW, shape, 2, h_x);

    CLTensor* output = cl_fc_layer_forward(cl, layer, X);

    float* h_out = (float*)malloc(output->nbytes);
    cl_tensor_download(output, h_out);

    printf("  Input: [1.0, 2.0, 3.0]\n");
    printf("  Output: [%.4f, %.4f]\n", h_out[0], h_out[1]);

    int passed = (fabsf(h_out[0] - 1.0f) < 0.01f && fabsf(h_out[1] - 2.0f) < 0.01f);
    printf("  Expected: [1.0, 2.0] %s\n", passed ? "✓" : "✗");

    free(h_out);
    free(h_w);
    free(h_b);
    cl_tensor_free(output);
    cl_tensor_free(X);
    cl_fc_layer_free(layer);

    if (passed) {
        printf("  PASSED\n");
        return 0;
    }
    printf("  FAILED\n");
    return 1;
}

static int test_relu(CLOpenCL* cl) {
    printf("=== Test: ReLU ===\n");

    float h_x[] = {-1.0f, 2.0f, -3.0f, 4.0f, 5.0f, -6.0f};
    float expected[] = {0.0f, 2.0f, 0.0f, 4.0f, 5.0f, 0.0f};
    size_t shape[] = {2, 3};

    CLTensor* X = cl_tensor_create_from_host(cl, CL_TENSOR_DTYPE_F32,
                                              CL_TENSOR_LAYOUT_NCHW, shape, 2, h_x);

    CLTensor* mask = NULL;
    CLTensor* result = cl_relu_forward(cl, X, &mask);

    float* h_out = (float*)malloc(result->nbytes);
    cl_tensor_download(result, h_out);

    int passed = 1;
    for (size_t i = 0; i < 6; i++) {
        if (fabsf(h_out[i] - expected[i]) > 0.001f) {
            passed = 0;
            printf("  Mismatch at %zu: got %.4f, expected %.4f\n", i, h_out[i], expected[i]);
        }
    }

    printf("  Input:  [-1, 2, -3, 4, 5, -6]\n");
    printf("  Output: [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f]\n",
           h_out[0], h_out[1], h_out[2], h_out[3], h_out[4], h_out[5]);

    free(h_out);
    // result == X (cl_relu_forward modifies in place and returns same pointer)
    cl_tensor_free(result);
    if (mask) cl_tensor_free(mask);
    // Don't free X again - it's the same as result

    if (passed) {
        printf("  PASSED\n");
        return 0;
    }
    printf("  FAILED\n");
    return 1;
}

int main(void) {
    printf("OpenCL MLP Test Suite\n");
    printf("====================\n\n");

    CLOpenCL cl;
    memset(&cl, 0, sizeof(cl));

    if (!cl_init(&cl, CL_DEVICE_TYPE_GPU)) {
        printf("No GPU found, trying CPU...\n");
        if (!cl_init(&cl, CL_DEVICE_TYPE_CPU)) {
            printf("FAILED: Cannot initialize OpenCL\n");
            return 1;
        }
    }
    cl_print_device_info(cl.device);
    printf("\n");

    int failed = 0;

    failed += test_layer_creation(&cl);
    failed += test_fc_forward(&cl);
    failed += test_relu(&cl);
    failed += test_mlp_forward_only(&cl);
    failed += test_xor(&cl);
    failed += test_simple_classification(&cl);

    cl_release(&cl);

    printf("\n====================\n");
    if (failed == 0) {
        printf("All tests PASSED!\n");
    } else {
        printf("%d test(s) FAILED\n", failed);
    }

    return failed;
}