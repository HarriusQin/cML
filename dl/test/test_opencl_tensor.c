#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "opencl_tensor.h"

static bool float_equal(float a, float b, float eps) {
    return fabsf(a - b) < eps;
}

static int test_cl_init(void) {
    printf("=== Test: OpenCL Init ===\n");
    CLOpenCL cl;
    memset(&cl, 0, sizeof(cl));

    if (!cl_init(&cl, CL_DEVICE_TYPE_GPU)) {
        printf("  FAILED: Cannot initialize OpenCL (no GPU?)\n");
        printf("  Falling back to CPU...\n");
        if (!cl_init(&cl, CL_DEVICE_TYPE_CPU)) {
            printf("  FAILED: Cannot initialize OpenCL\n");
            return 1;
        }
    }

    cl_print_device_info(cl.device);
    cl_release(&cl);
    printf("  PASSED\n");
    return 0;
}

static int test_tensor_create(CLOpenCL* cl) {
    printf("=== Test: Tensor Create ===\n");

    size_t shape2d[] = {3, 4};
    CLTensor* t = cl_tensor_create(cl, CL_TENSOR_DTYPE_F32, CL_TENSOR_LAYOUT_NCHW, shape2d, 2);
    if (!t) {
        printf("  FAILED: Cannot create tensor\n");
        return 1;
    }

    if (t->ndim != 2 || t->shape[0] != 3 || t->shape[1] != 4 || t->size != 12) {
        printf("  FAILED: Wrong tensor metadata\n");
        cl_tensor_free(t);
        return 1;
    }

    cl_tensor_print(t, "Created");
    cl_tensor_free(t);
    printf("  PASSED\n");
    return 0;
}

static int test_tensor_upload_download(CLOpenCL* cl) {
    printf("=== Test: Upload/Download ===\n");

    size_t shape[] = {2, 3};
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float out[6] = {0};

    CLTensor* t = cl_tensor_create_from_host(cl, CL_TENSOR_DTYPE_F32,
                                              CL_TENSOR_LAYOUT_NCHW, shape, 2, data);
    if (!t) {
        printf("  FAILED: Cannot create tensor from host\n");
        return 1;
    }

    cl_tensor_download(t, out);

    bool ok = true;
    for (int i = 0; i < 6; i++) {
        if (!float_equal(data[i], out[i], 1e-5f)) {
            printf("  FAILED: Mismatch at index %d: expected %.4f, got %.4f\n",
                   i, data[i], out[i]);
            ok = false;
        }
    }

    cl_tensor_free(t);

    if (ok) {
        printf("  PASSED\n");
        return 0;
    }
    return 1;
}

static int test_relu(CLOpenCL* cl) {
    printf("=== Test: ReLU ===\n");

    size_t shape[] = {2, 4};
    float data[] = {-1.0f, 2.0f, -3.0f, 4.0f, 5.0f, -6.0f, 7.0f, -8.0f};
    float expected[] = {0.0f, 2.0f, 0.0f, 4.0f, 5.0f, 0.0f, 7.0f, 0.0f};

    CLTensor* input = cl_tensor_create_from_host(cl, CL_TENSOR_DTYPE_F32,
                                                  CL_TENSOR_LAYOUT_NCHW, shape, 2, data);
    if (!input) {
        printf("  FAILED: Cannot create input tensor\n");
        return 1;
    }

    CLTensor* output = cl_tensor_relu(cl, input);
    if (!output) {
        printf("  FAILED: ReLU returned NULL\n");
        cl_tensor_free(input);
        return 1;
    }

    float* out_data = (float*)malloc(output->nbytes);
    cl_tensor_download(output, out_data);

    bool ok = true;
    for (int i = 0; i < 8; i++) {
        if (!float_equal(expected[i], out_data[i], 1e-5f)) {
            printf("  FAILED: Mismatch at index %d: expected %.4f, got %.4f\n",
                   i, expected[i], out_data[i]);
            ok = false;
        }
    }

    free(out_data);
    cl_tensor_free(input);
    cl_tensor_free(output);

    if (ok) {
        printf("  PASSED\n");
        return 0;
    }
    return 1;
}

static int test_matmul(CLOpenCL* cl) {
    printf("=== Test: Matrix Multiplication ===\n");

    // A [2x3] @ B [3x2] = C [2x2]
    size_t shape_a[] = {2, 3};
    size_t shape_b[] = {3, 2};
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float b_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    // Expected: C = [[22, 28], [49, 64]]
    float expected[] = {22.0f, 28.0f, 49.0f, 64.0f};

    CLTensor* A = cl_tensor_create_from_host(cl, CL_TENSOR_DTYPE_F32,
                                              CL_TENSOR_LAYOUT_NCHW, shape_a, 2, a_data);
    CLTensor* B = cl_tensor_create_from_host(cl, CL_TENSOR_DTYPE_F32,
                                              CL_TENSOR_LAYOUT_NCHW, shape_b, 2, b_data);
    if (!A || !B) {
        printf("  FAILED: Cannot create input tensors\n");
        if (A) cl_tensor_free(A);
        if (B) cl_tensor_free(B);
        return 1;
    }

    CLTensor* C = cl_tensor_matmul(cl, A, B);
    if (!C) {
        printf("  FAILED: matmul returned NULL\n");
        cl_tensor_free(A);
        cl_tensor_free(B);
        return 1;
    }

    float* c_data = (float*)malloc(C->nbytes);
    cl_tensor_download(C, c_data);

    bool ok = true;
    for (int i = 0; i < 4; i++) {
        if (!float_equal(expected[i], c_data[i], 1e-3f)) {
            printf("  FAILED: Mismatch at index %d: expected %.4f, got %.4f\n",
                   i, expected[i], c_data[i]);
            ok = false;
        }
    }

    free(c_data);
    cl_tensor_free(A);
    cl_tensor_free(B);
    cl_tensor_free(C);

    if (ok) {
        printf("  PASSED\n");
        return 0;
    }
    return 1;
}

static int test_softmax(CLOpenCL* cl) {
    printf("=== Test: Softmax ===\n");

    // Input: [1, 3] -> softmax over axis=1
    size_t shape[] = {1, 3};
    float data[] = {1.0f, 2.0f, 3.0f};

    // softmax([1,2,3]) = [0.090030, 0.244728, 0.665241]
    float expected[] = {0.090030f, 0.244728f, 0.665241f};

    CLTensor* input = cl_tensor_create_from_host(cl, CL_TENSOR_DTYPE_F32,
                                                   CL_TENSOR_LAYOUT_NCHW, shape, 2, data);
    if (!input) {
        printf("  FAILED: Cannot create input tensor\n");
        return 1;
    }

    CLTensor* output = cl_tensor_softmax(cl, input, 1);
    if (!output) {
        printf("  FAILED: softmax returned NULL\n");
        cl_tensor_free(input);
        return 1;
    }

    float* out_data = (float*)malloc(output->nbytes);
    cl_tensor_download(output, out_data);

    bool ok = true;
    for (int i = 0; i < 3; i++) {
        if (!float_equal(expected[i], out_data[i], 1e-4f)) {
            printf("  FAILED: Mismatch at index %d: expected %.6f, got %.6f\n",
                   i, expected[i], out_data[i]);
            ok = false;
        }
    }

    free(out_data);
    cl_tensor_free(input);
    cl_tensor_free(output);

    if (ok) {
        printf("  PASSED\n");
        return 0;
    }
    return 1;
}

static int test_fill(CLOpenCL* cl) {
    printf("=== Test: Fill ===\n");

    size_t shape[] = {10, 10};
    CLTensor* t = cl_tensor_create(cl, CL_TENSOR_DTYPE_F32, CL_TENSOR_LAYOUT_NCHW, shape, 2);
    if (!t) {
        printf("  FAILED: Cannot create tensor\n");
        return 1;
    }

    cl_tensor_fill_f32(cl, t, 3.14f);

    float* out = (float*)malloc(t->nbytes);
    cl_tensor_download(t, out);

    bool ok = true;
    for (size_t i = 0; i < t->size; i++) {
        if (!float_equal(3.14f, out[i], 1e-5f)) {
            printf("  FAILED: Mismatch at index %zu: expected 3.14, got %.4f\n", i, out[i]);
            ok = false;
            break;
        }
    }

    free(out);
    cl_tensor_free(t);

    if (ok) {
        printf("  PASSED\n");
        return 0;
    }
    return 1;
}

int main(void) {
    printf("OpenCL Tensor Test Suite\n");
    printf("========================\n\n");

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
    failed += test_cl_init();
    failed += test_tensor_create(&cl);
    failed += test_tensor_upload_download(&cl);
    failed += test_fill(&cl);
    failed += test_relu(&cl);
    failed += test_matmul(&cl);
    failed += test_softmax(&cl);

    cl_release(&cl);

    printf("\n========================\n");
    if (failed == 0) {
        printf("All tests PASSED!\n");
    } else {
        printf("%d test(s) FAILED\n", failed);
    }

    return failed;
}
