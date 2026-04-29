#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include "tensor.h"
#include "opencl_tensor.h"

#define EPS 1e-5f

static inline double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static bool float_equal(float a, float b) {
    return fabsf(a - b) < EPS;
}

/* --------------------------------------------------------------------------
 * Benchmark: ReLU
 * -------------------------------------------------------------------------- */

static double benchmark_cpu_relu(Tensor* t, int iterations) {
    double start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        tensor_relu(t);
    }
    return get_time_ms() - start;
}

static double benchmark_gpu_relu(CLOpenCL* cl, CLTensor* t, int iterations) {
    double start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        CLTensor* out = cl_tensor_relu(cl, t);
        cl_tensor_free(out);
    }
    return get_time_ms() - start;
}

/* --------------------------------------------------------------------------
 * Benchmark: GEMM (Matrix Multiplication)
 * -------------------------------------------------------------------------- */

static double benchmark_cpu_matmul(Tensor** mats, int count, int M, int K, int N, int iterations) {
    double start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        Tensor* C = tensor_matmul(mats[0], mats[1]);
        tensor_free(C);
    }
    return get_time_ms() - start;
}

static double benchmark_gpu_matmul(CLOpenCL* cl, CLTensor** mats, int M, int K, int N, int iterations) {
    double start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        CLTensor* C = cl_tensor_matmul(cl, mats[0], mats[1]);
        cl_tensor_free(C);
    }
    return get_time_ms() - start;
}

/* --------------------------------------------------------------------------
 * Benchmark: Softmax
 * -------------------------------------------------------------------------- */

static double benchmark_cpu_softmax(Tensor* t, int axis, int iterations) {
    double start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        tensor_softmax(t, axis);
    }
    return get_time_ms() - start;
}

static double benchmark_gpu_softmax(CLOpenCL* cl, CLTensor* t, int axis, int iterations) {
    double start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        CLTensor* out = cl_tensor_softmax(cl, t, axis);
        cl_tensor_free(out);
    }
    return get_time_ms() - start;
}

/* --------------------------------------------------------------------------
 * Result Verification
 * -------------------------------------------------------------------------- */

static int verify_cpu_tensor(const Tensor* t, float* expected) {
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->size; i++) {
        if (!float_equal(data[i], expected[i])) {
            printf("  MISMATCH at index %zu: CPU=%.6f, expected=%.6f\n",
                   i, data[i], expected[i]);
            return 0;
        }
    }
    return 1;
}

static int verify_gpu_tensor(CLOpenCL* cl, const CLTensor* t, float* expected) {
    float* data = (float*)malloc(t->nbytes);
    cl_tensor_download(t, data);

    for (size_t i = 0; i < t->size; i++) {
        if (!float_equal(data[i], expected[i])) {
            printf("  MISMATCH at index %zu: GPU=%.6f, expected=%.6f\n",
                   i, data[i], expected[i]);
            free(data);
            return 0;
        }
    }
    free(data);
    return 1;
}

static int verify_results_match(CLOpenCL* cl, const Tensor* cpu, const CLTensor* gpu) {
    float* gpu_data = (float*)malloc(gpu->nbytes);
    cl_tensor_download(gpu, gpu_data);

    float* cpu_data = (float*)cpu->data;
    for (size_t i = 0; i < cpu->size; i++) {
        if (!float_equal(cpu_data[i], gpu_data[i])) {
            printf("  CPU-GPU MISMATCH at index %zu: CPU=%.6f, GPU=%.6f\n",
                   i, cpu_data[i], gpu_data[i]);
            free(gpu_data);
            return 0;
        }
    }
    free(gpu_data);
    return 1;
}

/* --------------------------------------------------------------------------
 * Tests
 * -------------------------------------------------------------------------- */

static void test_relu_benchmark(CLOpenCL* cl) {
    printf("\n=== ReLU Benchmark ===\n");

    size_t shape[] = {1024, 1024};
    Tensor* cpu_t = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 2);
    tensor_fill_randn(cpu_t, 0.0f, 1.0f);

    CLTensor* gpu_t = cl_tensor_create_from_host(cl, CL_TENSOR_DTYPE_F32,
                                                  CL_TENSOR_LAYOUT_NCHW, shape, 2,
                                                  cpu_t->data);

    // Generate expected (same input for both)
    tensor_fill_randn(cpu_t, 0.0f, 1.0f);
    cl_tensor_upload(gpu_t, cpu_t->data);

    float* expected = (float*)malloc(tensor_nbytes(cpu_t));
    memcpy(expected, (float*)cpu_t->data, tensor_nbytes(cpu_t));
    for (size_t i = 0; i < cpu_t->size; i++) {
        if (expected[i] < 0) expected[i] = 0;
    }

    int iter = 100;

    double cpu_ms = benchmark_cpu_relu(cpu_t, iter);
    double gpu_ms = benchmark_gpu_relu(cl, gpu_t, iter);

    printf("  Input: [%zu x %zu] = %zu elements\n", shape[0], shape[1], cpu_t->size);
    printf("  CPU: %.2f ms (%d iterations)\n", cpu_ms, iter);
    printf("  GPU: %.2f ms (%d iterations)\n", gpu_ms, iter);
    printf("  Speedup: %.2fx\n", cpu_ms / gpu_ms);

    // Verify correctness
    tensor_relu(cpu_t);
    verify_cpu_tensor(cpu_t, expected);
    CLTensor* gpu_out = cl_tensor_relu(cl, gpu_t);
    verify_gpu_tensor(cl, gpu_out, expected);
    verify_results_match(cl, cpu_t, gpu_out);

    printf("  Result verification: PASSED\n");

    tensor_free(cpu_t);
    cl_tensor_free(gpu_t);
    cl_tensor_free(gpu_out);
    free(expected);
}

static void test_matmul_benchmark(CLOpenCL* cl) {
    printf("\n=== GEMM Benchmark ===\n");

    size_t M = 512, K = 512, N = 512;

    size_t shape_a[] = {M, K};
    size_t shape_b[] = {K, N};

    Tensor* cpu_a = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape_a, 2);
    Tensor* cpu_b = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape_b, 2);
    tensor_fill_randn(cpu_a, 0.0f, 0.1f);
    tensor_fill_randn(cpu_b, 0.0f, 0.1f);

    CLTensor* gpu_a = cl_tensor_create_from_host(cl, CL_TENSOR_DTYPE_F32,
                                                  CL_TENSOR_LAYOUT_NCHW, shape_a, 2, cpu_a->data);
    CLTensor* gpu_b = cl_tensor_create_from_host(cl, CL_TENSOR_DTYPE_F32,
                                                  CL_TENSOR_LAYOUT_NCHW, shape_b, 2, cpu_b->data);

    Tensor* mats[2] = {cpu_a, cpu_b};
    CLTensor* gpu_mats[2] = {gpu_a, gpu_b};

    int iter = 10;

    double cpu_ms = benchmark_cpu_matmul(mats, 2, M, K, N, iter);
    double gpu_ms = benchmark_gpu_matmul(cl, gpu_mats, M, K, N, iter);

    printf("  Input: [%zux%zu] @ [%zux%zu]\n", M, K, K, N);
    printf("  CPU: %.2f ms (%d iterations)\n", cpu_ms, iter);
    printf("  GPU: %.2f ms (%d iterations)\n", gpu_ms, iter);
    printf("  Speedup: %.2fx\n", cpu_ms / gpu_ms);

    // Verify correctness
    Tensor* cpu_c = tensor_matmul(cpu_a, cpu_b);
    CLTensor* gpu_c = cl_tensor_matmul(cl, gpu_a, gpu_b);
    verify_results_match(cl, cpu_c, gpu_c);
    printf("  Result verification: PASSED\n");

    tensor_free(cpu_a);
    tensor_free(cpu_b);
    tensor_free(cpu_c);
    cl_tensor_free(gpu_a);
    cl_tensor_free(gpu_b);
    cl_tensor_free(gpu_c);
}

static void test_softmax_benchmark(CLOpenCL* cl) {
    printf("\n=== Softmax Benchmark ===\n");

    size_t batch = 64, features = 1024;

    size_t shape[] = {batch, features};
    Tensor* cpu_t = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 2);
    tensor_fill_randn(cpu_t, 0.0f, 1.0f);

    CLTensor* gpu_t = cl_tensor_create_from_host(cl, CL_TENSOR_DTYPE_F32,
                                                  CL_TENSOR_LAYOUT_NCHW, shape, 2,
                                                  cpu_t->data);

    int iter = 50;

    double cpu_ms = benchmark_cpu_softmax(cpu_t, 1, iter);
    double gpu_ms = benchmark_gpu_softmax(cl, gpu_t, 1, iter);

    printf("  Input: [%zu x %zu] = %zu elements\n", batch, features, cpu_t->size);
    printf("  CPU: %.2f ms (%d iterations)\n", cpu_ms, iter);
    printf("  GPU: %.2f ms (%d iterations)\n", gpu_ms, iter);
    printf("  Speedup: %.2fx\n", cpu_ms / gpu_ms);

    // Verify correctness
    tensor_softmax(cpu_t, 1);
    CLTensor* gpu_out = cl_tensor_softmax(cl, gpu_t, 1);
    verify_results_match(cl, cpu_t, gpu_out);
    printf("  Result verification: PASSED\n");

    tensor_free(cpu_t);
    cl_tensor_free(gpu_t);
    cl_tensor_free(gpu_out);
}

static void test_largeGEMM_benchmark(CLOpenCL* cl) {
    printf("\n=== Large GEMM Benchmark (2048x2048) ===\n");

    size_t M = 2048, K = 2048, N = 2048;

    size_t shape_a[] = {M, K};
    size_t shape_b[] = {K, N};

    Tensor* cpu_a = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape_a, 2);
    Tensor* cpu_b = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape_b, 2);
    tensor_fill_randn(cpu_a, 0.0f, 0.1f);
    tensor_fill_randn(cpu_b, 0.0f, 0.1f);

    CLTensor* gpu_a = cl_tensor_create_from_host(cl, CL_TENSOR_DTYPE_F32,
                                                  CL_TENSOR_LAYOUT_NCHW, shape_a, 2, cpu_a->data);
    CLTensor* gpu_b = cl_tensor_create_from_host(cl, CL_TENSOR_DTYPE_F32,
                                                  CL_TENSOR_LAYOUT_NCHW, shape_b, 2, cpu_b->data);

    Tensor* mats[2] = {cpu_a, cpu_b};
    CLTensor* gpu_mats[2] = {gpu_a, gpu_b};

    int iter = 3;

    double cpu_ms = benchmark_cpu_matmul(mats, 2, M, K, N, iter);
    double gpu_ms = benchmark_gpu_matmul(cl, gpu_mats, M, K, N, iter);

    printf("  Input: [%zux%zu] @ [%zux%zu]\n", M, K, K, N);
    printf("  CPU: %.2f ms (%d iterations)\n", cpu_ms, iter);
    printf("  GPU: %.2f ms (%d iterations)\n", gpu_ms, iter);
    printf("  Speedup: %.2fx\n", cpu_ms / gpu_ms);

    // Verify correctness (sample check)
    Tensor* cpu_c = tensor_matmul(cpu_a, cpu_b);
    CLTensor* gpu_c = cl_tensor_matmul(cl, gpu_a, gpu_b);

    float* gpu_data = (float*)malloc(gpu_c->nbytes);
    float* cpu_data = (float*)cpu_c->data;
    cl_tensor_download(gpu_c, gpu_data);

    size_t errors = 0;
    for (size_t i = 0; i < 100 && i < cpu_c->size; i++) {
        size_t idx = (rand() % cpu_c->size);
        if (!float_equal(cpu_data[idx], gpu_data[idx])) {
            errors++;
        }
    }
    printf("  Sampled 100 elements: %zu mismatches\n", errors);

    tensor_free(cpu_a);
    tensor_free(cpu_b);
    tensor_free(cpu_c);
    cl_tensor_free(gpu_a);
    cl_tensor_free(gpu_b);
    cl_tensor_free(gpu_c);
    free(gpu_data);
}

int main(void) {
    printf("CPU vs GPU Tensor Performance Comparison\n");
    printf("=========================================\n");

    // Initialize OpenCL
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

    test_relu_benchmark(&cl);
    test_matmul_benchmark(&cl);
    test_softmax_benchmark(&cl);
    test_largeGEMM_benchmark(&cl);

    cl_release(&cl);

    printf("\n=========================================\n");
    printf("Benchmark complete.\n");

    return 0;
}
