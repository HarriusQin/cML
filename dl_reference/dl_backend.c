/**
 * @file dl_backend.c
 * @brief CPU backend implementation
 */

#include "dl_backend.h"
#include "dl_tensor.h"
#include <string.h>

/* ============================================================================
 * CPU GEMM (Matrix Multiplication)
 * ============================================================================ */

/**
 * @brief General matrix multiplication (GEMM)
 *
 * Computes: C = alpha * op(A) * op(B) + beta * C
 * For simplicity, we use alpha=1, beta=0 (no in-place modification)
 */
static void cpu_gemm(double *C, const double *A, const double *B,
                     size_t m, size_t n, size_t k, bool transpose_a, bool transpose_b) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t p = 0; p < k; p++) {
                double a = transpose_a ? A[p * m + i] : A[i * k + p];
                double b = transpose_b ? B[j * k + p] : B[p * n + j];
                sum += a * b;
            }
            C[i * n + j] = sum;
        }
    }
}

/* ============================================================================
 * CPU MEMORY OPERATIONS
 * ============================================================================ */

static void *cpu_malloc(size_t size) {
    return malloc(size);
}

static void cpu_free(void *ptr) {
    free(ptr);
}

static void *cpu_malloc_pinned(size_t size) {
    return malloc(size);  /* CPU backend: no pinned memory concept */
}

static void cpu_copy(void *dst, const void *src, size_t size) {
    memcpy(dst, src, size);
}

static void cpu_to_device(Tensor *t) {
    /* CPU backend: data already on host, nothing to do */
}

static void cpu_to_host(Tensor *t) {
    /* CPU backend: data already on host, nothing to do */
}

static void cpu_synchronize(void) {
    /* CPU backend: always synchronized */
}

/* ============================================================================
 * CPU BACKEND INSTANCE
 * ============================================================================ */

DLBackend cpu_backend = {
    .name = "CPU",
    .device = DL_DEVICE_CPU,
    .malloc = cpu_malloc,
    .free = cpu_free,
    .malloc_pinned = cpu_malloc_pinned,
    .gemm = cpu_gemm,
    .copy = cpu_copy,
    .to_device = cpu_to_device,
    .to_host = cpu_to_host,
    .synchronize = cpu_synchronize
};

/* ============================================================================
 * BACKEND MANAGEMENT
 * ============================================================================ */

static DLBackend *default_backend = &cpu_backend;

DLBackend *dl_get_backend(DL_Device_t device) {
    switch (device) {
        case DL_DEVICE_CPU:
            return &cpu_backend;
        case DL_DEVICE_CUDA:
            DL_WARN("CUDA backend not available, falling back to CPU");
            return &cpu_backend;
        case DL_DEVICE_OPENCL:
            DL_WARN("OpenCL backend not available, falling back to CPU");
            return &cpu_backend;
        default:
            return &cpu_backend;
    }
}

void dl_set_default_backend(DLBackend *backend) {
    if (backend) {
        default_backend = backend;
    }
}

int dl_backend_init(DLBackend *backend) {
    if (!backend) return -1;
    return 0;  /* CPU backend needs no initialization */
}

void dl_backend_free(DLBackend *backend) {
    /* CPU backend has no resources to free */
}
