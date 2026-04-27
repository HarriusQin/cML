/**
 * @file dl_backend.h
 * @brief Backend abstraction for device-specific operations
 *
 * Provides abstraction for different computing backends (CPU, CUDA, OpenCL).
 * The CPU backend is fully implemented. CUDA/OpenCL are planned for future.
 */

#ifndef __DL_BACKEND_H__
#define __DL_BACKEND_H__

#include "dl_base.h"

/* ============================================================================
 * DEVICE BACKEND
 * ============================================================================ */

/**
 * @brief Backend operation function pointers
 */
typedef struct DLBackend {
    const char *name;
    DL_Device_t device;

    /* Memory management */
    void *(*malloc)(size_t size);
    void (*free)(void *ptr);
    void *(*malloc_pinned)(size_t size);  /**< Pinned memory for faster GPU transfers */

    /* Tensor operations */
    void (*gemm)(double *C, const double *A, const double *B,
                 size_t m, size_t n, size_t k, bool transpose_a, bool transpose_b);

    /* Data transfer */
    void (*copy)(void *dst, const void *src, size_t size);
    void (*to_device)(Tensor *t);
    void (*to_host)(Tensor *t);

    /* Synchronization (no-op for CPU) */
    void (*synchronize)(void);
} DLBackend;

/* ============================================================================
 * CPU BACKEND
 * ============================================================================ */

extern DLBackend cpu_backend;

/* ============================================================================
 * BACKEND MANAGEMENT
 * ============================================================================ */

/**
 * @brief Get backend for device type
 */
DLBackend *dl_get_backend(DL_Device_t device);

/**
 * @brief Set default backend
 */
void dl_set_default_backend(DLBackend *backend);

/**
 * @brief Initialize backend (call before use)
 */
int dl_backend_init(DLBackend *backend);

/**
 * @brief Cleanup backend resources
 */
void dl_backend_free(DLBackend *backend);

/* ============================================================================
 * MEMORY HELPER FUNCTIONS
 * ============================================================================ */

/**
 * @brief Allocate zero-initialized memory on backend
 */
static inline void *dl_alloc(DLBackend *backend, size_t size) {
    return backend->malloc(size);
}

/**
 * @brief Free memory on backend
 */
static inline void dl_free(DLBackend *backend, void *ptr) {
    if (ptr) backend->free(ptr);
}

/**
 * @brief Copy data between hosts (CPU to CPU)
 */
static inline void dl_memcpy(void *dst, const void *src, size_t size) {
    cpu_backend.copy(dst, src, size);
}

/**
 * @brief Synchronize backend
 */
static inline void dl_sync(void) {
    cpu_backend.synchronize();
}

#endif /* __DL_BACKEND_H__ */
