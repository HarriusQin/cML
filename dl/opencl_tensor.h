#ifndef __OPENCL_TENSOR_H__
#define __OPENCL_TENSOR_H__

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

/* ============================================================================
 * OpenCL Tensor Data Structure
 * GPU-resident tensor with host-side metadata
 * ============================================================================ */

typedef enum {
    CL_TENSOR_LAYOUT_NCHW,
    CL_TENSOR_LAYOUT_NHWC,
    CL_TENSOR_LAYOUT_CHWN,
} CLTensorLayout;

typedef enum {
    CL_TENSOR_DTYPE_F32,
    CL_TENSOR_DTYPE_F16,
    CL_TENSOR_DTYPE_INT8,
    CL_TENSOR_DTYPE_INT32,
} CLTensorDType;

typedef struct {
    cl_mem buffer;
    cl_context context;
    cl_command_queue queue;
    CLTensorLayout layout;
    CLTensorDType dtype;
    uint8_t ndim;
    size_t* shape;
    size_t* strides;
    size_t size;
    size_t nbytes;
    cl_uint dev_id;
} CLTensor;

/* ============================================================================
 * Kernel Cache
 * Pre-compiles all OpenCL C source strings once, reuses cached programs/kernels
 * ============================================================================ */

typedef struct {
    cl_program fill_prog, relu_prog, gemm_prog;
    cl_program elem_mul_prog, bias_add_prog, softmax_prog;
    cl_program cross_entropy_grad_prog, sgd_update_prog, clip_prog, copy_prog;
    cl_kernel fill_k, relu_k, gemm_k;
    cl_kernel elem_mul_k, bias_add_k, softmax_k;
    cl_kernel cross_entropy_grad_k, sgd_update_k, clip_k, copy_k;
    bool compiled;
} cl_kernel_cache_t;

/* ============================================================================
 * OpenCL Context & Queue Management
 * ============================================================================ */

typedef struct {
    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
    cl_uint dev_count;
    cl_platform_id platform;
    cl_kernel_cache_t kernel_cache;
} CLOpenCL;

static inline const char* cl_get_error_string(cl_int err) {
    switch (err) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        default: return "Unknown";
    }
}

#define CL_CHECK(err, msg) \
    do { \
        if (err != CL_SUCCESS) { \
            fprintf(stderr, "OpenCL error %d: %s at %s:%d - %s\n", \
                    err, cl_get_error_string(err), __FILE__, __LINE__, msg); \
            exit(1); \
        } \
    } while (0)

/* --------------------------------------------------------------------------
 * Platform & Device Discovery
 * -------------------------------------------------------------------------- */

static bool cl_init(CLOpenCL* cl, cl_device_type device_type) {
    cl_int err;

    err = clGetPlatformIDs(1, &cl->platform, NULL);
    if (err != CL_SUCCESS) return false;

    err = clGetDeviceIDs(cl->platform, device_type, 1, &cl->device, &cl->dev_count);
    if (err != CL_SUCCESS) return false;

    cl->context = clCreateContext(NULL, 1, &cl->device, NULL, NULL, &err);
    if (err != CL_SUCCESS) return false;

    cl->queue = clCreateCommandQueue(cl->context, cl->device, 0, &err);
    if (err != CL_SUCCESS) return false;

    return true;
}

static void cl_release(CLOpenCL* cl) {
    if (cl->queue) clReleaseCommandQueue(cl->queue);
    if (cl->context) clReleaseContext(cl->context);
}

static void cl_print_device_info(cl_device_id device) {
    char name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
    printf("OpenCL Device: %s\n", name);

    cl_ulong global_mem, local_mem;
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, NULL);
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem), &local_mem, NULL);
    printf("  Global Memory: %.2f GB\n", global_mem / (1024.0 * 1024.0 * 1024.0));
    printf("  Local Memory: %.2f KB\n", local_mem / 1024.0);
}

/* ============================================================================
 * Kernel Cache
 * Pre-compiles all OpenCL C source strings once, reuses cached programs/kernels
 * ============================================================================ */

static void cl_release_cache(cl_kernel_cache_t* cache) {
    if (!cache) return;
    if (cache->fill_k) clReleaseKernel(cache->fill_k);
    if (cache->relu_k) clReleaseKernel(cache->relu_k);
    if (cache->gemm_k) clReleaseKernel(cache->gemm_k);
    if (cache->elem_mul_k) clReleaseKernel(cache->elem_mul_k);
    if (cache->bias_add_k) clReleaseKernel(cache->bias_add_k);
    if (cache->softmax_k) clReleaseKernel(cache->softmax_k);
    if (cache->cross_entropy_grad_k) clReleaseKernel(cache->cross_entropy_grad_k);
    if (cache->sgd_update_k) clReleaseKernel(cache->sgd_update_k);
    if (cache->clip_k) clReleaseKernel(cache->clip_k);
    if (cache->copy_k) clReleaseKernel(cache->copy_k);

    if (cache->fill_prog) clReleaseProgram(cache->fill_prog);
    if (cache->relu_prog) clReleaseProgram(cache->relu_prog);
    if (cache->gemm_prog) clReleaseProgram(cache->gemm_prog);
    if (cache->elem_mul_prog) clReleaseProgram(cache->elem_mul_prog);
    if (cache->bias_add_prog) clReleaseProgram(cache->bias_add_prog);
    if (cache->softmax_prog) clReleaseProgram(cache->softmax_prog);
    if (cache->cross_entropy_grad_prog) clReleaseProgram(cache->cross_entropy_grad_prog);
    if (cache->sgd_update_prog) clReleaseProgram(cache->sgd_update_prog);
    if (cache->clip_prog) clReleaseProgram(cache->clip_prog);
    if (cache->copy_prog) clReleaseProgram(cache->copy_prog);

    memset(cache, 0, sizeof(cl_kernel_cache_t));
}

/* ============================================================================
 * CLTensor Lifecycle
 * ============================================================================ */

static inline size_t cl_tensor_dtype_size(CLTensorDType dtype) {
    switch (dtype) {
        case CL_TENSOR_DTYPE_F32:
        case CL_TENSOR_DTYPE_INT32: return 4;
        case CL_TENSOR_DTYPE_F16:
        case CL_TENSOR_DTYPE_INT8:  return 1;
        default: return 4;
    }
}

static inline size_t cl_tensor_elem_count(const CLTensor* t) {
    size_t count = 1;
    for (uint8_t i = 0; i < t->ndim; i++) count *= t->shape[i];
    return count;
}

static void cl_tensor_compute_strides(CLTensor* t) {
    t->strides[t->ndim - 1] = 1;
    for (int i = (int)t->ndim - 2; i >= 0; i--) {
        t->strides[i] = t->strides[i + 1] * t->shape[i + 1];
    }
}

static CLTensor* cl_tensor_create(CLOpenCL* cl, CLTensorDType dtype,
                                  CLTensorLayout layout, const size_t* shape, uint8_t ndim) {
    CLTensor* t = (CLTensor*)malloc(sizeof(CLTensor));
    t->dtype = dtype;
    t->layout = layout;
    t->ndim = ndim;
    t->shape = (size_t*)malloc(sizeof(size_t) * ndim);
    t->strides = (size_t*)malloc(sizeof(size_t) * ndim);
    t->size = 1;

    for (uint8_t i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->size *= shape[i];
    }
    cl_tensor_compute_strides(t);

    t->nbytes = t->size * cl_tensor_dtype_size(dtype);
    t->context = cl->context;
    t->queue = cl->queue;
    t->dev_id = 0;

    cl_int err;
    t->buffer = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, t->nbytes, NULL, &err);
    if (err != CL_SUCCESS) {
        free(t->shape);
        free(t->strides);
        free(t);
        return NULL;
    }

    return t;
}

static CLTensor* cl_tensor_create_from_host(CLOpenCL* cl, CLTensorDType dtype,
                                            CLTensorLayout layout, const size_t* shape,
                                            uint8_t ndim, const void* host_data) {
    CLTensor* t = cl_tensor_create(cl, dtype, layout, shape, ndim);
    if (!t) return NULL;

    cl_int err = clEnqueueWriteBuffer(cl->queue, t->buffer, CL_TRUE, 0, t->nbytes,
                                       host_data, 0, NULL, NULL);
    CL_CHECK(err, "Failed to copy data to device");

    return t;
}

static void cl_tensor_free(CLTensor* t) {
    if (!t) return;
    if (t->buffer) clReleaseMemObject(t->buffer);
    free(t->shape);
    free(t->strides);
    free(t);
}

static void cl_tensor_upload(CLTensor* t, const void* host_data) {
    cl_int err = clEnqueueWriteBuffer(t->queue, t->buffer, CL_TRUE, 0, t->nbytes,
                                       host_data, 0, NULL, NULL);
    CL_CHECK(err, "Failed to upload to device");
}

static void cl_tensor_download(const CLTensor* t, void* host_data) {
    cl_int err = clEnqueueReadBuffer(t->queue, t->buffer, CL_TRUE, 0, t->nbytes,
                                      host_data, 0, NULL, NULL);
    CL_CHECK(err, "Failed to download from device");
}

/* ============================================================================
 * GPU Kernel Sources
 * ============================================================================ */

/* ---- fill ---- */
static const char* cl_kernel_fill_source =
"__kernel void fill_f32(__global float* output, float value, int size) {\n"
"    int gid = get_global_id(0);\n"
"    if (gid < size) {\n"
"        output[gid] = value;\n"
"    }\n"
"}\n";

/* ---- relu ---- */
static const char* cl_kernel_relu_source =
"__kernel void relu_f32(__global float* output, __global const float* input, int size) {\n"
"    int gid = get_global_id(0);\n"
"    if (gid < size) {\n"
"        float v = input[gid];\n"
"        output[gid] = v > 0 ? v : 0;\n"
"    }\n"
"}\n";

/* ---- gemm ---- */
static const char* cl_kernel_gemm_source =
"__kernel void gemm_f32(__global float* C,\n"
"                       __global const float* A, __global const float* B,\n"
"                       int M, int N, int K, float alpha, float beta) {\n"
"    int row = get_global_id(1);\n"
"    int col = get_global_id(0);\n"
"\n"
"    if (row < M && col < N) {\n"
"        float sum = 0.0f;\n"
"        for (int k = 0; k < K; k++) {\n"
"            sum += A[row * K + k] * B[k * N + col];\n"
"        }\n"
"        C[row * N + col] = alpha * sum + beta * C[row * N + col];\n"
"    }\n"
"}\n";

/* ---- element-wise multiply ---- */
static const char* cl_kernel_elem_mul_source =
"__kernel void elem_mul_f32(__global float* output,\n"
"                            __global const float* a,\n"
"                            __global const float* b, int size) {\n"
"    int gid = get_global_id(0);\n"
"    if (gid < size) {\n"
"        output[gid] = a[gid] * b[gid];\n"
"    }\n"
"}\n";

/* ---- bias add (broadcast bias over batch) ---- */
static const char* cl_kernel_bias_add_source =
"__kernel void bias_add_f32(__global float* output,\n"
"                            __global const float* input,\n"
"                            __global const float* bias, int batch, int features) {\n"
"    int gid = get_global_id(0);\n"
"    if (gid < (int)(batch * features)) {\n"
"        int b = gid / features;\n"
"        int f = gid % features;\n"
"        output[gid] = input[gid] + bias[f];\n"
"    }\n"
"}\n";

/* ---- softmax: computes softmax along last axis (num_classes) ---- */
static const char* cl_kernel_softmax_source =
"__kernel void softmax_f32(__global float* output,\n"
"                           __global const float* input,\n"
"                           int batch, int num_classes) {\n"
"    int b = get_global_id(0);\n"
"    if (b >= batch) return;\n"
"\n"
"    int offset = b * num_classes;\n"
"    // find max\n"
"    float max_val = input[offset];\n"
"    for (int c = 1; c < num_classes; c++) {\n"
"        float v = input[offset + c];\n"
"        if (v > max_val) max_val = v;\n"
"    }\n"
"    // exp and sum\n"
"    float sum = 0.0f;\n"
"    for (int c = 0; c < num_classes; c++) {\n"
"        float v = exp(input[offset + c] - max_val);\n"
"        output[offset + c] = v;\n"
"        sum += v;\n"
"    }\n"
"    // normalize\n"
"    for (int c = 0; c < num_classes; c++) {\n"
"        output[offset + c] = output[offset + c] / sum;\n"
"    }\n"
"}\n";

/* ---- cross-entropy gradient ---- */
static const char* cl_kernel_cross_entropy_grad_source =
"__kernel void cross_entropy_grad_f32(__global float* grad,\n"
"                                      __global const float* pred,\n"
"                                      __global const int* targets,\n"
"                                      int batch, int num_classes) {\n"
"    int b = get_global_id(0);\n"
"    int c = get_global_id(1);\n"
"    if (b >= batch || c >= num_classes) return;\n"
"\n"
"    int idx = b * num_classes + c;\n"
"    grad[idx] = pred[idx] - (c == targets[b] ? 1.0f : 0.0f);\n"
"}\n";

/* ---- sgd update ---- */
static const char* cl_kernel_sgd_update_source =
"__kernel void sgd_update_f32(__global float* params,\n"
"                              __global const float* grad,\n"
"                              __global float* velocity,\n"
"                              float lr, float momentum,\n"
"                              float weight_decay, int size) {\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= size) return;\n"
"\n"
"    float g = grad[gid];\n"
"    float p = params[gid];\n"
"    float v;\n"
"\n"
"    if (velocity != NULL) {\n"
"        v = velocity[gid];\n"
"        v = momentum * v - lr * g - weight_decay * p;\n"
"        velocity[gid] = v;\n"
"        p += v;\n"
"    } else {\n"
"        p -= lr * g + weight_decay * p;\n"
"    }\n"
"    params[gid] = p;\n"
"}\n";

/* ---- copy ---- */
static const char* cl_kernel_copy_source =
"__kernel void copy_f32(__global float* dst, __global const float* src, int size) {\n"
"    int gid = get_global_id(0);\n"
"    if (gid < size) {\n"
"        dst[gid] = src[gid];\n"
"    }\n"
"}\n";

/* ---- fill zeros (for velocity init) ---- */
static const char* cl_kernel_zeros_source =
"__kernel void zeros_f32(__global float* data, int size) {\n"
"    int gid = get_global_id(0);\n"
"    if (gid < size) {\n"
"        data[gid] = 0.0f;\n"
"    }\n"
"}\n";

/* ---- gradient clipping: scale down if norm exceeds max ---- */
static const char* cl_kernel_clip_source =
"__kernel void clip_scale_f32(__global float* data, float max_norm, int size) {\n"
"    int gid = get_global_id(0);\n"
"    if (gid < size) {\n"
"        // Norm already computed on CPU; this kernel does in-place scaling\n"
"        // max_norm is passed as arg; data[i] *= (max_norm / computed_norm) done on CPU side\n"
"        // For now just identity (actual scaling done by copying scaled data)\n"
"        (void)max_norm;\n"
"    }\n"
"}\n";

/* ============================================================================
 * Kernel Cache Compilation (call once at startup)
 * ============================================================================ */

static cl_program cl_compile_one(CLOpenCL* cl, const char* source,
                                 const char* kernel_name, cl_kernel* out_kernel) {
    cl_int err;
    size_t src_len = strlen(source);
    cl_program prog = clCreateProgramWithSource(cl->context, 1, &source, &src_len, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create program for %s: %d\n", kernel_name, err);
        exit(1);
    }
    err = clBuildProgram(prog, 1, &cl->device, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(prog, cl->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size + 1);
        clGetProgramBuildInfo(prog, cl->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = '\0';
        fprintf(stderr, "Build log for %s:\n%s\n", kernel_name, log);
        free(log);
        exit(1);
    }
    cl_kernel k = clCreateKernel(prog, kernel_name, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create kernel %s: %d\n", kernel_name, err);
        exit(1);
    }
    *out_kernel = k;
    return prog;
}

static void cl_compile_kernels(CLOpenCL* cl, cl_kernel_cache_t* cache) {
    if (cache->compiled) return;
    memset(cache, 0, sizeof(cl_kernel_cache_t));

    cache->fill_prog  = cl_compile_one(cl, cl_kernel_fill_source, "fill_f32", &cache->fill_k);
    cache->relu_prog  = cl_compile_one(cl, cl_kernel_relu_source, "relu_f32", &cache->relu_k);
    cache->gemm_prog  = cl_compile_one(cl, cl_kernel_gemm_source, "gemm_f32", &cache->gemm_k);
    cache->elem_mul_prog = cl_compile_one(cl, cl_kernel_elem_mul_source, "elem_mul_f32", &cache->elem_mul_k);
    cache->bias_add_prog = cl_compile_one(cl, cl_kernel_bias_add_source, "bias_add_f32", &cache->bias_add_k);
    cache->softmax_prog  = cl_compile_one(cl, cl_kernel_softmax_source, "softmax_f32", &cache->softmax_k);
    cache->cross_entropy_grad_prog = cl_compile_one(cl, cl_kernel_cross_entropy_grad_source, "cross_entropy_grad_f32", &cache->cross_entropy_grad_k);
    cache->sgd_update_prog = cl_compile_one(cl, cl_kernel_sgd_update_source, "sgd_update_f32", &cache->sgd_update_k);
    cache->copy_prog  = cl_compile_one(cl, cl_kernel_copy_source, "copy_f32", &cache->copy_k);

    cache->compiled = true;
}

/* ============================================================================
 * Kernel Wrappers (use cached kernels, no per-call compilation)
 * ============================================================================ */

static void cl_tensor_fill_f32_impl(CLOpenCL* cl, cl_kernel_cache_t* cache,
                                     CLTensor* t, float val) {
    cl_int err;
    err = clSetKernelArg(cache->fill_k, 0, sizeof(cl_mem), &t->buffer);
    err |= clSetKernelArg(cache->fill_k, 1, sizeof(float), &val);
    err |= clSetKernelArg(cache->fill_k, 2, sizeof(int), (int*)&t->size);
    CL_CHECK(err, "fill: set arg");

    size_t global = ((t->size + 255) / 256) * 256;
    err = clEnqueueNDRangeKernel(cl->queue, cache->fill_k, 1, NULL, &global, NULL, 0, NULL, NULL);
    CL_CHECK(err, "fill: enqueue");
    clFinish(cl->queue);
}

static CLTensor* cl_tensor_relu_impl(CLOpenCL* cl, cl_kernel_cache_t* cache,
                                     const CLTensor* input) {
    CLTensor* output = cl_tensor_create(cl, input->dtype, input->layout,
                                        input->shape, input->ndim);
    if (!output) return NULL;

    cl_int err;
    err = clSetKernelArg(cache->relu_k, 0, sizeof(cl_mem), &output->buffer);
    err |= clSetKernelArg(cache->relu_k, 1, sizeof(cl_mem), &input->buffer);
    err |= clSetKernelArg(cache->relu_k, 2, sizeof(int), (int*)&input->size);
    CL_CHECK(err, "relu: set arg");

    size_t global = ((input->size + 255) / 256) * 256;
    err = clEnqueueNDRangeKernel(cl->queue, cache->relu_k, 1, NULL, &global, NULL, 0, NULL, NULL);
    CL_CHECK(err, "relu: enqueue");
    clFinish(cl->queue);
    return output;
}

static CLTensor* cl_tensor_matmul_impl(CLOpenCL* cl, cl_kernel_cache_t* cache,
                                       const CLTensor* a, const CLTensor* b) {
    size_t M = a->shape[0];
    size_t K = a->shape[1];
    size_t N = b->shape[1];

    size_t out_shape[] = {M, N};
    CLTensor* output = cl_tensor_create(cl, CL_TENSOR_DTYPE_F32, a->layout, out_shape, 2);
    if (!output) return NULL;

    cl_int err;
    int M_i = (int)M, N_i = (int)N, K_i = (int)K;
    float alpha = 1.0f, beta = 0.0f;

    err = clSetKernelArg(cache->gemm_k, 0, sizeof(cl_mem), &output->buffer);
    err |= clSetKernelArg(cache->gemm_k, 1, sizeof(cl_mem), &a->buffer);
    err |= clSetKernelArg(cache->gemm_k, 2, sizeof(cl_mem), &b->buffer);
    err |= clSetKernelArg(cache->gemm_k, 3, sizeof(int), &M_i);
    err |= clSetKernelArg(cache->gemm_k, 4, sizeof(int), &N_i);
    err |= clSetKernelArg(cache->gemm_k, 5, sizeof(int), &K_i);
    err |= clSetKernelArg(cache->gemm_k, 6, sizeof(float), &alpha);
    err |= clSetKernelArg(cache->gemm_k, 7, sizeof(float), &beta);
    CL_CHECK(err, "gemm: set arg");

    size_t global[2] = {((N + 15) / 16) * 16, ((M + 15) / 16) * 16};
    size_t local[2] = {16, 16};
    err = clEnqueueNDRangeKernel(cl->queue, cache->gemm_k, 2, NULL, global, local, 0, NULL, NULL);
    CL_CHECK(err, "gemm: enqueue");
    clFinish(cl->queue);
    return output;
}

static void cl_tensor_elem_mul_impl(CLOpenCL* cl, cl_kernel_cache_t* cache,
                                   const CLTensor* a, const CLTensor* b, CLTensor* output) {
    cl_int err;
    err = clSetKernelArg(cache->elem_mul_k, 0, sizeof(cl_mem), &output->buffer);
    err |= clSetKernelArg(cache->elem_mul_k, 1, sizeof(cl_mem), &a->buffer);
    err |= clSetKernelArg(cache->elem_mul_k, 2, sizeof(cl_mem), &b->buffer);
    err |= clSetKernelArg(cache->elem_mul_k, 3, sizeof(int), (int*)&a->size);
    CL_CHECK(err, "elem_mul: set arg");

    size_t global = ((a->size + 255) / 256) * 256;
    err = clEnqueueNDRangeKernel(cl->queue, cache->elem_mul_k, 1, NULL, &global, NULL, 0, NULL, NULL);
    CL_CHECK(err, "elem_mul: enqueue");
    clFinish(cl->queue);
}

static void cl_tensor_bias_add_impl(CLOpenCL* cl, cl_kernel_cache_t* cache,
                                    CLTensor* inout, const CLTensor* bias) {
    cl_int err;
    int batch = (int)inout->shape[0];
    int features = (int)inout->shape[1];

    err = clSetKernelArg(cache->bias_add_k, 0, sizeof(cl_mem), &inout->buffer);
    err |= clSetKernelArg(cache->bias_add_k, 1, sizeof(cl_mem), &inout->buffer);
    err |= clSetKernelArg(cache->bias_add_k, 2, sizeof(cl_mem), &bias->buffer);
    err |= clSetKernelArg(cache->bias_add_k, 3, sizeof(int), &batch);
    err |= clSetKernelArg(cache->bias_add_k, 4, sizeof(int), &features);
    CL_CHECK(err, "bias_add: set arg");

    size_t global = ((size_t)batch * features + 255) / 256 * 256;
    err = clEnqueueNDRangeKernel(cl->queue, cache->bias_add_k, 1, NULL, &global, NULL, 0, NULL, NULL);
    CL_CHECK(err, "bias_add: enqueue");
    clFinish(cl->queue);
}

static CLTensor* cl_tensor_softmax_impl(CLOpenCL* cl, cl_kernel_cache_t* cache,
                                        const CLTensor* t) {
    CLTensor* output = cl_tensor_create(cl, t->dtype, t->layout, t->shape, t->ndim);
    if (!output) return NULL;

    int batch = (int)t->shape[0];
    int num_classes = (int)t->shape[1];

    cl_int err;
    err = clSetKernelArg(cache->softmax_k, 0, sizeof(cl_mem), &output->buffer);
    err |= clSetKernelArg(cache->softmax_k, 1, sizeof(cl_mem), &t->buffer);
    err |= clSetKernelArg(cache->softmax_k, 2, sizeof(int), &batch);
    err |= clSetKernelArg(cache->softmax_k, 3, sizeof(int), &num_classes);
    CL_CHECK(err, "softmax: set arg");

    size_t global = batch;
    err = clEnqueueNDRangeKernel(cl->queue, cache->softmax_k, 1, NULL, &global, NULL, 0, NULL, NULL);
    CL_CHECK(err, "softmax: enqueue");
    clFinish(cl->queue);
    return output;
}

static void cl_tensor_cross_entropy_grad_impl(CLOpenCL* cl, cl_kernel_cache_t* cache,
                                              CLTensor* grad, const CLTensor* pred,
                                              const int* targets, size_t batch, size_t num_classes) {
    cl_int err;

    // Create host-pinned buffer for targets (kernel can't read int* directly)
    cl_mem targets_buf = clCreateBuffer(cl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        sizeof(int) * batch, (void*)targets, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "cross_entropy_grad: failed to create targets buffer: %d\n", err);
        exit(1);
    }

    int b = (int)batch;
    int c = (int)num_classes;

    err = clSetKernelArg(cache->cross_entropy_grad_k, 0, sizeof(cl_mem), &grad->buffer);
    err |= clSetKernelArg(cache->cross_entropy_grad_k, 1, sizeof(cl_mem), &pred->buffer);
    err |= clSetKernelArg(cache->cross_entropy_grad_k, 2, sizeof(cl_mem), &targets_buf);
    err |= clSetKernelArg(cache->cross_entropy_grad_k, 3, sizeof(int), &b);
    err |= clSetKernelArg(cache->cross_entropy_grad_k, 4, sizeof(int), &c);
    CL_CHECK(err, "cross_entropy_grad: set arg");

    size_t global[2] = {num_classes, batch};  // [c, b] for [c==targets[b]] indexing
    err = clEnqueueNDRangeKernel(cl->queue, cache->cross_entropy_grad_k, 2, NULL, global, NULL, 0, NULL, NULL);
    CL_CHECK(err, "cross_entropy_grad: enqueue");
    clFinish(cl->queue);

    clReleaseMemObject(targets_buf);
}

static void cl_tensor_sgd_update_impl(CLOpenCL* cl, cl_kernel_cache_t* cache,
                                      CLTensor* params, const CLTensor* grad,
                                      CLTensor* velocity, float lr, float momentum,
                                      float weight_decay) {
    cl_int err;
    int size = (int)params->size;
    float lr_f = lr, mom_f = momentum, wd_f = weight_decay;

    cl_mem vel_mem = velocity ? velocity->buffer : NULL;

    err = clSetKernelArg(cache->sgd_update_k, 0, sizeof(cl_mem), &params->buffer);
    err |= clSetKernelArg(cache->sgd_update_k, 1, sizeof(cl_mem), &grad->buffer);
    err |= clSetKernelArg(cache->sgd_update_k, 2, sizeof(cl_mem), &vel_mem);
    err |= clSetKernelArg(cache->sgd_update_k, 3, sizeof(float), &lr_f);
    err |= clSetKernelArg(cache->sgd_update_k, 4, sizeof(float), &mom_f);
    err |= clSetKernelArg(cache->sgd_update_k, 5, sizeof(float), &wd_f);
    err |= clSetKernelArg(cache->sgd_update_k, 6, sizeof(int), &size);
    CL_CHECK(err, "sgd_update: set arg");

    size_t global = ((size_t)size + 255) / 256 * 256;
    err = clEnqueueNDRangeKernel(cl->queue, cache->sgd_update_k, 1, NULL, &global, NULL, 0, NULL, NULL);
    CL_CHECK(err, "sgd_update: enqueue");
    clFinish(cl->queue);
}

static void cl_tensor_copy_impl(CLOpenCL* cl, cl_kernel_cache_t* cache,
                                 CLTensor* dst, const CLTensor* src) {
    if (dst->size != src->size) {
        fprintf(stderr, "copy: size mismatch %zu vs %zu\n", dst->size, src->size);
        return;
    }
    cl_int err;
    int size = (int)dst->size;
    err = clSetKernelArg(cache->copy_k, 0, sizeof(cl_mem), &dst->buffer);
    err |= clSetKernelArg(cache->copy_k, 1, sizeof(cl_mem), &src->buffer);
    err |= clSetKernelArg(cache->copy_k, 2, sizeof(int), &size);
    CL_CHECK(err, "copy: set arg");

    size_t global = ((size_t)size + 255) / 256 * 256;
    err = clEnqueueNDRangeKernel(cl->queue, cache->copy_k, 1, NULL, &global, NULL, 0, NULL, NULL);
    CL_CHECK(err, "copy: enqueue");
    clFinish(cl->queue);
}

/* ============================================================================
 * Softmax (host fallback - complex indexing / reduction)
 * cross-entropy gradient (requires int* targets on CPU)
 * ============================================================================ */

static CLTensor* cl_tensor_softmax(CLOpenCL* cl, const CLTensor* t, size_t axis) {
    (void)axis;
    CLTensor* output = cl_tensor_create(cl, t->dtype, t->layout, t->shape, t->ndim);
    if (!output) return NULL;

    float* h_data = (float*)malloc(t->nbytes);
    cl_tensor_download(t, h_data);

    size_t stride = t->strides[axis];
    size_t block = t->shape[axis];
    size_t blocks = t->size / block;

    for (size_t b = 0; b < blocks; b++) {
        size_t base = b * block * stride;
        float max_val = h_data[base];
        for (size_t i = 1; i < block; i++) {
            float v = h_data[base + i * stride];
            if (v > max_val) max_val = v;
        }
        float sum = 0.0f;
        for (size_t i = 0; i < block; i++) {
            h_data[base + i * stride] = expf(h_data[base + i * stride] - max_val);
            sum += h_data[base + i * stride];
        }
        for (size_t i = 0; i < block; i++) {
            h_data[base + i * stride] /= sum;
        }
    }

    cl_tensor_upload(output, h_data);
    free(h_data);
    return output;
}

/* ============================================================================
 * Element-wise add with broadcasting (host fallback)
 * ============================================================================ */

static CLTensor* cl_tensor_add(CLOpenCL* cl, const CLTensor* a, const CLTensor* b) {
    uint8_t max_ndim = (a->ndim > b->ndim) ? a->ndim : b->ndim;
    size_t out_shape[4] = {1, 1, 1, 1};

    for (int i = 0; i < max_ndim; i++) {
        size_t dim_a = (i < (int)(max_ndim - a->ndim)) ? 1 : a->shape[i - (max_ndim - a->ndim)];
        size_t dim_b = (i < (int)(max_ndim - b->ndim)) ? 1 : b->shape[i - (max_ndim - b->ndim)];
        out_shape[i] = (dim_a > dim_b) ? dim_a : dim_b;
    }

    CLTensor* output = cl_tensor_create(cl, a->dtype, a->layout, out_shape, max_ndim);
    if (!output) return NULL;

    float* h_a = (float*)malloc(a->nbytes);
    float* h_b = (float*)malloc(b->nbytes);
    float* h_out = (float*)malloc(output->nbytes);

    cl_tensor_download(a, h_a);
    cl_tensor_download(b, h_b);

    size_t total = output->size;
    size_t idx_buf[4] = {0};
    for (size_t i = 0; i < total; i++) {
        size_t temp = i;
        for (int d = max_ndim - 1; d >= 0; d--) {
            idx_buf[d] = temp % out_shape[d];
            temp /= out_shape[d];
        }

        size_t idx_a = 0;
        for (int d = 0; d < a->ndim; d++) {
            size_t effective_idx = (a->shape[d] == 1) ? 0 : idx_buf[d + (max_ndim - a->ndim)];
            idx_a += effective_idx * a->strides[d];
        }

        size_t idx_b = 0;
        for (int d = 0; d < b->ndim; d++) {
            size_t effective_idx = (b->shape[d] == 1) ? 0 : idx_buf[d + (max_ndim - b->ndim)];
            idx_b += effective_idx * b->strides[d];
        }

        h_out[i] = h_a[idx_a] + h_b[idx_b];
    }

    cl_tensor_upload(output, h_out);
    free(h_a); free(h_b); free(h_out);
    return output;
}

/* ============================================================================
 * Kernel Wrappers (public API - compile kernels on first use)
 * ============================================================================ */

static void cl_tensor_fill_f32(CLOpenCL* cl, CLTensor* t, float val) {
    cl_compile_kernels(cl, &cl->kernel_cache);
    cl_tensor_fill_f32_impl(cl, &cl->kernel_cache, t, val);
}

static CLTensor* cl_tensor_relu(CLOpenCL* cl, const CLTensor* input) {
    cl_compile_kernels(cl, &cl->kernel_cache);
    return cl_tensor_relu_impl(cl, &cl->kernel_cache, input);
}

static CLTensor* cl_tensor_matmul(CLOpenCL* cl, const CLTensor* a, const CLTensor* b) {
    cl_compile_kernels(cl, &cl->kernel_cache);
    return cl_tensor_matmul_impl(cl, &cl->kernel_cache, a, b);
}

/* ============================================================================
 * Debug
 * ============================================================================ */

static void cl_tensor_print(const CLTensor* t, const char* name) {
    printf("%s: shape=[", name ? name : "cl_tensor");
    for (uint8_t i = 0; i < t->ndim; i++) printf("%zu%s", t->shape[i], i < t->ndim-1 ? "," : "");
    printf("], dtype=%d, size=%zu, nbytes=%zu\n", t->dtype, t->size, t->nbytes);
}

static void cl_tensor_print_data(const CLTensor* t, const char* name, size_t max_elem) {
    float* data = (float*)malloc(t->nbytes);
    clEnqueueReadBuffer(t->queue, t->buffer, CL_TRUE, 0, t->nbytes, data, 0, NULL, NULL);
    clFinish(t->queue);

    cl_tensor_print(t, name);
    printf("  data (first %zu): ", max_elem < t->size ? max_elem : t->size);
    for (size_t i = 0; i < (max_elem < t->size ? max_elem : t->size); i++) {
        printf("%.4f ", data[i]);
    }
    printf("\n");

    free(data);
}

#endif /* __OPENCL_TENSOR_H__ */
