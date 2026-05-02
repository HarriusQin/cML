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
 * OpenCL Context & Queue Management
 * ============================================================================ */

typedef struct {
    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
    cl_uint dev_count;
    cl_platform_id platform;
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
 * Basic OpenCL Kernels (inline kernel source)
 * ============================================================================ */

/* --------------------------------------------------------------------------
 * Fill with constant
 * -------------------------------------------------------------------------- */

static const char* cl_kernel_fill_source =
"__kernel void fill_f32(__global float* output, float value, int size) {\n"
"    int gid = get_global_id(0);\n"
"    if (gid < size) {\n"
"        output[gid] = value;\n"
"    }\n"
"}\n";

static void cl_tensor_fill_f32(CLOpenCL* cl, CLTensor* t, float val) {
    cl_int err;
    size_t src_len = strlen(cl_kernel_fill_source);

    cl_program prog = clCreateProgramWithSource(cl->context, 1,
                                                &cl_kernel_fill_source, &src_len, &err);
    CL_CHECK(err, "Failed to create program");

    err = clBuildProgram(prog, 1, &cl->device, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(prog, cl->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(prog, cl->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build log:\n%s\n", log);
        free(log);
        exit(1);
    }

    cl_kernel kernel = clCreateKernel(prog, "fill_f32", &err);
    CL_CHECK(err, "Failed to create kernel");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &t->buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(float), &val);
    err |= clSetKernelArg(kernel, 2, sizeof(int), (int*)&t->size);
    CL_CHECK(err, "Failed to set kernel args");

    size_t global = ((t->size + 255) / 256) * 256;
    err = clEnqueueNDRangeKernel(cl->queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    CL_CHECK(err, "Failed to enqueue kernel");
    clFinish(cl->queue);

    clReleaseKernel(kernel);
    clReleaseProgram(prog);
}

/* --------------------------------------------------------------------------
 * ReLU: y = max(0, x)
 * -------------------------------------------------------------------------- */

static const char* cl_kernel_relu_source =
"__kernel void relu_f32(__global float* output, __global const float* input, int size) {\n"
"    int gid = get_global_id(0);\n"
"    if (gid < size) {\n"
"        float v = input[gid];\n"
"        output[gid] = v > 0 ? v : 0;\n"
"    }\n"
"}\n";

static CLTensor* cl_tensor_relu(CLOpenCL* cl, const CLTensor* input) {
    CLTensor* output = cl_tensor_create(cl, input->dtype, input->layout,
                                        input->shape, input->ndim);
    if (!output) return NULL;

    cl_int err;
    size_t src_len = strlen(cl_kernel_relu_source);

    cl_program prog = clCreateProgramWithSource(cl->context, 1,
                                                &cl_kernel_relu_source, &src_len, &err);
    CL_CHECK(err, "Failed to create program");

    err = clBuildProgram(prog, 1, &cl->device, "-cl-fast-relaxed-math", NULL, NULL);
    CL_CHECK(err, "Failed to build program");

    cl_kernel kernel = clCreateKernel(prog, "relu_f32", &err);
    CL_CHECK(err, "Failed to create kernel");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &output->buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &input->buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(int), (int*)&input->size);
    CL_CHECK(err, "Failed to set kernel args");

    size_t global = ((input->size + 255) / 256) * 256;
    err = clEnqueueNDRangeKernel(cl->queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    CL_CHECK(err, "Failed to enqueue kernel");
    clFinish(cl->queue);

    clReleaseKernel(kernel);
    clReleaseProgram(prog);

    return output;
}

/* --------------------------------------------------------------------------
 * GEMM: C = alpha * A @ B + beta * C
 * -------------------------------------------------------------------------- */

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

static CLTensor* cl_tensor_matmul(CLOpenCL* cl, const CLTensor* a, const CLTensor* b) {
    size_t M = a->shape[0];
    size_t K = a->shape[1];
    size_t N = b->shape[1];

    size_t out_shape[] = {M, N};
    CLTensor* output = cl_tensor_create(cl, CL_TENSOR_DTYPE_F32, a->layout, out_shape, 2);
    if (!output) return NULL;

    cl_int err;
    size_t src_len = strlen(cl_kernel_gemm_source);

    cl_program prog = clCreateProgramWithSource(cl->context, 1,
                                                &cl_kernel_gemm_source, &src_len, &err);
    CL_CHECK(err, "Failed to create program");

    err = clBuildProgram(prog, 1, &cl->device, "-cl-fast-relaxed-math", NULL, NULL);
    CL_CHECK(err, "Failed to build program");

    cl_kernel kernel = clCreateKernel(prog, "gemm_f32", &err);
    CL_CHECK(err, "Failed to create kernel");

    int M_i = (int)M, N_i = (int)N, K_i = (int)K;
    float alpha = 1.0f, beta = 0.0f;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &output->buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &b->buffer);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &M_i);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &N_i);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &K_i);
    err |= clSetKernelArg(kernel, 6, sizeof(float), &alpha);
    err |= clSetKernelArg(kernel, 7, sizeof(float), &beta);
    CL_CHECK(err, "Failed to set kernel args");

    size_t global[2] = {((N + 15) / 16) * 16, ((M + 15) / 16) * 16};
    size_t local[2] = {16, 16};

    err = clEnqueueNDRangeKernel(cl->queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
    CL_CHECK(err, "Failed to enqueue kernel");
    clFinish(cl->queue);

    clReleaseKernel(kernel);
    clReleaseProgram(prog);

    return output;
}

/* --------------------------------------------------------------------------
 * Softmax (host fallback - more reliable for complex indexing)
 * -------------------------------------------------------------------------- */

static CLTensor* cl_tensor_softmax(CLOpenCL* cl, const CLTensor* t, size_t axis) {
    CLTensor* output = cl_tensor_create(cl, t->dtype, t->layout, t->shape, t->ndim);
    if (!output) return NULL;

    float* h_data = (float*)malloc(t->nbytes);
    cl_tensor_download(t, h_data);

    // Compute softmax directly
    size_t stride = t->strides[axis];
    size_t block = t->shape[axis];
    size_t blocks = t->size / block;

    for (size_t b = 0; b < blocks; b++) {
        size_t base = b * block * stride;
        // Find max
        float max_val = h_data[base];
        for (size_t i = 1; i < block; i++) {
            float v = h_data[base + i * stride];
            if (v > max_val) max_val = v;
        }
        // Exp and sum
        float sum = 0.0f;
        for (size_t i = 0; i < block; i++) {
            h_data[base + i * stride] = expf(h_data[base + i * stride] - max_val);
            sum += h_data[base + i * stride];
        }
        // Normalize
        for (size_t i = 0; i < block; i++) {
            h_data[base + i * stride] /= sum;
        }
    }

    cl_tensor_upload(output, h_data);
    free(h_data);
    return output;
}

/* --------------------------------------------------------------------------
 * Element-wise add with broadcasting (host fallback)
 * -------------------------------------------------------------------------- */

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
    size_t idx_buf[4] = {0, 0, 0, 0};
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

    free(h_a);
    free(h_b);
    free(h_out);

    return output;
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
