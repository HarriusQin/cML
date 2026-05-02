# OpenCL Tensor

GPU-resident tensors with host-side metadata for OpenCL acceleration.

## Overview

`CLTensor` wraps an OpenCL buffer (`cl_mem`) with metadata describing the tensor shape, layout, and data type. Operations are performed on the GPU.

## Data Structure

```c
typedef struct {
    cl_mem buffer;              // OpenCL memory buffer
    cl_context context;         // OpenCL context
    cl_command_queue queue;     // Command queue
    CLTensorLayout layout;
    CLTensorDType dtype;
    uint8_t ndim;
    size_t* shape;
    size_t* strides;
    size_t size;
    size_t nbytes;
    cl_uint dev_id;
} CLTensor;
```

## Layout and Data Types

```c
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
```

## OpenCL Context

```c
typedef struct {
    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
    cl_uint dev_count;
    cl_platform_id platform;
    cl_kernel_cache_t kernel_cache;  // Pre-compiled kernels
} CLOpenCL;

// Initialize
bool cl_init(CLOpenCL* cl, cl_device_type device_type);
void cl_release(CLOpenCL* cl);
void cl_print_device_info(cl_device_id device);
```

## Tensor Creation

```c
// Create GPU tensor
CLTensor* cl_tensor_create(CLOpenCL* cl, CLTensorDType dtype,
                            CLTensorLayout layout,
                            size_t* shape, uint8_t ndim);

// Create from existing host data
CLTensor* cl_tensor_create_from_host(CLOpenCL* cl, CLTensorDType dtype,
                                      CLTensorLayout layout,
                                      size_t* shape, uint8_t ndim,
                                      const void* host_data);

// Free
void cl_tensor_free(CLTensor* t);
```

## Data Transfer

```c
// Upload to GPU
void cl_tensor_upload(CLTensor* t, const void* host_data);

// Download from GPU
void cl_tensor_download(const CLTensor* t, void* host_data);
```

## GPU Operations

```c
// ReLU activation
CLTensor* cl_tensor_relu(CLOpenCL* cl, const CLTensor* input);

// Matrix multiplication
CLTensor* cl_tensor_matmul(CLOpenCL* cl, const CLTensor* a, const CLTensor* b);

// Element-wise fill
void cl_tensor_fill_f32(CLOpenCL* cl, CLTensor* t, float val);

// Softmax
CLTensor* cl_tensor_softmax(CLOpenCL* cl, const CLTensor* t, size_t axis);
```

## Example

```c
#define CL_TENSOR_IMPLEMENTATION
#include "opencl_tensor.h"

// Initialize OpenCL
CLOpenCL cl;
if (!cl_init(&cl, CL_DEVICE_TYPE_GPU)) {
    fprintf(stderr, "No OpenCL device found\n");
    return 1;
}
cl_print_device_info(cl.device);

// Create GPU tensor from host data
size_t shape[] = {1024, 1024};
float* h_data = (float*)malloc(1024 * 1024 * sizeof(float));
// ... fill h_data ...

CLTensor* gpu_tensor = cl_tensor_create_from_host(&cl, CL_TENSOR_DTYPE_F32,
                                                    CL_TENSOR_LAYOUT_NCHW,
                                                    shape, 2, h_data);

// GPU operation
CLTensor* gpu_result = cl_tensor_relu(&cl, gpu_tensor);

// Download result
float* h_result = (float*)malloc(gpu_result->nbytes);
cl_tensor_download(gpu_result, h_result);

// Cleanup
cl_tensor_free(gpu_tensor);
cl_tensor_free(gpu_result);
free(h_data);
free(h_result);
cl_release(&cl);
```

## Kernel Caching

The `CLOpenCL` structure includes a `kernel_cache` that pre-compiles OpenCL kernels once and reuses them:

- `fill_k`, `relu_k`, `gemm_k` - compiled kernels
- Lazy compilation: kernels are compiled on first use

This avoids the overhead of JIT compilation for each operation.

## Notes

- All GPU operations are synchronous by default (blocking)
- For async operations, use non-blocking enqueue functions
- Memory is allocated on the GPU device
- Always call `cl_tensor_free()` to avoid memory leaks
