# Tensor

N-dimensional array data structure and operations.

## Overview

The `Tensor` is the core data structure for deep learning, supporting:
- Arbitrary dimensions
- Row-major layout (NCHW for images)
- Quantized operations (int8)
- Automatic differentiation (planned)

## Data Structure

```c
typedef enum {
    TENSOR_DTYPE_F32,           // float32
    TENSOR_DTYPE_F16,            // float16
    TENSOR_DTYPE_INT8,          // int8_t
    TENSOR_DTYPE_INT32,         // int32_t
    TENSOR_DTYPE_UINT8,         // uint8_t
} TensorDType;

typedef enum {
    TENSOR_LAYOUT_NCHW,         // Batch, Channel, Height, Width
    TENSOR_LAYOUT_NHWC,         // Batch, Height, Width, Channel
    TENSOR_LAYOUT_CHWN,         // Channel, Height, Width, Batch
    TENSOR_LAYOUT_ANY           // Strides provided
} TensorLayout;

typedef struct {
    void* data;                 // Raw data buffer
    size_t size;               // Total number of elements
    TensorDType dtype;         // Data type
    TensorLayout layout;       // Memory layout
    uint8_t ndim;              // Number of dimensions
    size_t shape[4];          // Max 4 dimensions
    size_t strides[4];         // Stride for each dimension
    bool owner;               // Whether we own the data
} Tensor;
```

## Tensor Creation

```c
// Create tensor with shape
Tensor* tensor_create(TensorDType dtype, TensorLayout layout,
                      size_t* shape, uint8_t ndim);

// Create from existing data (no copy)
Tensor* tensor_wrap(void* data, TensorDType dtype, TensorLayout layout,
                    size_t* shape, uint8_t ndim);

// Clone tensor
Tensor* tensor_clone(const Tensor* t);

// Free tensor
void tensor_free(Tensor* t);
```

## Fill Operations

```c
void tensor_fill_f32(Tensor* t, float val);
void tensor_fill_randn(Tensor* t, float mean, float std_dev);
void tensor_fill_xavier(Tensor* t);
```

## Element-wise Operations

```c
Tensor* tensor_add(const Tensor* a, const Tensor* b);
Tensor* tensor_sub(const Tensor* a, const Tensor* b);
Tensor* tensor_mul(const Tensor* a, const Tensor* b);
Tensor* tensor_div(const Tensor* a, const Tensor* b);
void tensor_scale(Tensor* t, float scalar);
void tensor_relu(Tensor* t);
void tensor_sigmoid(Tensor* t);
void tensor_tanh(Tensor* t);
```

## Reduction Operations

```c
Tensor* tensor_sum(const Tensor* t, size_t axis);
Tensor* tensor_mean(const Tensor* t, size_t axis);
Tensor* tensor_max(const Tensor* t, size_t axis);
```

## Matrix Operations

```c
Tensor* tensor_matmul(const Tensor* a, const Tensor* b);
void tensor_gemm(float* C, const float* A, const float* B,
                size_t M, size_t N, size_t K,
                float alpha, float beta);
```

## Shape Operations

```c
Tensor* tensor_reshape(const Tensor* t, size_t* new_shape, uint8_t ndim);
Tensor* tensor_transpose(const Tensor* t, uint8_t axis0, uint8_t axis1);
Tensor* tensor_slice(const Tensor* t, size_t dim, size_t start, size_t end);
```

## Pooling and Convolution

```c
Tensor* tensor_maxpool2d(const Tensor* input, size_t pool_h, size_t pool_w,
                        size_t stride_h, size_t stride_w);

Tensor* tensor_avgpool2d(const Tensor* input, size_t pool_h, size_t pool_w,
                        size_t stride_h, size_t stride_w);

Tensor* tensor_conv2d(const Tensor* input, const Tensor* weight,
                     const Conv2DParams* params);
```

## Quantization

```c
// Quantize to int8 with affine mapping
Tensor* tensor_quantize_affine(const Tensor* t, TensorDType dtype);

// Dequantize back to float32
Tensor* tensor_dequantize(const Tensor* t);
```

## Accessing Elements

```c
float tensor_get_f32(const Tensor* t, size_t* indices);
void tensor_set_f32(Tensor* t, size_t* indices, float val);
```

## Example

```c
#define TENSOR_IMPLEMENTATION
#include "tensor.h"

// Create [32, 3, 224, 224] tensor (batch of images)
size_t shape[] = {32, 3, 224, 224};
Tensor* input = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 4);

// Fill with random values
tensor_fill_randn(input, 0.0f, 1.0f);

// Forward pass through convolution
size_t weight_shape[] = {64, 3, 3, 3};
Tensor* weight = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, weight_shape, 4);
tensor_fill_randn(weight, 0.0f, 0.01f);

Conv2DParams params = {
    .stride_h = 1, .stride_w = 1,
    .pad_h = 1, .pad_w = 1
};
Tensor* output = tensor_conv2d(input, weight, &params);

// ReLU activation
tensor_relu(output);

// Cleanup
tensor_free(input);
tensor_free(weight);
tensor_free(output);
```

## Notes

- All tensors use row-major (C-style) layout
- Shape and stride arrays are fixed at 4 elements for simplicity
- Quantized tensors store scale and zero_point for accurate conversion
- Most operations support in-place variants (e.g., tensor_relu_inplace)
