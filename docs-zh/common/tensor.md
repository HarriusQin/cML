# 张量

N 维数组数据结构及运算。

## 概述

`Tensor` 是深度学习的核心数据结构，支持：
- 任意维度
- 行主序布局（NCHW 用于图像）
- 量化运算（int8）
- 自动微分（计划中）

## 数据结构

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
    TENSOR_LAYOUT_ANY           // 手动指定 strides
} TensorLayout;

typedef struct {
    void* data;                 // 原始数据缓冲区
    size_t size;               // 元素总数
    TensorDType dtype;         // 数据类型
    TensorLayout layout;       // 内存布局
    uint8_t ndim;              // 维度数
    size_t shape[4];          // 最大 4 维
    size_t strides[4];         // 每维步长
    bool owner;               // 是否拥有数据
} Tensor;
```

## 张量创建

```c
// 根据形状创建张量
Tensor* tensor_create(TensorDType dtype, TensorLayout layout,
                      size_t* shape, uint8_t ndim);

// 从现有数据创建（不复制）
Tensor* tensor_wrap(void* data, TensorDType dtype, TensorLayout layout,
                    size_t* shape, uint8_t ndim);

// 克隆张量
Tensor* tensor_clone(const Tensor* t);

// 释放张量
void tensor_free(Tensor* t);
```

## 填充运算

```c
void tensor_fill_f32(Tensor* t, float val);
void tensor_fill_randn(Tensor* t, float mean, float std_dev);
void tensor_fill_xavier(Tensor* t);
```

## 逐元素运算

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

## 归约运算

```c
Tensor* tensor_sum(const Tensor* t, size_t axis);
Tensor* tensor_mean(const Tensor* t, size_t axis);
Tensor* tensor_max(const Tensor* t, size_t axis);
```

## 矩阵运算

```c
Tensor* tensor_matmul(const Tensor* a, const Tensor* b);
void tensor_gemm(float* C, const float* A, const float* B,
                size_t M, size_t N, size_t K,
                float alpha, float beta);
```

## 形状运算

```c
Tensor* tensor_reshape(const Tensor* t, size_t* new_shape, uint8_t ndim);
Tensor* tensor_transpose(const Tensor* t, uint8_t axis0, uint8_t axis1);
Tensor* tensor_slice(const Tensor* t, size_t dim, size_t start, size_t end);
```

## 池化和卷积

```c
Tensor* tensor_maxpool2d(const Tensor* input, size_t pool_h, size_t pool_w,
                        size_t stride_h, size_t stride_w);

Tensor* tensor_avgpool2d(const Tensor* input, size_t pool_h, size_t pool_w,
                        size_t stride_h, size_t stride_w);

Tensor* tensor_conv2d(const Tensor* input, const Tensor* weight,
                     const Conv2DParams* params);
```

## 量化

```c
// 量化为 int8（仿射映射）
Tensor* tensor_quantize_affine(const Tensor* t, TensorDType dtype);

// 反量化回 float32
Tensor* tensor_dequantize(const Tensor* t);
```

## 示例

```c
#define TENSOR_IMPLEMENTATION
#include "tensor.h"

// 创建 [32, 3, 224, 224] 张量（图像批次）
size_t shape[] = {32, 3, 224, 224};
Tensor* input = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 4);

// 填充随机值
tensor_fill_randn(input, 0.0f, 1.0f);

// 前向传播
Conv2DParams params = {
    .stride_h = 1, .stride_w = 1,
    .pad_h = 1, .pad_w = 1
};
Tensor* output = tensor_conv2d(input, weight, &params);
tensor_relu(output);

// 释放
tensor_free(input);
tensor_free(output);
```

## 说明

- 所有张量使用行主序（C 风格）布局
- shape 和 stride 数组固定为 4 个元素以简化实现
- 量化张量存储 scale 和 zero_point 以便精确转换
- 大多数运算支持原地变体（如 tensor_relu_inplace）
