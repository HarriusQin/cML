# OpenCL 张量

带有 GPU 端元数据的 GPU 常驻张量用于 OpenCL 加速。

## 概述

`CLTensor` 使用元数据（张量形状、布局和数据类型）包装 OpenCL 缓冲区 (`cl_mem`)。操作在 GPU 上执行。

## 数据结构

```c
typedef struct {
    cl_mem buffer;              // OpenCL 内存缓冲区
    cl_context context;         // OpenCL 上下文
    cl_command_queue queue;     // 命令队列
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

## 布局和数据类型

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

## OpenCL 上下文

```c
typedef struct {
    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
    cl_uint dev_count;
    cl_platform_id platform;
    cl_kernel_cache_t kernel_cache;  // 预编译内核
} CLOpenCL;

// 初始化
bool cl_init(CLOpenCL* cl, cl_device_type device_type);
void cl_release(CLOpenCL* cl);
void cl_print_device_info(cl_device_id device);
```

## 张量创建

```c
// 创建 GPU 张量
CLTensor* cl_tensor_create(CLOpenCL* cl, CLTensorDType dtype,
                            CLTensorLayout layout,
                            size_t* shape, uint8_t ndim);

// 从现有主机数据创建
CLTensor* cl_tensor_create_from_host(CLOpenCL* cl, CLTensorDType dtype,
                                      CLTensorLayout layout,
                                      size_t* shape, uint8_t ndim,
                                      const void* host_data);

// 释放
void cl_tensor_free(CLTensor* t);
```

## 数据传输

```c
// 上传到 GPU
void cl_tensor_upload(CLTensor* t, const void* host_data);

// 下载从 GPU
void cl_tensor_download(const CLTensor* t, void* host_data);
```

## GPU 运算

```c
// ReLU 激活
CLTensor* cl_tensor_relu(CLOpenCL* cl, const CLTensor* input);

// 矩阵乘法
CLTensor* cl_tensor_matmul(CLOpenCL* cl, const CLTensor* a, const CLTensor* b);

// 逐元素填充
void cl_tensor_fill_f32(CLOpenCL* cl, CLTensor* t, float val);

// Softmax
CLTensor* cl_tensor_softmax(CLOpenCL* cl, const CLTensor* t, size_t axis);
```

## 示例

```c
#define CL_TENSOR_IMPLEMENTATION
#include "opencl_tensor.h"

// 初始化 OpenCL
CLOpenCL cl;
if (!cl_init(&cl, CL_DEVICE_TYPE_GPU)) {
    fprintf(stderr, "No OpenCL device found\n");
    return 1;
}
cl_print_device_info(cl.device);

// 从主机数据创建 GPU 张量
size_t shape[] = {1024, 1024};
float* h_data = (float*)malloc(1024 * 1024 * sizeof(float));
// ... 填充 h_data ...

CLTensor* gpu_tensor = cl_tensor_create_from_host(&cl, CL_TENSOR_DTYPE_F32,
                                                    CL_TENSOR_LAYOUT_NCHW,
                                                    shape, 2, h_data);

// GPU 运算
CLTensor* gpu_result = cl_tensor_relu(&cl, gpu_tensor);

// 下载结果
float* h_result = (float*)malloc(gpu_result->nbytes);
cl_tensor_download(gpu_result, h_result);

// 清理
cl_tensor_free(gpu_tensor);
cl_tensor_free(gpu_result);
free(h_data);
free(h_result);
cl_release(&cl);
```

## 内核缓存

`CLOpenCL` 结构包含 `kernel_cache`，预编译 OpenCL 内核一次并重用：

- `fill_k`, `relu_k`, `gemm_k` - 已编译内核
- 延迟编译：内核在首次使用时编译

这避免了每次操作进行 JIT 编译的开销。

## 说明

- 所有 GPU 操作默认同步（阻塞）
- 对于异步操作，使用非阻塞入队函数
- 内存分配在 GPU 设备上
- 始终调用 `cl_tensor_free()` 以避免内存泄漏
