# OpenCL Tensor 运算实现文档

## 1. 概述

`opencl_tensor.h` 是基于 OpenCL 的 GPU 加速张量运算实现，对应 CPU 版本 [tensor.h](tensor.h)。该实现使用内联内核源码（Inline Kernel Source）的方式，在运行时编译 OpenCL C 内核，无需预编译。

---

## 2. 核心数据结构

### 2.1 CLTensor（GPU 张量）

```c
typedef struct {
    cl_mem buffer;        // OpenCL 内存对象（GPU VRAM 数据存储）
    cl_context context;   // OpenCL 上下文
    cl_command_queue queue; // 命令队列
    CLTensorLayout layout; // 内存布局
    CLTensorDType dtype;   // 数据类型
    uint8_t ndim;          // 维度数
    size_t* shape;         // 各维度大小
    size_t* strides;       // 各维度步长（元素为单位）
    size_t size;           // 元素总数
    size_t nbytes;         // 字节数
    cl_uint dev_id;        // 设备 ID
} CLTensor;
```

**设计要点：**
- `cl_mem buffer` 是 GPU VRAM 中的数据存储，通过 OpenCL API 操作
- 元数据（shape, strides）保留在主机端（GPU 无法直接访问指针）
- 支持 NCHW / NHWC / CHWN 三种数据布局
- 与 CPU Tensor 的主要区别：数据在 VRAM，通过 `clEnqueueWriteBuffer`/`clEnqueueReadBuffer` 传输

### 2.2 CLOpenCL（OpenCL 运行环境）

```c
typedef struct {
    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
    cl_uint dev_count;
    cl_platform_id platform;
} CLOpenCL;
```

### 2.3 数据类型枚举

```c
typedef enum {
    CL_TENSOR_LAYOUT_NCHW,  // [N, C, H, W]
    CL_TENSOR_LAYOUT_NHWC,  // [N, H, W, C]
    CL_TENSOR_LAYOUT_CHWN,  // [C, H, W, N]
} CLTensorLayout;

typedef enum {
    CL_TENSOR_DTYPE_F32,   // float 32
    CL_TENSOR_DTYPE_F16,   // float 16 (half)
    CL_TENSOR_DTYPE_INT8,   // int 8
    CL_TENSOR_DTYPE_INT32,  // int 32
} CLTensorDType;
```

---

## 3. 初始化与清理

### 3.1 初始化 OpenCL

```c
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
```

**流程：**
1. `clGetPlatformIDs` — 获取第一个 OpenCL 平台
2. `clGetDeviceIDs` — 在指定平台查找指定类型设备（GPU/CPU）
3. `clCreateContext` — 创建 OpenCL 上下文
4. `clCreateCommandQueue` — 创建命令队列

### 3.2 打印设备信息

```c
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
```

### 3.3 释放资源

```c
static void cl_release(CLOpenCL* cl) {
    if (cl->queue) clReleaseCommandQueue(cl->queue);
    if (cl->context) clReleaseContext(cl->context);
}
```

---

## 4. 数据类型辅助函数

```c
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
```

---

## 5. 步长计算与内存布局详解

### 5.1 步长计算实现

```c
static void cl_tensor_compute_strides(CLTensor* t) {
    t->strides[t->ndim - 1] = 1;
    for (int i = (int)t->ndim - 2; i >= 0; i--) {
        t->strides[i] = t->strides[i + 1] * t->shape[i + 1];
    }
}
```

与 CPU 版本完全一致 — OpenCL 版本也使用行主序策略。`strides` 同样以**元素为单位**。

### 5.2 各维度内存布局详解

GPU 张量的内存布局与 CPU 版本相同，只是数据存储在 VRAM 而非 RAM。以下以 `CL_TENSOR_DTYPE_F32`（4字节）为例说明物理排布。

#### 1维张量：向量 `[D]`

```
shape = [4], strides = [1]
物理内存: [v0, v1, v2, v3]  (VRAM 中连续存储)
flat_index(i) = i
```

#### 2维张量：矩阵 `[H, W]`

| 布局 | shape | strides | 物理排布（2×3矩阵）|
|------|-------|---------|--------------------|
| NCHW | [2, 3] | [3, 1] | 行主序，先行0元素，再行1元素 |
| NHWC | [2, 3] | [1, 2] | 列主（实际少用）|

```
NCHW: shape=[2,3], strides=[3,1]
  物理: [r0c0, r0c1, r0c2, r1c0, r1c1, r1c2]
  flat_index(1,2) = 1*3 + 2*1 = 5
```

#### 3维张量：`[D0, D1, D2]`

| 布局 | shape | strides | 典型应用 |
|------|-------|---------|----------|
| NCHW | [B,C,W] | [C×W, W, 1] | 图像批次（batch, channel, pixel） |
| NHWC | [B,W,C] | [W×C, C, 1] | TensorFlow 视频格式 |
| CHWN | [C,W,B] | [W×B, B, 1] | 通道优先卷积中间结果 |

**NCHW 3维示例（图像批次）：**
```
shape=[2, 3, 4], strides=[12, 4, 1]
  → 2张图像, 3通道, 每通道4像素

物理内存排布（按 flat_index 顺序）:
  index 0-3:   b0c0: [p0, p1, p2, p3]
  index 4-7:   b0c1: [p0, p1, p2, p3]
  index 8-11:  b0c2: [p0, p1, p2, p3]
  index 12-15: b1c0: [p0, p1, p2, p3]
  ...
  index 20-23: b1c2: [p0, p1, p2, p3]

flat_index(b,c,w) = b*12 + c*4 + w*1
```

**NHWC 3维示例：**
```
shape=[2, 4, 3], strides=[12, 3, 1]
  → 2张图, 高4, 3通道(RGB)

物理内存（逐像素排列，同一像素三通道连续）:
  b0h0w0: [R,G,B], b0h0w1: [R,G,B], ...b0h1w0: [R,G,B], ...
  flat_index(b,h,w,c) = b*12 + h*3 + w*1 + c*1
```

**CHWN 3维示例：**
```
shape=[3, 4, 2], strides=[8, 2, 1]
  → 3通道, 高4, 2批次

物理内存:
  c0: b0h0,b0h1,b0h2,b0h3, b1h0,b1h1,b1h2,b1h3  (8个float)
  c1: 同上，紧接在 c0 后
  c2: 同上

flat_index(c,h,b) = c*8 + h*2 + b*1
```

#### 4维张量：`[D0, D1, D2, D3]` — 最常用（卷积场景）

| 布局 | shape | strides | 物理含义 |
|------|-------|---------|----------|
| CL_NCHW | [N, C, H, W] | [C×H×W, H×W, W, 1] | PyTorch 格式 |
| CL_NHWC | [N, H, W, C] | [H×W×C, W×C, C, 1] | TensorFlow 格式 |
| CL_CHWN | [C, H, W, N] | [H×W×N, W×N, N, 1] | 通道优先 |

**CL_NCHW 4维示例：**
```
shape=[2, 3, 4, 5]  → 2张图, 3通道, 高4, 宽5
strides=[60, 20, 5, 1]

物理内存（按 flat_index）：
  index    内容
  0-4:     b0c0h0w0..w4
  5-9:     b0c0h1w0..w4
  ...
  15-19:   b0c0h3w0..w4   (ch0h0plane结束)
  20-39:   b0c1h0..h3    (ch1plane)
  40-59:   b0c2h0..h3    (ch2plane)
  60-119:  b1c0h0..h3..c2 (batch1)

flat_index(n,c,h,w) = n*60 + c*20 + h*5 + w*1
```

**CL_NHWC 4维示例（TensorFlow 默认）：**
```
shape=[2, 4, 5, 3]  → 2张图, 高4, 宽5, 3通道
strides=[60, 15, 3, 1]

物理内存（逐像素，同一像素通道值连续）：
  b0h0w0: R G B → indices 0,1,2
  b0h0w1: R G B → indices 3,4,5
  b0h0w2: R G B → indices 6,7,8
  ...
  b0h1w0: R G B → indices 15,16,17
  b1h0w0: R G B → indices 60,61,62

flat_index(n,h,w,c) = n*60 + h*15 + w*3 + c*1
```
**优势：** 同一像素的多个通道值连续，卷积计算时数据局部性更好。

**CL_CHWN 4维示例：**
```
shape=[3, 4, 5, 2]  → 3通道, 高4, 宽5, 2批次
strides=[40, 10, 2, 1]

物理内存（按通道展开）：
  c0: b0h0w0..b0h3w4, b1h0w0..b1h3w4   (40个float)
  c1: 同上，紧接在 c0 后 (40个float)
  c2: 同上 (40个float)

flat_index(c,h,w,n) = c*40 + h*10 + w*2 + n*1
```

#### 5维张量：`[D0, D1, D2, D3, D4]`

| 布局 | shape | strides | 典型应用 |
|------|-------|---------|----------|
| CL_NCHW | [B, C, D, H, W] | [C×D×H×W, D×H×W, H×W, W, 1] | 3D卷积/视频批次 |
| CL_NHWC | [B, D, H, W, C] | [D×H×W×C, H×W×C, W×C, C, 1] | TensorFlow 3D卷积 |
| CL_CHWN | [C, D, H, W, B] | [D×H×W×B, H×W×B, W×B, B, 1] | 通道优先3D卷积 |

**CL_NCHW 5维示例（视频批次 3D 卷积）：**
```
shape=[2, 3, 4, 5, 6]  → 2视频段, 3通道, 帧数4, 高5, 宽6
strides=[360, 120, 30, 6, 1]

flat_index(b,c,d,h,w) = b*360 + c*120 + d*30 + h*6 + w*1

物理内存排布：
  b0c0d0: h0w0..w5, h1w0..w5, h2w0..w5, h3w0..w5   (30个float)
  b0c0d1: h0w0..w5, ...                             (30个float)
  ...
  b0c0: 4个 depth planes (120个float)
  b0c1: 4个 depth planes (120个float)
  b0c2: 4个 depth planes (120个float)
  b1c0..b1c2: (batch1, 同上)
```

**各布局对比总结（4维）：**

```
设 shape=[N,C,H,W] = [2,3,4,5]
  CL_NCHW strides: [3*4*5=60, 4*5=20, 5, 1]
  CL_NHWC strides: [4*5*3=60, 5*3=15, 3, 1]
  CL_CHWN strides: [4*5*2=40, 5*2=10, 2, 1]

物理地址 = Σ(dim_index[i] × strides[i])

NCHW 优势: 同一通道的所有H×W数据连续，适合需要整通道数据的操作
NHWC 优势: 同一像素的通道值连续，卷积计算局部性好
CHWN 优势: 同一batch的所有通道连续，方便批处理
```

---

## 6. CLTensor 生命周期

### 6.1 创建空张量

```c
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
```

**流程：**
1. 分配 `CLTensor` 结构体
2. 复制 shape[]，计算 size 和 nbytes
3. 计算 strides
4. **调用 `clCreateBuffer` 在 GPU VRAM 上分配内存**（关键差异）
5. 保存 context 和 queue 引用（后续传输操作用）

### 6.2 从主机数据直接创建

```c
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
```

先分配 VRAM buffer，再通过 `clEnqueueWriteBuffer` 同步拷贝主机数据。

### 6.3 上传 / 下载数据

```c
static void cl_tensor_upload(CLTensor* t, const void* host_data) {
    cl_int err = clEnqueueWriteBuffer(cl->queue, t->buffer, CL_TRUE, 0, t->nbytes,
                                       host_data, 0, NULL, NULL);
    CL_CHECK(err, "Failed to upload to device");
}

static void cl_tensor_download(const CLTensor* t, void* host_data) {
    cl_int err = clEnqueueReadBuffer(cl->queue, t->buffer, CL_TRUE, 0, t->nbytes,
                                      host_data, 0, NULL, NULL);
    CL_CHECK(err, "Failed to download from device");
}
```

- `clEnqueueWriteBuffer`：主机 → GPU（异步，入队后立即返回）
- `clEnqueueReadBuffer`：GPU → 主机（同步，第四参数 `CL_TRUE` 阻塞等待完成）
- `clFinish` 确保 GPU 端操作已完成

### 6.4 释放张量

```c
static void cl_tensor_free(CLTensor* t) {
    if (!t) return;
    if (t->buffer) clReleaseMemObject(t->buffer);  // 释放 VRAM 内存
    free(t->shape);
    free(t->strides);
    free(t);
}
```

---

## 7. 核心 GPU 内核（OpenCL C 源码）

### 7.1 Fill（常量填充）

```c
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

    size_t global = ((t->size + 255) / 256) * 256;  // 256 对齐
    err = clEnqueueNDRangeKernel(cl->queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    CL_CHECK(err, "Failed to enqueue kernel");
    clFinish(cl->queue);

    clReleaseKernel(kernel);
    clReleaseProgram(prog);
}
```

**内核源码：**
```c
static const char* cl_kernel_fill_source =
"__kernel void fill_f32(__global float* output, float value, int size) {\n"
"    int gid = get_global_id(0);\n"
"    if (gid < size) {\n"
"        output[gid] = value;\n"
"    }\n"
"}\n";
```

**执行流程：**
```
主机端                     GPU端
  |--- clCreateProgramWithSource --→ 编译 OpenCL C 源码
  |--- clBuildProgram ------------→ 生成可执行内核
  |--- clCreateKernel ------------→ 创建 "fill_f32" 内核对象
  |--- clSetKernelArg ------------→ 设置参数: buffer指针, float值, size
  |--- clEnqueueNDRangeKernel --→ 入队执行 (256对齐全局工作项)
  |--- clFinish ------------------→ 等待 GPU 完成
```

### 7.2 ReLU 激活

```c
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
```

**内核源码：**
```c
static const char* cl_kernel_relu_source =
"__kernel void relu_f32(__global float* output,\n"
"                       __global const float* input, int size) {\n"
"    int gid = get_global_id(0);\n"
"    if (gid < size) {\n"
"        float v = input[gid];\n"
"        output[gid] = v > 0 ? v : 0;\n"
"    }\n"
"}\n";
```

**工作原理：**
- 每个 work-item（线程）处理一个元素：`gid = get_global_id(0)`
- `__global float*`：指向 VRAM 缓冲区的指针
- `__global const float*`：只读输入缓冲区
- `get_global_id(0)` 返回全局线程 ID，实现数据并行

### 7.3 GEMM（矩阵乘法）

```c
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
```

**内核源码：**
```c
static const char* cl_kernel_gemm_source =
"__kernel void gemm_f32(__global float* C,\n"
"                       __global const float* A, __global const float* B,\n"
"                       int M, int N, int K, float alpha, float beta) {\n"
"    int row = get_global_id(1);  // M 维度\n"
"    int col = get_global_id(0);  // N 维度\n"
"\n"
"    if (row < M && col < N) {\n"
"        float sum = 0.0f;\n"
"        for (int k = 0; k < K; k++) {\n"
"            sum += A[row * K + k] * B[k * N + col];\n"
"        }\n"
"        C[row * N + col] = alpha * sum + beta * C[row * N + col];\n"
"    }\n"
"}\n";
```

**执行配置解析：**
```
全局工作空间: {N_aligned, M_aligned} = {ceil(N/16)*16, ceil(M/16)*16}
本地工作组:   {16, 16}

工作项排布（2D Grid）:
            col=0   col=1   ... col=15  col=16 ...
  row=0:   (0,0)   (1,0)   ... (15,0)  (16,0)
  row=1:   (0,1)   (1,1)   ... (15,1)  (16,1)
  ...
  row=16:  (0,16)  (1,16)  ...

每个 work-item 处理 C[row][col] 的点积计算
行号 = get_global_id(1)，列号 = get_global_id(0)

C[row*N+col] = Σ(A[row][k] * B[k][col])  for k in [0,K)
```

**形状要求：**
```
a: shape=[M, K]   strides=[K, 1]
b: shape=[K, N]   strides=[N, 1]
output: shape=[M, N]

内存排布（假设 M=2, K=3, N=4）：
  A: [a00,a01,a02, a10,a11,a12]          strides=[3,1]
  B: [b00,b01,b02,b03, b10,b11,b12,b13, b20,b21,b22,b23] strides=[4,1]
  C: [c00,c01,c02,c03, c10,c11,c12,c13]  strides=[4,1]
```

---

## 8. 主机端回退运算

部分运算由于索引复杂或需要特殊归约，在主机端实现：

### 8.1 Softmax

```c
static CLTensor* cl_tensor_softmax(CLOpenCL* cl, const CLTensor* t, size_t axis) {
    CLTensor* output = cl_tensor_create(cl, t->dtype, t->layout, t->shape, t->ndim);
    if (!output) return NULL;

    float* h_data = (float*)malloc(t->nbytes);
    cl_tensor_download(t, h_data);  // GPU → 主机

    size_t stride = t->strides[axis];
    size_t block = t->shape[axis];
    size_t blocks = t->size / block;

    for (size_t b = 0; b < blocks; b++) {
        size_t base = b * block * stride;
        // 1. 找最大值
        float max_val = h_data[base];
        for (size_t i = 1; i < block; i++) {
            float v = h_data[base + i * stride];
            if (v > max_val) max_val = v;
        }
        // 2. 指数化并求和
        float sum = 0.0f;
        for (size_t i = 0; i < block; i++) {
            h_data[base + i * stride] = expf(h_data[base + i * stride] - max_val);
            sum += h_data[base + i * stride];
        }
        // 3. 归一化
        for (size_t i = 0; i < block; i++) {
            h_data[base + i * stride] /= sum;
        }
    }

    cl_tensor_upload(output, h_data);  // 主机 → GPU
    free(h_data);
    return output;
}
```

**Softmax 沿不同轴的索引示例：**
```
shape=[2, 3, 4], axis=1, strides=[12, 4, 1]
block = shape[1] = 3 (沿 axis=1 归约)
stride = strides[1] = 4
blocks = size / block = 24 / 3 = 8

对 b=0 (第1个batch):
  base = 0
  axis=1 方向的元素索引: base+0*4, base+1*4, base+2*4
    即 indices: [0,4,8] 对应 c0h0w0, c1h0w0, c2h0w0（同一空间位置的3通道）

对 b=1 (第2个batch):
  base = 12
  indices: [12,16,20] 对应 c0h0w0(b1), c1h0w0(b1), c2h0w0(b1)
```

**性能瓶颈：** 每次调用 softmax 需执行 `GPU→CPU` + 计算 + `CPU→GPU` 两次完整数据传输。

### 8.2 Element-wise Add（带广播）

```c
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
    cl_tensor_download(a, h_a);  // 下载 a 到主机
    cl_tensor_download(b, h_b);  // 下载 b 到主机

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
    cl_tensor_upload(output, h_out);  // 上传结果到 GPU
    free(h_a); free(h_b); free(h_out);
    return output;
}
```

**广播示例（与 CPU 版本一致）：**
```
a: shape=[3, 1, 4]  strides=[4, 4, 1]
b: shape=[1, 5, 4]  strides=[20, 4, 1]
输出: shape=[3, 5, 4]

对输出索引 [2, 3, 1]:
  A: effective_idx[0] = 2, effective_idx[1] = 0(broadcast), effective_idx[2] = 1
     idx_a = 2*4 + 0*4 + 1*1 = 9
  B: effective_idx[0] = 0(broadcast), effective_idx[1] = 3, effective_idx[2] = 1
     idx_b = 0*20 + 3*4 + 1*1 = 13
```

---

## 9. 错误处理

### 9.1 错误码转字符串

```c
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
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        default: return "Unknown";
    }
}
```

### 9.2 错误检查宏

```c
#define CL_CHECK(err, msg) \
    do { \
        if (err != CL_SUCCESS) { \
            fprintf(stderr, "OpenCL error %d: %s at %s:%d - %s\n", \
                    err, cl_get_error_string(err), __FILE__, __LINE__, msg); \
            exit(1); \
        } \
    } while (0)
```

### 9.3 内核构建日志

构建失败时获取并打印详细编译日志：
```c
size_t log_size;
clGetProgramBuildInfo(prog, cl->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
char* log = (char*)malloc(log_size);
clGetProgramBuildInfo(prog, cl->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
fprintf(stderr, "Build log:\n%s\n", log);
free(log);
```

常见构建错误：内核语法错误、不支持 `-cl-fast-relaxed-math`（设备不支持 fast math）等。

---

## 10. 与 CPU 版本对比

| 特性 | tensor.h (CPU) | opencl_tensor.h (GPU) |
|------|----------------|----------------------|
| 数据存储 | `void* data`（malloc/堆） | `cl_mem buffer`（VRAM） |
| 内存布局 | NCHW, NHWC, CHWN, OWI | NCHW, NHWC, CHWN |
| 数据类型 | 9种（含f64/UINT16） | 4种（f32/f16/int8/int32） |
| 量化支持 | 完整（affine + symmetric） | 暂不支持 |
| GEMM | 16×16 缓存分块（多层循环）| 16×16 2D tile（单内核循环）|
| 卷积 | Im2Col + blocked GEMM | 暂不支持 |
| 池化 | MaxPool + AvgPool | 暂不支持 |
| BatchNorm | 完整实现 | 暂不支持 |
| Softmax | 原位沿轴归约 | 主机端回退（含两次数据传输）|
| Add（广播）| 全广播索引计算 | 主机端回退（含两次数据传输）|
| 步长计算 | 区分 NHWC/NCHW | 统一行主序 |
| 内核编译 | N/A | 每次调用动态编译（开销较大）|
| 数据传输 | 直接内存访问 | 需 clEnqueueWriteBuffer/clEnqueueReadBuffer |

---

## 11. 使用示例

```c
// 初始化 OpenCL（使用 GPU）
CLOpenCL cl;
if (!cl_init(&cl, CL_DEVICE_TYPE_GPU)) {
    printf("Failed to initialize OpenCL\n");
    return;
}
cl_print_device_info(cl.device);

// 创建张量并从主机数据初始化
float host_a[128 * 256];
float host_b[256 * 64];
// ... 填充 host_a, host_b ...

size_t shape_a[] = {128, 256};
size_t shape_b[] = {256, 64};
CLTensor* a = cl_tensor_create_from_host(&cl, CL_TENSOR_DTYPE_F32,
                                          CL_TENSOR_LAYOUT_NCHW, shape_a, 2, host_a);
CLTensor* b = cl_tensor_create(&cl, CL_TENSOR_DTYPE_F32, CL_TENSOR_LAYOUT_NCHW, shape_b, 2);
cl_tensor_upload(b, host_b);  // 上传 b 到 GPU

// 矩阵乘法
CLTensor* c = cl_tensor_matmul(&cl, a, b);

// 下载结果
float result[128 * 64];
cl_tensor_download(c, result);

// 打印前几个结果
for (int i = 0; i < 5; i++) {
    printf("c[%d] = %.4f\n", i, result[i]);
}

// 清理
cl_tensor_free(a);
cl_tensor_free(b);
cl_tensor_free(c);
cl_release(&cl);
```

---

## 12. 当前限制

1. **无量化支持** — 缺少 INT8 量化张量的 GPU 运算
2. **无 BatchNorm** — 未实现 GPU 端的批量归一化
3. **Softmax/Add 需数据传输** — 复杂索引操作仍在 CPU 执行
4. **无卷积** — 未实现 Im2Col + GEMM 的 GPU 版本
5. **无池化** — 缺少 MaxPool/AvgPool 的 GPU 实现
6. **每次内核编译** — `clCreateProgramWithSource` 每运算调用一次，重复开销大
7. **无内核缓存** — 未实现编译后内核的持久化
8. **OWI 布局缺失** — CPU 版本支持的展平权重布局（OWI）在 GPU 版本中不支持
