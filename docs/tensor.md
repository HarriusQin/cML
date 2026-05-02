# CPU 张量数据结构与运算实现文档

本文档详述 `tensor.h` 中定义的张量（Tensor）数据结构及其各类运算的 CPU 实现。该实现为纯头文件形式（header-only），所有函数均为 `static inline` 实现。

---

## 1. 核心数据结构

### 1.1 Tensor 主结构体

```c
typedef struct {
    TensorDType dtype;     // 数据类型（f32/f16/f64/int8/int16/int32/int64/uint8/uint16）
    TensorLayout layout;  // 内存布局（NCHW/NHWC/CHWN/OWI）
    uint8_t ndim;         // 维度数
    size_t* shape;        // 各维度大小 [d0, d1, ..., dn-1]
    size_t* strides;      // 各维度步长（以元素为单位，非字节）
    void* data;           // 数据指针
    size_t size;          // 元素总数
    size_t capacity;      // 已分配的容量

    // 量化信息
    QuantType quant_type;
    QuantParams quant;
} Tensor;
```

**设计要点：**
- `strides` 以**元素为单位**而非字节，便于索引计算
- 支持量化（affine / symmetric 两种模式）
- `data` 指针可共享（`tensor_reshape` 不复制数据）
- `capacity` 用于区分重塑后的视图与真实分配

### 1.2 内存布局枚举

```c
typedef enum {
    TENSOR_LAYOUT_NCHW,  // [N, C, H, W] — PyTorch 默认布局
    TENSOR_LAYOUT_NHWC,  // [N, H, W, C] — TensorFlow 默认布局
    TENSOR_LAYOUT_CHWN,  // [C, H, W, N] — 通道优先，用于卷积优化
    TENSOR_LAYOUT_OWI,   // [Out, Win, In] — 展平的权重矩阵
} TensorLayout;
```

### 1.3 数据类型枚举

```c
typedef enum {
    TENSOR_DTYPE_F32, TENSOR_DTYPE_F16, TENSOR_DTYPE_F64,
    TENSOR_DTYPE_INT8, TENSOR_DTYPE_INT16, TENSOR_DTYPE_INT32, TENSOR_DTYPE_INT64,
    TENSOR_DTYPE_UINT8, TENSOR_DTYPE_UINT16,
} TensorDType;
```

### 1.4 量化参数

```c
typedef enum {
    QUANT_NONE,
    QUANT_AFFINE,       // y = scale * (x - zero_point)
    QUANT_SYMMETRIC,    // zero_point = 0
} QuantType;

typedef struct {
    float scale;
    int32_t zero_point;
} QuantParams;
```

---

## 2. 数据类型辅助函数

### 2.1 dtype_size — 数据类型字节数

```c
static inline size_t tensor_dtype_size(TensorDType dtype) {
    switch (dtype) {
        case TENSOR_DTYPE_F32:
        case TENSOR_DTYPE_INT32:  return 4;
        case TENSOR_DTYPE_F16:
        case TENSOR_DTYPE_INT16:  return 2;
        case TENSOR_DTYPE_F64:
        case TENSOR_DTYPE_INT64:  return 8;
        case TENSOR_DTYPE_INT8:
        case TENSOR_DTYPE_UINT8:  return 1;
        default: return 4;
    }
}
```

### 2.2 elem_count — 元素总数

```c
static inline size_t tensor_elem_count(const Tensor* t) {
    size_t count = 1;
    for (uint8_t i = 0; i < t->ndim; i++) count *= t->shape[i];
    return count;
}
```

### 2.3 nbytes — 总字节数

```c
static inline size_t tensor_nbytes(const Tensor* t) {
    return tensor_elem_count(t) * tensor_dtype_size(t->dtype);
}
```

---

## 3. 步长计算与内存布局详解

### 3.1 步长计算实现

```c
static void tensor_compute_strides(Tensor* t) {
    if (t->layout == TENSOR_LAYOUT_NHWC) {
        // NHWC: [N, H, W, C] -> strides = [H*W*C, W*C, C, 1]
        t->strides[3] = 1;
        t->strides[2] = t->shape[3];
        t->strides[1] = t->shape[2] * t->shape[3];
        t->strides[0] = t->shape[1] * t->shape[2] * t->shape[3];
    } else {
        // Default NCHW / row-major: strides[ndim-1] = 1, strides[i] = strides[i+1] * shape[i+1]
        t->strides[t->ndim - 1] = 1;
        for (int i = (int)t->ndim - 2; i >= 0; i--) {
            t->strides[i] = t->strides[i + 1] * t->shape[i + 1];
        }
    }
}
```

### 3.2 各维度内存布局详解

张量的内存是一维连续数组，通过 `shape` + `strides` 实现多维视图。

#### 1维张量：向量 `[D]`

```
shape = [5], strides = [1]
内存布局: [v0, v1, v2, v3, v4]
          0    1    2    3    4
索引映射: flat_index = indices[0]
```

#### 2维张量：矩阵 `[H, W]`

| 布局 | shape | strides | 内存布局（2×3矩阵）|
|------|-------|---------|--------------------|
| NCHW/行主序 | [2, 3] | [3, 1] | `[r0c0, r0c1, r0c2, r1c0, r1c1, r1c2]` |
| NHWC（不常见） | [2, 3] | [1, 2] | `[r0c0, r1c0, r0c1, r1c1, r0c2, r1c2]` |

**行主序（NCHW）示例：**
```
矩阵:
  [a, b, c]
  [d, e, f]

shape=[2,3], strides=[3,1]

flat_index(0,0) = 0*3 + 0*1 = 0  → a
flat_index(0,1) = 0*3 + 1*1 = 1  → b
flat_index(0,2) = 0*3 + 2*1 = 2  → c
flat_index(1,0) = 1*3 + 0*1 = 3  → d
flat_index(1,1) = 1*3 + 1*1 = 4  → e
flat_index(1,2) = 1*3 + 2*1 = 5  → f
```

**列主序（NHWC 示例）示例：**
```
shape=[2,3], strides=[1,2]

flat_index(0,0) = 0*1 + 0*2 = 0  → a
flat_index(0,1) = 0*1 + 1*2 = 2  → c
flat_index(0,2) = 0*1 + 2*2 = 4  → e
flat_index(1,0) = 1*1 + 0*2 = 1  → b
flat_index(1,1) = 1*1 + 1*2 = 3  → d
flat_index(1,2) = 1*1 + 2*2 = 5  → f
```

#### 3维张量：`[D0, D1, D2]`

| 布局 | shape | strides | 物理含义 |
|------|-------|---------|----------|
| NCHW | [2, 3, 4] | [12, 4, 1] | `[batch, channel, width]` |
| NHWC | [2, 3, 4] | [12, 4, 1] | `[batch, height, width]`（同 NCHW）|
| CHWN | [2, 3, 4] | [1, 2, 6] | `[channel, height, batch]`（特殊顺序）|

**NCHW 3维示例（图像批次）：**
```
shape=[2, 3, 4]  → 2张图像，每图像3通道，每通道4像素
strides=[12, 4, 1]

逻辑视图:
  batch=0: [ch0: [p0,p1,p2,p3], ch1: [p0,p1,p2,p3], ch2: [p0,p1,p2,p3]]
  batch=1: [ch0: [p0,p1,p2,p3], ch1: [p0,p1,p2,p3], ch2: [p0,p1,p2,p3]]

物理线性内存（先填满channel0，再填channel1...）:
  [b0ch0p0, b0ch0p1, b0ch0p2, b0ch0p3, b0ch1p0, ...b0ch2p3, b1ch0p0, ...]
       0        1        2        3        4      ...    11     12   ...

flat_index(d0,d1,d2) = d0*12 + d1*4 + d2*1
```

**CHWN 3维示例（通道优先）：**
```
shape=[2, 3, 4], strides=[1, 2, 6]

物理线性内存（先填batch0，再填batch1...）:
  [ch0b0, ch0b1, ch1b0, ch1b1, ch2b0, ch2b1, ch0b0(p2...), ...]
       0      1      2      3      4      5      6        ...

flat_index(d0,d1,d2) = d0*1 + d1*2 + d2*6
```

#### 4维张量：`[D0, D1, D2, D3]` — 最常见（卷积场景）

| 布局 | shape | strides | 物理含义 |
|------|-------|---------|----------|
| NCHW | [N, C, H, W] | [C×H×W, H×W, W, 1] | `[batch, channel, height, width]` |
| NHWC | [N, H, W, C] | [H×W×C, W×C, C, 1] | `[batch, height, width, channel]` |
| CHWN | [C, H, W, N] | [H×W×N, W×N, N, 1] | `[channel, height, width, batch]` |

**NCHW 4维示例（典型 Batch Conv）：**
```
shape=[2, 3, 4, 5]  → 2张图, 3通道, 高4宽5
strides=[60, 20, 5, 1]

逻辑: batch=0有两张图(ch0:[4×5],ch1:[4×5],ch2:[4×5])

物理内存排布（逐通道展开）：
  b0ch0: h0w0, h0w1, h0w2, h0w3, h0w4,
          h1w0, h1w1, h1w2, h1w3, h1w4,
          h2w0, h2w1, h2w2, h2w3, h2w4,
          h3w0, h3w1, h3w2, h3w3, h3w4   (共20个)
  b0ch1: ... (接在b0ch0后)
  b0ch2: ... (接在b0ch1后)
  b1ch0: ... (接在b0ch2后)

flat_index(n,c,h,w) = n*60 + c*20 + h*5 + w*1
```

**NHWC 4维示例（TensorFlow 格式）：**
```
shape=[2, 4, 5, 3]  → 2张图, 高4, 宽5, 3通道(RGB)
strides=[60, 15, 3, 1]

物理内存（逐像素展开）：
  b0h0w0: RGB (3个连续值)
  b0h0w1: RGB
  b0h0w2: RGB
  ...
  b0h0: ch0,ch1,ch2  (同一像素的3通道连续)
  b0h1: ch0,ch1,ch2
  ...
  b1h0: ch0,ch1,ch2  (batch=1)

flat_index(n,h,w,c) = n*60 + h*15 + w*3 + c*1
```
**优势：** 同一像素的所有通道值连续存储，卷积计算时局部性更好。

**CHWN 4维示例（某些硬件优化格式）：**
```
shape=[3, 4, 5, 2]  → 3通道, 高4, 宽5, 2批次
strides=[20, 5, 1, 60]

物理内存（逐通道展开，每通道含所有批次）：
  ch0: b0h0w0,b0h0w1...b0h3w4, b1h0w0...b1h3w4  (20个/batch × 2batch = 40个)
  ch1: ... (接在ch0后)
  ch2: ...

flat_index(c,h,w,n) = c*20 + h*5 + w*1 + n*60
```
**优势：** 每个通道的数据独立连续存储，适合卷积核直接读取。

#### 5维张量：`[D0, D1, D2, D3, D4]`

| 布局 | shape | strides | 物理含义 |
|------|-------|---------|----------|
| NCHW | [B, C, D, H, W] | [C×D×H×W, D×H×W, H×W, W, 1] | 3D卷积/视频批次 |
| NHWC | [B, D, H, W, C] | [D×H×W×C, H×W×C, W×C, C, 1] | 3D视频 TensorFlow 格式 |
| CHWN | [C, D, H, W, B] | [D×H×W×B, H×W×B, W×B, B, 1] | 通道优先 3D 卷积 |

**NCHW 5维示例（视频批次，批量3D卷积）：**
```
shape=[2, 3, 4, 5, 6]  → 2个视频, 3通道, 每帧高4宽5, 帧数6
strides=[360, 120, 30, 6, 1]

flat_index(b,c,d,h,w) = b*360 + c*120 + d*30 + h*6 + w*1
```

**NHWC 5维示例（视频 TensorFlow 格式）：**
```
shape=[2, 4, 5, 6, 3]  → 2视频, 帧数4, 高5, 宽6, 3通道
strides=[360, 90, 18, 3, 1]

flat_index(b,d,h,w,c) = b*360 + d*90 + h*18 + w*3 + c*1
```

**各布局对比总结（4维 NCHW/NHWC/CHWN）：**

```
设 shape=[N,C,H,W]
  NCHW strides = [C*H*W, H*W, W, 1]
  NHWC strides = [H*W*C, W*C, C, 1]
  CHWN strides = [H*W*N, W*N, N, 1]

物理地址计算：
  NCHW: addr = n*(C*H*W) + c*(H*W) + h*W + w
  NHWC: addr = n*(H*W*C) + h*(W*C) + w*C + c
  CHWN: addr = c*(H*W*N) + h*(W*N) + w*N + n
```

---

## 4. 生命周期管理

### 4.1 创建张量

```c
static Tensor* tensor_create(TensorDType dtype, TensorLayout layout,
                             const size_t* shape, uint8_t ndim) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
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
    t->capacity = t->size;
    tensor_compute_strides(t);

    size_t bytes = t->size * tensor_dtype_size(dtype);
    t->data = malloc(bytes);
    memset(t->data, 0, bytes);

    t->quant_type = QUANT_NONE;
    t->quant.scale = 1.0f;
    t->quant.zero_point = 0;
    return t;
}
```

### 4.2 释放张量

```c
static void tensor_free(Tensor* t) {
    if (!t) return;
    free(t->shape);
    free(t->strides);
    free(t->data);
    free(t);
}
```

### 4.3 索引操作

```c
// 多维索引 → 线性偏移量
static inline size_t tensor_flat_index(const Tensor* t, const size_t* indices) {
    size_t idx = 0;
    for (uint8_t i = 0; i < t->ndim; i) idx += indices[i] * t->strides[i];
    return idx;
}

// 按多维索引获取元素指针
static inline void* tensor_at(const Tensor* t, const size_t* indices) {
    return (uint8_t*)t->data + tensor_flat_index(t, indices) * tensor_dtype_size(t->dtype);
}

// 读取 float 值（自动类型转换）
static inline float tensor_get_f32(const Tensor* t, const size_t* indices) {
    size_t idx = tensor_flat_index(t, indices);
    switch (t->dtype) {
        case TENSOR_DTYPE_F32: return ((float*)t->data)[idx];
        case TENSOR_DTYPE_INT32: return (float)((int32_t*)t->data)[idx];
        case TENSOR_DTYPE_INT8: return (float)((int8_t*)t->data)[idx];
        case TENSOR_DTYPE_UINT8: return (float)((uint8_t*)t->data)[idx];
        default: return 0.0f;
    }
}

// 写入 float 值（自动类型转换）
static inline void tensor_set_f32(Tensor* t, const size_t* indices, float val) {
    size_t idx = tensor_flat_index(t, indices);
    switch (t->dtype) {
        case TENSOR_DTYPE_F32: ((float*)t->data)[idx] = val; break;
        case TENSOR_DTYPE_INT32: ((int32_t*)t->data)[idx] = (int32_t)val; break;
        case TENSOR_DTYPE_INT8: ((int8_t*)t->data)[idx] = (int8_t)val; break;
        case TENSOR_DTYPE_UINT8: ((uint8_t*)t->data)[idx] = (uint8_t)val; break;
        default: break;
    }
}
```

### 4.4 变形（共享数据）

```c
static Tensor* tensor_reshape(const Tensor* t, const size_t* new_shape, uint8_t ndim) {
    size_t new_size = 1;
    for (uint8_t i = 0; i < ndim; i++) new_size *= new_shape[i];
    if (new_size != t->size) return NULL;  // 元素总数必须相等

    Tensor* r = (Tensor*)malloc(sizeof(Tensor));
    r->dtype = t->dtype;
    r->layout = t->layout;
    r->ndim = ndim;
    r->shape = (size_t*)malloc(sizeof(size_t) * ndim);
    r->strides = (size_t*)malloc(sizeof(size_t) * ndim);
    memcpy(r->shape, new_shape, sizeof(size_t) * ndim);
    r->size = t->size;
    r->capacity = t->capacity;
    r->data = t->data;           // 共享同一数据指针，不复制
    r->quant_type = t->quant_type;
    r->quant = t->quant;
    tensor_compute_strides(r);
    return r;
}
```

### 4.5 克隆（深拷贝）

```c
static Tensor* tensor_clone(const Tensor* t) {
    Tensor* c = tensor_create(t->dtype, t->layout, t->shape, t->ndim);
    memcpy(c->data, t->data, tensor_nbytes(t));
    c->quant_type = t->quant_type;
    c->quant = t->quant;
    return c;
}
```

### 4.6 转置

```c
static Tensor* tensor_transpose(const Tensor* t, uint8_t axis0, uint8_t axis1) {
    if (axis0 >= t->ndim || axis1 >= t->ndim) return NULL;

    Tensor* r = tensor_create(t->dtype, t->layout, t->shape, t->ndim);
    size_t elem_size = tensor_dtype_size(t->dtype);
    uint8_t* src = (uint8_t*)t->data;
    uint8_t* dst = (uint8_t*)r->data;

    // 对每个输出元素，找到对应的源数据并复制
    size_t idx[8];
    for (size_t i = 0; i < r->size; i++) {
        size_t remaining = i;
        for (uint8_t d = 0; d < t->ndim; d++) {
            idx[d] = remaining % r->shape[d];
            remaining /= r->shape[d];
        }
        // 交换轴得到源索引
        size_t src_idx0 = idx[axis0]; idx[axis0] = idx[axis1]; idx[axis1] = src_idx0;
        size_t src_flat = 0;
        for (uint8_t d = 0; d < t->ndim; d++) {
            src_flat += idx[d] * t->strides[d];
        }
        memcpy(dst + i * elem_size, src + src_flat * elem_size, elem_size);
    }

    // 交换 shape 和 strides
    size_t tmp = r->shape[axis0]; r->shape[axis0] = r->shape[axis1]; r->shape[axis1] = tmp;
    tmp = r->strides[axis0]; r->strides[axis0] = r->strides[axis1]; r->strides[axis1] = tmp;
    return r;
}
```

### 4.7 切片

```c
static Tensor* tensor_slice(const Tensor* t, size_t dim, size_t start, size_t end) {
    if (dim >= t->ndim || start >= t->shape[dim] || end > t->shape[dim] || start >= end) return NULL;

    Tensor* r = tensor_create(t->dtype, t->layout, t->shape, t->ndim);
    r->shape[dim] = end - start;
    r->size = tensor_elem_count(r);
    r->capacity = t->capacity;

    size_t elem_size = tensor_dtype_size(t->dtype);
    size_t offset = start * t->strides[dim];
    r->data = (uint8_t*)t->data + offset * elem_size;  // 偏移指针，共享数据

    r->quant_type = t->quant_type;
    r->quant = t->quant;
    return r;
}
```

---

## 5. 元素级运算（Element-wise）

### 5.1 广播机制

```c
// 判断两个张量是否可以广播，并计算输出形状
static bool tensor_broadcast_shape(const Tensor* a, const Tensor* b,
                                   size_t* out_shape, uint8_t* out_ndim) {
    uint8_t max_ndim = (a->ndim > b->ndim) ? a->ndim : b->ndim;
    uint8_t offset_a = max_ndim - a->ndim;
    uint8_t offset_b = max_ndim - b->ndim;

    for (uint8_t i = 0; i < max_ndim; i++) {
        size_t dim_a = (i < offset_a) ? 1 : a->shape[i - offset_a];
        size_t dim_b = (i < offset_b) ? 1 : b->shape[i - offset_b];
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) return false;
        out_shape[i] = (dim_a > dim_b) ? dim_a : dim_b;
    }
    *out_ndim = max_ndim;
    return true;
}

// 计算广播后的线性索引（尺寸为1的维度取索引0）
static size_t tensor_get_broadcast_index(const Tensor* t, const size_t* indices,
                                         uint8_t offset, uint8_t t_ndim) {
    size_t idx = 0;
    for (uint8_t i = 0; i < t_ndim; i++) {
        size_t effective_idx = (t->shape[i] == 1) ? 0 : indices[i + offset];
        idx += effective_idx * t->strides[i];
    }
    return idx;
}
```

**广播示例：**
```
A: shape=[3, 1, 4]  → strides=[4, 4, 1] (offset=0)
B: shape=[1, 5, 4]  → strides=[20, 4, 1] (offset=0)
输出: shape=[3, 5, 4]

对输出索引 [2, 3, 1]:
  A: idx = 2*4 + 3*4 + 1*1 =  8+12+1 = 21  (第3行第4列，实际为第2行)
  B: idx = 2*20 + 3*4 + 1*1 = 40+12+1 = 53  (第3行第4列，同上)
```

### 5.2 加法（支持广播）

```c
static Tensor* tensor_add(const Tensor* a, const Tensor* b) {
    size_t out_shape[8];
    uint8_t out_ndim;
    if (!tensor_broadcast_shape(a, b, out_shape, &out_ndim)) return NULL;

    Tensor* c = tensor_create(a->dtype, a->layout, out_shape, out_ndim);
    float* ca = (float*)c->data;
    uint8_t offset_a = out_ndim - a->ndim;
    uint8_t offset_b = out_ndim - b->ndim;

    size_t total = c->size;
    size_t idx_buf[8] = {0};

    for (size_t i = 0; i < total; i++) {
        size_t temp = i;
        for (uint8_t j = out_ndim; j > 0; j--) {
            idx_buf[j-1] = temp % out_shape[j-1];
            temp /= out_shape[j-1];
        }
        size_t idx_a = tensor_get_broadcast_index(a, idx_buf, offset_a, a->ndim);
        size_t idx_b = tensor_get_broadcast_index(b, idx_buf, offset_b, b->ndim);
        ca[i] = ((float*)a->data)[idx_a] + ((float*)b->data)[idx_b];
    }
    return c;
}
```

### 5.3 减法/乘法/除法

```c
static Tensor* tensor_sub(const Tensor* a, const Tensor* b) {
    size_t out_shape[8]; uint8_t out_ndim;
    if (!tensor_broadcast_shape(a, b, out_shape, &out_ndim)) return NULL;
    Tensor* c = tensor_create(a->dtype, a->layout, out_shape, out_ndim);
    float* ca = (float*)c->data;
    uint8_t offset_a = out_ndim - a->ndim, offset_b = out_ndim - b->ndim;
    size_t total = c->size; size_t idx_buf[8] = {0};
    for (size_t i = 0; i < total; i++) {
        size_t temp = i;
        for (uint8_t j = out_ndim; j > 0; j--) { idx_buf[j-1] = temp % out_shape[j-1]; temp /= out_shape[j-1]; }
        ca[i] = ((float*)a->data)[tensor_get_broadcast_index(a, idx_buf, offset_a, a->ndim)]
              - ((float*)b->data)[tensor_get_broadcast_index(b, idx_buf, offset_b, b->ndim)];
    }
    return c;
}

static Tensor* tensor_mul(const Tensor* a, const Tensor* b) {
    size_t out_shape[8]; uint8_t out_ndim;
    if (!tensor_broadcast_shape(a, b, out_shape, &out_ndim)) return NULL;
    Tensor* c = tensor_create(a->dtype, a->layout, out_shape, out_ndim);
    float* ca = (float*)c->data;
    uint8_t offset_a = out_ndim - a->ndim, offset_b = out_ndim - b->ndim;
    size_t total = c->size; size_t idx_buf[8] = {0};
    for (size_t i = 0; i < total; i++) {
        size_t temp = i;
        for (uint8_t j = out_ndim; j > 0; j--) { idx_buf[j-1] = temp % out_shape[j-1]; temp /= out_shape[j-1]; }
        ca[i] = ((float*)a->data)[tensor_get_broadcast_index(a, idx_buf, offset_a, a->ndim)]
              * ((float*)b->data)[tensor_get_broadcast_index(b, idx_buf, offset_b, b->ndim)];
    }
    return c;
}

static Tensor* tensor_div(const Tensor* a, const Tensor* b) {
    size_t out_shape[8]; uint8_t out_ndim;
    if (!tensor_broadcast_shape(a, b, out_shape, &out_ndim)) return NULL;
    Tensor* c = tensor_create(a->dtype, a->layout, out_shape, out_ndim);
    float* ca = (float*)c->data;
    uint8_t offset_a = out_ndim - a->ndim, offset_b = out_ndim - b->ndim;
    size_t total = c->size; size_t idx_buf[8] = {0};
    for (size_t i = 0; i < total; i++) {
        size_t temp = i;
        for (uint8_t j = out_ndim; j > 0; j--) { idx_buf[j-1] = temp % out_shape[j-1]; temp /= out_shape[j-1]; }
        ca[i] = ((float*)a->data)[tensor_get_broadcast_index(a, idx_buf, offset_a, a->ndim)]
              / (((float*)b->data)[tensor_get_broadcast_index(b, idx_buf, offset_b, b->ndim)] + 1e-8f);
    }
    return c;
}
```

### 5.4 原地加法 / 标度

```c
static void tensor_add_inplace(Tensor* a, const Tensor* b) {
    size_t out_shape[8]; uint8_t out_ndim;
    if (!tensor_broadcast_shape(a, b, out_shape, &out_ndim)) return;
    uint8_t offset_b = out_ndim - b->ndim;
    size_t idx_buf[8] = {0};
    for (size_t i = 0; i < a->size; i++) {
        size_t temp = i;
        for (uint8_t j = a->ndim; j > 0; j--) { idx_buf[j-1] = temp % a->shape[j-1]; temp /= a->shape[j-1]; }
        size_t out_idx_buf[8] = {0};
        for (uint8_t j = 0; j < a->ndim; j++) out_idx_buf[offset_b + j] = idx_buf[j];
        ((float*)a->data)[i] += ((float*)b->data)[tensor_get_broadcast_index(b, out_idx_buf, offset_b, b->ndim)];
    }
}

static void tensor_scale(Tensor* t, float scalar) {
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->size; i++) data[i] *= scalar;
}
```

---

## 6. 矩阵乘法（GEMM）

### 6.1 分块 GEMM（缓存优化）

```c
#define TENSOR_GEMM_BLOCK 16

static void tensor_gemm(float* C, const float* A, const float* B,
                        size_t M, size_t N, size_t K,
                        float alpha, float beta) {
    // C = alpha * A @ B + beta * C
    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++)
            C[i*N + j] *= beta;

    for (size_t i = 0; i < M; i += TENSOR_GEMM_BLOCK) {
        for (size_t k = 0; k < K; k += TENSOR_GEMM_BLOCK) {
            size_t ib = (i + TENSOR_GEMM_BLOCK < M) ? TENSOR_GEMM_BLOCK : M - i;
            size_t kb = (k + TENSOR_GEMM_BLOCK < K) ? TENSOR_GEMM_BLOCK : K - k;
            for (size_t ii = 0; ii < ib; ii++) {
                for (size_t kk = 0; kk < kb; kk++) {
                    float a_ik = A[(i + ii) * K + (k + kk)];
                    const float* B_row = B + (k + kk) * N;
                    float* C_row = C + (i + ii) * N;
                    for (size_t j = 0; j < N; j++) {
                        C_row[j] += alpha * a_ik * B_row[j];
                    }
                }
            }
        }
    }
}
```

**分块策略解析：**

```
A[M×K]  B[K×N]  C[M×N]

A按行分块，B按列分块：
  A: | A00 | A01 |
      |-----|-----|
  B: | B00 |     |
      |-----|
  C: | A00@B00 | A01@B10 |
      |---------|----------|

内存顺序（M=64, K=64, N=64, BLOCK=16）：
  最外层: i(0,16,32,48) → k(0,16,32,48) → ii(0..15) → kk(0..15) → j(0..63)
  最热数据: A[i:ib, k:kb] 和 B[k:kb, :] 共同组成 ib×kb 的小GEMM
  L1缓存: 16×16×4B×3 ≈ 3KB < 32KB L1
```

### 6.2 高层矩阵乘法（自动批处理）

```c
static Tensor* tensor_matmul(const Tensor* a, const Tensor* b) {
    if (a->ndim < 2 || b->ndim < 2) return NULL;
    bool a_has_batch = (a->ndim > 2);
    bool b_has_batch = (b->ndim > 2);
    size_t M, K, N, B;

    if (!a_has_batch && !b_has_batch) {
        // [M,K] @ [K,N] → [M,N]
        M = a->shape[0]; K = a->shape[1];
        N = b->shape[1];
        if (b->shape[0] != K) return NULL;
        size_t shape[] = {M, N};
        Tensor* c = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 2);
        tensor_gemm((float*)c->data, (float*)a->data, (float*)b->data, M, N, K, 1.0f, 0.0f);
        return c;
    } else if (a_has_batch && !b_has_batch) {
        // [B,M,K] @ [K,N] → [B,M,N]
        B = a->shape[0]; M = a->shape[1]; K = a->shape[2];
        N = b->shape[1];
        if (b->shape[0] != K) return NULL;
        size_t out_shape[] = {B, M, N};
        Tensor* c = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, out_shape, 3);
        float* c_data = (float*)c->data;
        float* a_data = (float*)a->data;
        float* b_data = (float*)b->data;
        for (size_t batch = 0; batch < B; batch++) {
            tensor_gemm(c_data + batch * M * N, a_data + batch * M * K, b_data, M, N, K, 1.0f, 0.0f);
        }
        return c;
    } else if (!a_has_batch && b_has_batch) {
        // [M,K] @ [B,K,N] → [B,M,N]
        M = a->shape[0]; K = a->shape[1];
        B = b->shape[0]; N = b->shape[2];
        if (b->shape[1] != K) return NULL;
        size_t out_shape[] = {B, M, N};
        Tensor* c = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, out_shape, 3);
        float* c_data = (float*)c->data;
        float* a_data = (float*)a->data;
        float* b_data = (float*)b->data;
        for (size_t batch = 0; batch < B; batch++) {
            tensor_gemm(c_data + batch * M * N, a_data, b_data + batch * K * N, M, N, K, 1.0f, 0.0f);
        }
        return c;
    } else {
        // [B,M,K] @ [B,K,N] → [B,M,N]
        if (a->ndim != 3 || b->ndim != 3) return NULL;
        B = a->shape[0]; M = a->shape[1]; K = a->shape[2];
        if (b->shape[0] != B || b->shape[1] != K) return NULL;
        N = b->shape[2];
        size_t out_shape[] = {B, M, N};
        Tensor* c = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, out_shape, 3);
        float* c_data = (float*)c->data;
        float* a_data = (float*)a->data;
        float* b_data = (float*)b->data;
        for (size_t batch = 0; batch < B; batch++) {
            tensor_gemm(c_data + batch * M * N, a_data + batch * M * K, b_data + batch * K * N, M, N, K, 1.0f, 0.0f);
        }
        return c;
    }
}
```

---

## 7. 归约运算（Reduction）

### 7.1 Sum

```c
static Tensor* tensor_sum(const Tensor* t, size_t axis) {
    if (axis >= t->ndim) return NULL;
    size_t* new_shape = (size_t*)malloc(sizeof(size_t) * (t->ndim - 1));
    size_t new_ndim = 0;
    for (size_t i = 0; i < t->ndim; i++) {
        if (i != axis) new_shape[new_ndim++] = t->shape[i];
    }
    Tensor* result = tensor_create(t->dtype, t->layout, new_shape, new_ndim);
    float* dst = (float*)result->data;
    float* src = (float*)t->data;

    size_t outer_size = 1;
    for (size_t i = 0; i < t->ndim; i++) {
        if (i != axis) outer_size *= t->shape[i];
    }
    size_t stride = t->strides[axis];
    size_t block = t->shape[axis];

    for (size_t outer = 0; outer < outer_size; outer++) {
        float sum = 0.0f;
        size_t base = 0;
        size_t temp = outer;
        for (size_t d = 0; d < t->ndim; d++) {
            if (d == axis) continue;
            size_t dim_size = (d == 0) ? t->strides[0] : t->strides[d-1] / t->strides[d];
            size_t idx = temp % dim_size;
            temp /= dim_size;
            base += idx * t->strides[d];
        }
        for (size_t i = 0; i < block; i++) {
            sum += src[base + i * stride];
        }
        dst[outer] = sum;
    }
    free(new_shape);
    return result;
}
```

### 7.2 Mean / Max

```c
static Tensor* tensor_mean(const Tensor* t, size_t axis) {
    if (axis >= t->ndim) return NULL;
    Tensor* result = tensor_sum(t, axis);
    float* data = (float*)result->data;
    size_t n = t->shape[axis];
    for (size_t i = 0; i < result->size; i++) data[i] /= n;
    return result;
}

static Tensor* tensor_max(const Tensor* t, size_t axis) {
    if (axis >= t->ndim) return NULL;
    size_t* new_shape = (size_t*)malloc(sizeof(size_t) * (t->ndim - 1));
    size_t new_ndim = 0;
    for (size_t i = 0; i < t->ndim; i++) {
        if (i != axis) new_shape[new_ndim++] = t->shape[i];
    }
    Tensor* result = tensor_create(t->dtype, t->layout, new_shape, new_ndim);
    float* dst = (float*)result->data;
    float* src = (float*)t->data;
    size_t outer_size = 1;
    for (size_t i = 0; i < t->ndim; i++) { if (i != axis) outer_size *= t->shape[i]; }
    size_t stride = t->strides[axis];
    size_t block = t->shape[axis];
    for (size_t outer = 0; outer < outer_size; outer++) {
        size_t base = 0;
        size_t temp = outer;
        for (size_t d = 0; d < t->ndim; d++) {
            if (d == axis) continue;
            size_t dim_size = (d == 0) ? t->strides[0] : t->strides[d-1] / t->strides[d];
            size_t idx = temp % dim_size;
            temp /= dim_size;
            base += idx * t->strides[d];
        }
        float max_val = -FLT_MAX;
        for (size_t i = 0; i < block; i++) {
            float v = src[base + i * stride];
            if (v > max_val) max_val = v;
        }
        dst[outer] = max_val;
    }
    free(new_shape);
    return result;
}
```

---

## 8. 激活函数

```c
static void tensor_relu(Tensor* t) {
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->size; i++) if (data[i] < 0) data[i] = 0;
}

static void tensor_sigmoid(Tensor* t) {
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->size; i++) data[i] = 1.0f / (1.0f + expf(-data[i]));
}

static void tensor_tanh(Tensor* t) {
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->size; i++) data[i] = tanhf(data[i]);
}

static void tensor_softmax(Tensor* t, size_t axis) {
    float* data = (float*)t->data;
    size_t stride = t->strides[axis];
    size_t block = t->shape[axis];
    size_t blocks = t->size / block;
    for (size_t b = 0; b < blocks; b++) {
        size_t base = b * block * stride;
        float max_val = data[base];
        for (size_t i = 1; i < block; i++) {
            float v = data[base + i * stride];
            if (v > max_val) max_val = v;
        }
        float sum = 0.0f;
        for (size_t i = 0; i < block; i++) {
            data[base + i * stride] = expf(data[base + i * stride] - max_val);
            sum += data[base + i * stride];
        }
        for (size_t i = 0; i < block; i++) {
            data[base + i * stride] /= sum;
        }
    }
}
```

**softmax 沿不同轴的索引计算示例：**
```
shape=[2, 3, 4], axis=1 (在通道轴归约)
strides = [12, 4, 1]
block = shape[1] = 3
stride = strides[1] = 4
blocks = size / block = 24 / 3 = 8

对 b=0 (第1个batch):
  base = 0 * 12 = 0
  base到 base+3*4 对应 [ch0h0w0, ch0h0w1, ch0h0w2, ch0h0w3,
                       ch1h0w0, ch1h0w1, ch1h0w2, ch1h0w3,
                       ch2h0w0, ch2h0w1, ch2h0w2, ch2h0w3]
  索引: base + i*stride 对应 ch_i 的所有 H×W 平面
```

---

## 9. 卷积（Im2Col）

### 9.1 卷积参数结构体

```c
typedef struct {
    size_t stride_h;
    size_t stride_w;
    size_t pad_h;
    size_t pad_w;
    size_t dilation_h;
    size_t dilation_w;
} Conv2DParams;
```

### 9.2 Im2Col + GEMM 卷积实现

```c
static Tensor* tensor_conv2d(const Tensor* input, const Tensor* weight,
                             const Conv2DParams* params) {
    // input: [N, C, H, W]
    // weight: [OutC, C, KH, KW]
    if (input->ndim != 4 || weight->ndim != 4) return NULL;
    size_t N = input->shape[0];
    size_t C = input->shape[1];
    size_t H = input->shape[2];
    size_t W = input->shape[3];
    size_t OutC = weight->shape[0];
    size_t KH = weight->shape[2];
    size_t KW = weight->shape[3];
    size_t stride = params ? params->stride_h : 1;
    size_t pad = params ? params->pad_h : 0;
    size_t H_out = (H + 2 * pad - KH) / stride + 1;
    size_t W_out = (W + 2 * pad - KW) / stride + 1;

    size_t out_shape[] = {N, OutC, H_out, W_out};
    Tensor* output = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, out_shape, 4);
    float* out_data = (float*)output->data;
    float* in_data = (float*)input->data;
    float* w_data = (float*)weight->data;

    // Im2Col: 每个 patch 展平为列
    size_t col_h = KH * KW * C;
    size_t col_w = H_out * W_out;
    float* col = (float*)malloc(sizeof(float) * col_h * col_w);

    for (size_t n = 0; n < N; n++) {
        // Im2Col
        for (size_t h = 0; h < H_out; h++) {
            for (size_t w = 0; w < W_out; w++) {
                size_t col_idx = h * W_out + w;
                size_t in_h_base = h * stride;
                size_t in_w_base = w * stride;
                size_t idx = 0;
                for (size_t kh = 0; kh < KH; kh++) {
                    for (size_t kw = 0; kw < KW; kw++) {
                        for (size_t c = 0; c < C; c++) {
                            size_t src_h = in_h_base + kh - pad;
                            size_t src_w = in_w_base + kw - pad;
                            if (src_h < H && src_w < W && src_h >= 0 && src_w >= 0) {
                                col[idx * col_w + col_idx] = in_data[n*C*H*W + c*H*W + src_h*W + src_w];
                            } else {
                                col[idx * col_w + col_idx] = 0.0f;
                            }
                            idx++;
                        }
                    }
                }
            }
        }
        // GEMM: [OutC, KH*KW*C] @ [KH*KW*C, H_out*W_out] → [OutC, H_out*W_out]
        for (size_t oc = 0; oc < OutC; oc++) {
            for (size_t hw = 0; hw < H_out * W_out; hw++) {
                float sum = 0.0f;
                for (size_t c = 0; c < KH * KW * C; c++) {
                    sum += w_data[oc * KH * KW * C + c] * col[c * col_w + hw];
                }
                out_data[n * OutC * H_out * W_out + oc * H_out * W_out + hw] = sum;
            }
        }
    }
    free(col);
    return output;
}
```

**Im2Col 数据重排示例（简化 1×1 卷积）：**
```
输入: shape=[1, 2, 2, 2] (N=1,C=2,H=2,W=2), KH=1,KW=1, stride=1, pad=0
H_out=(2+0-1)/1+1=2, W_out=2

输入数据:
  ch0: [a,b; c,d]   ch1: [e,f; g,h]
  flat: [a,b,c,d, e,f,g,h]

Im2Col 产生的 col 矩阵: col_h = 1*1*2 = 2, col_w = 2*2 = 4

  patch(h=0,w=0): [a,e]  → col[:, 0]
  patch(h=0,w=1): [b,f]  → col[:, 1]
  patch(h=1,w=0): [c,g]  → col[:, 2]
  patch(h=1,w=1): [d,h]  → col[:, 3]

col = [[a, b, c, d],   (ch0 四个 patch)
        [e, f, g, h]]  (ch1 四个 patch)
```

---

## 10. 池化

### 10.1 MaxPool

```c
static Tensor* tensor_maxpool2d(const Tensor* input, size_t pool_h, size_t pool_w,
                                 size_t stride_h, size_t stride_w) {
    if (input->ndim != 4) return NULL;
    size_t N = input->shape[0], C = input->shape[1];
    size_t H = input->shape[2], W = input->shape[3];
    size_t H_out = (H - pool_h) / stride_h + 1;
    size_t W_out = (W - pool_w) / stride_w + 1;
    size_t out_shape[] = {N, C, H_out, W_out};
    Tensor* output = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, out_shape, 4);
    float* out_data = (float*)output->data;
    float* in_data = (float*)input->data;
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H_out; h++) {
                for (size_t w = 0; w < W_out; w++) {
                    float max_val = -FLT_MAX;
                    for (size_t ph = 0; ph < pool_h; ph++) {
                        for (size_t pw = 0; pw < pool_w; pw++) {
                            size_t src_h = h * stride_h + ph;
                            size_t src_w = w * stride_w + pw;
                            if (src_h < H && src_w < W) {
                                float val = in_data[n*C*H*W + c*H*W + src_h*W + src_w];
                                if (val > max_val) max_val = val;
                            }
                        }
                    }
                    out_data[n*C*H_out*W_out + c*H_out*W_out + h*W_out + w] = max_val;
                }
            }
        }
    }
    return output;
}
```

### 10.2 AvgPool

```c
static Tensor* tensor_avgpool2d(const Tensor* input, size_t pool_h, size_t pool_w,
                                size_t stride_h, size_t stride_w) {
    if (input->ndim != 4) return NULL;
    size_t N = input->shape[0], C = input->shape[1];
    size_t H = input->shape[2], W = input->shape[3];
    size_t H_out = (H - pool_h) / stride_h + 1;
    size_t W_out = (W - pool_w) / stride_w + 1;
    size_t out_shape[] = {N, C, H_out, W_out};
    Tensor* output = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, out_shape, 4);
    float* out_data = (float*)output->data;
    float* in_data = (float*)input->data;
    float inv_n = 1.0f / (pool_h * pool_w);
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H_out; h++) {
                for (size_t w = 0; w < W_out; w++) {
                    float sum = 0.0f;
                    for (size_t ph = 0; ph < pool_h; ph++) {
                        for (size_t pw = 0; pw < pool_w; pw++) {
                            size_t src_h = h * stride_h + ph;
                            size_t src_w = w * stride_w + pw;
                            if (src_h < H && src_w < W) {
                                sum += in_data[n*C*H*W + c*H*W + src_h*W + src_w];
                            }
                        }
                    }
                    out_data[n*C*H_out*W_out + c*H_out*W_out + h*W_out + w] = sum * inv_n;
                }
            }
        }
    }
    return output;
}
```

---

## 11. 归一化

### 11.1 BatchNorm 训练阶段

```c
static void tensor_batch_norm_train(const Tensor* x, Tensor* y,
                                    const Tensor* gamma, const Tensor* beta,
                                    Tensor* mean, Tensor* var,
                                    Tensor* running_mean, Tensor* running_var,
                                    float momentum, float eps) {
    size_t N = x->shape[0], C = x->shape[1];
    size_t H = (x->ndim == 4) ? x->shape[2] : 1;
    size_t W = (x->ndim == 4) ? x->shape[3] : 1;
    size_t HW = H * W;
    float* x_data = (float*)x->data;
    float* y_data = (float*)y->data;
    float* mean_data = (float*)mean->data;
    float* var_data = (float*)var->data;
    float* gamma_data = (float*)gamma->data;
    float* beta_data = (float*)beta->data;

    // 计算每个通道的均值
    for (size_t c = 0; c < C; c++) {
        float sum = 0.0f;
        for (size_t n = 0; n < N; n++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    size_t idx = n * C * HW + c * HW + h * W + w;
                    sum += x_data[idx];
                }
            }
        }
        mean_data[c] = sum / (N * HW);
    }

    // 计算每个通道的方差
    for (size_t c = 0; c < C; c++) {
        float sum = 0.0f;
        for (size_t n = 0; n < N; n++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    size_t idx = n * C * HW + c * HW + h * W + w;
                    float diff = x_data[idx] - mean_data[c];
                    sum += diff * diff;
                }
            }
        }
        var_data[c] = sum / (N * HW);
    }

    // 更新 running stats
    for (size_t c = 0; c < C; c++) {
        float* rm_data = (float*)running_mean->data;
        float* rv_data = (float*)running_var->data;
        rm_data[c] = momentum * rm_data[c] + (1 - momentum) * mean_data[c];
        rv_data[c] = momentum * rv_data[c] + (1 - momentum) * var_data[c];
    }

    // 归一化并仿射变换
    for (size_t c = 0; c < C; c++) {
        float std = sqrtf(var_data[c] + eps);
        float inv_std = 1.0f / std;
        float gamma_c = gamma_data[c], beta_c = beta_data[c], mean_c = mean_data[c];
        for (size_t n = 0; n < N; n++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    size_t idx = n * C * HW + c * HW + h * W + w;
                    y_data[idx] = gamma_c * (x_data[idx] - mean_c) * inv_std + beta_c;
                }
            }
        }
    }
}
```

### 11.2 BatchNorm 推理阶段

```c
static void tensor_batch_norm_inference(const Tensor* x, Tensor* y,
                                        const Tensor* gamma, const Tensor* beta,
                                        const Tensor* mean, const Tensor* var,
                                        float eps) {
    size_t N = x->shape[0], C = x->shape[1];
    size_t H = (x->ndim == 4) ? x->shape[2] : 1;
    size_t W = (x->ndim == 4) ? x->shape[3] : 1;
    size_t HW = H * W;
    float* x_data = (float*)x->data;
    float* y_data = (float*)y->data;
    float* mean_data = (float*)mean->data;
    float* var_data = (float*)var->data;
    float* gamma_data = (float*)gamma->data;
    float* beta_data = (float*)beta->data;

    for (size_t c = 0; c < C; c++) {
        float std = sqrtf(var_data[c] + eps);
        float inv_std = 1.0f / std;
        float gamma_c = gamma_data[c], beta_c = beta_data[c], mean_c = mean_data[c];
        for (size_t n = 0; n < N; n++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    size_t idx = n * C * HW + c * HW + h * W + w;
                    y_data[idx] = gamma_c * (x_data[idx] - mean_c) * inv_std + beta_c;
                }
            }
        }
    }
}
```

### 11.3 LayerNorm

```c
static void tensor_layer_norm_forward(const Tensor* x, Tensor* y,
                                      const Tensor* gamma, const Tensor* beta,
                                      float eps) {
    float* x_data = (float*)x->data;
    float* y_data = (float*)y->data;
    float* gamma_data = (float*)gamma->data;
    float* beta_data = (float*)beta->data;
    size_t last_dim = x->shape[x->ndim - 1];
    size_t outer_size = x->size / last_dim;
    for (size_t i = 0; i < outer_size; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < last_dim; j++) sum += x_data[i * last_dim + j];
        float mean = sum / last_dim;
        float var_sum = 0.0f;
        for (size_t j = 0; j < last_dim; j++) {
            float diff = x_data[i * last_dim + j] - mean;
            var_sum += diff * diff;
        }
        float std = sqrtf(var_sum / last_dim + eps);
        float inv_std = 1.0f / std;
        for (size_t j = 0; j < last_dim; j++) {
            float norm = (x_data[i * last_dim + j] - mean) * inv_std;
            y_data[i * last_dim + j] = gamma_data[j] * norm + beta_data[j];
        }
    }
}
```

---

## 12. Dropout

```c
static void tensor_dropout_forward(const Tensor* x, Tensor* y, Tensor* mask,
                                   float p, bool training) {
    float* x_data = (float*)x->data;
    float* y_data = (float*)y->data;
    uint8_t* mask_data = (uint8_t*)mask->data;
    if (!training) {
        for (size_t i = 0; i < x->size; i++) y_data[i] = x_data[i];
        return;
    }
    float scale = 1.0f / (1.0f - p);
    for (size_t i = 0; i < x->size; i++) {
        mask_data[i] = (rand() < (float)RAND_MAX * (1.0f - p)) ? 1 : 0;
        y_data[i] = x_data[i] * mask_data[i] * scale;
    }
}

static void tensor_dropout_backward(const Tensor* grad_output, Tensor* grad_input,
                                    const Tensor* mask, float p) {
    float* go_data = (float*)grad_output->data;
    float* gi_data = (float*)grad_input->data;
    uint8_t* mask_data = (uint8_t*)mask->data;
    float scale = 1.0f / (1.0f - p);
    for (size_t i = 0; i < grad_output->size; i++) {
        gi_data[i] = go_data[i] * mask_data[i] * scale;
    }
}
```

---

## 13. 量化运算

### 13.1 Affine 量化

```c
static Tensor* tensor_quantize_affine(Tensor* t, TensorDType dtype) {
    float min_val = FLT_MAX, max_val = -FLT_MAX;
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->size; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    float scale = (max_val - min_val) / 255.0f;
    int32_t zero_point = (int32_t)(-min_val / scale);
    Tensor* q = tensor_create(dtype, t->layout, t->shape, t->ndim);
    q->quant_type = QUANT_AFFINE;
    q->quant.scale = scale;
    q->quant.zero_point = zero_point;

    switch (dtype) {
        case TENSOR_DTYPE_UINT8: {
            uint8_t* dst = (uint8_t*)q->data;
            for (size_t i = 0; i < t->size; i++) {
                int32_t val = (int32_t)(data[i] / scale + zero_point);
                if (val < 0) val = 0; if (val > 255) val = 255;
                dst[i] = (uint8_t)val;
            }
            break;
        }
        case TENSOR_DTYPE_INT8: {
            int8_t* dst = (int8_t*)q->data;
            for (size_t i = 0; i < t->size; i++) {
                int32_t val = (int32_t)(data[i] / scale + zero_point);
                if (val < -128) val = -128; if (val > 127) val = 127;
                dst[i] = (int8_t)val;
            }
            break;
        }
        default: break;
    }
    return q;
}

static Tensor* tensor_dequantize(const Tensor* t) {
    if (t->quant_type == QUANT_NONE) return tensor_clone(t);
    Tensor* f = tensor_create(TENSOR_DTYPE_F32, t->layout, t->shape, t->ndim);
    float* dst = (float*)f->data;
    float scale = t->quant.scale;
    int32_t zp = t->quant.zero_point;
    switch (t->dtype) {
        case TENSOR_DTYPE_INT8: {
            int8_t* src = (int8_t*)t->data;
            for (size_t i = 0; i < t->size; i++) dst[i] = (src[i] - zp) * scale;
            break;
        }
        case TENSOR_DTYPE_UINT8: {
            uint8_t* src = (uint8_t*)t->data;
            for (size_t i = 0; i < t->size; i++) dst[i] = (src[i] - zp) * scale;
            break;
        }
        default: break;
    }
    return f;
}
```

### 13.2 量化 GEMM

```c
static void tensor_gemm_quantized(int32_t* C, const int8_t* A, const int8_t* B,
                                  float scale_A, float scale_B,
                                  size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M * N; i++) C[i] = 0;
    for (size_t i = 0; i < M; i++) {
        for (size_t k = 0; k < K; k++) {
            int32_t a_val = (int32_t)A[i * K + k];
            for (size_t j = 0; j < N; j++) {
                int32_t b_val = (int32_t)B[k * N + j];
                C[i * N + j] += a_val * b_val;
            }
        }
    }
    float scale = scale_A * scale_B;
    for (size_t i = 0; i < M * N; i++) {
        C[i] = (int32_t)(C[i] * scale);
    }
}
```

---

## 14. 工具操作

```c
static void tensor_fill_f32(Tensor* t, float val) {
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->size; i++) data[i] = val;
}

static void tensor_fill_randn(Tensor* t, float mean, float std) {
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->size; i++) {
        float u1 = (float)rand() / (float)(RAND_MAX);
        float u2 = (float)rand() / (float)(RAND_MAX);
        float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * acosf(-1.0f) * u2);
        data[i] = mean + std * z;
    }
}

static void tensor_fill_xavier(Tensor* t) {
    size_t fan_in = t->ndim > 0 ? t->shape[t->ndim - 1] : 1;
    size_t fan_out = t->ndim > 1 ? t->shape[t->ndim - 2] : 1;
    float std = sqrtf(2.0f / (fan_in + fan_out));
    tensor_fill_randn(t, 0.0f, std);
}

static Tensor* tensor_concat(const Tensor** tensors, size_t n, size_t axis) {
    if (n == 0) return NULL;
    if (axis >= tensors[0]->ndim) return NULL;
    for (size_t i = 1; i < n; i++) {
        if (tensors[i]->ndim != tensors[0]->ndim) return NULL;
        for (size_t j = 0; j < tensors[0]->ndim; j++) {
            if (j != axis && tensors[i]->shape[j] != tensors[0]->shape[j]) return NULL;
        }
    }
    size_t total_dim = 0;
    for (size_t i = 0; i < n; i++) total_dim += tensors[i]->shape[axis];
    size_t* out_shape = (size_t*)malloc(sizeof(size_t) * tensors[0]->ndim);
    for (size_t i = 0; i < tensors[0]->ndim; i++) {
        out_shape[i] = (i == axis) ? total_dim : tensors[0]->shape[i];
    }
    Tensor* output = tensor_create(tensors[0]->dtype, tensors[0]->layout, out_shape, tensors[0]->ndim);
    float* dst = (float*)output->data;
    size_t offset = 0;
    size_t stride = tensors[0]->strides[axis];
    for (size_t t_idx = 0; t_idx < n; t_idx++) {
        float* src = (float*)tensors[t_idx]->data;
        size_t dim_size = tensors[t_idx]->shape[axis];
        for (size_t i = 0; i < tensors[t_idx]->size; i++) {
            size_t src_idx = i;
            size_t dim_idx = (src_idx / stride) % dim_size;
            size_t flat_idx = (src_idx % stride) + dim_idx * stride + offset * stride;
            dst[flat_idx] = src[i];
        }
        offset += dim_size;
    }
    free(out_shape);
    return output;
}

static Tensor* tensor_upsample_nearest_2d(const Tensor* x, size_t scale_h, size_t scale_w) {
    if (x->ndim != 4) return NULL;
    size_t N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
    size_t out_shape[] = {N, C, H * scale_h, W * scale_w};
    Tensor* y = tensor_create(TENSOR_DTYPE_F32, x->layout, out_shape, 4);
    float* x_data = (float*)x->data;
    float* y_data = (float*)y->data;
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    float val = x_data[n*C*H*W + c*H*W + h*W + w];
                    for (size_t sh = 0; sh < scale_h; sh++) {
                        for (size_t sw = 0; sw < scale_w; sw++) {
                            size_t out_h = h * scale_h + sh;
                            size_t out_w = w * scale_w + sw;
                            y_data[n*C*(H*scale_h)*(W*scale_w) + c*(H*scale_h)*(W*scale_w) + out_h*(W*scale_w) + out_w] = val;
                        }
                    }
                }
            }
        }
    }
    return y;
}
```

---

## 15. 与 OpenCL 版本对比

| 特性 | tensor.h (CPU) | opencl_tensor.h (GPU) |
|------|----------------|----------------------|
| 数据存储 | `void* data`（malloc/堆） | `cl_mem buffer`（VRAM） |
| 内存布局 | NCHW, NHWC, CHWN, OWI | NCHW, NHWC, CHWN |
| 数据类型 | 9种（含f64/UINT16） | 4种（f32/f16/int8/int32） |
| 量化 | 完整（affine + symmetric） | 暂不支持 |
| GEMM | 16×16 缓存分块 | 16×16 2D tile |
| 卷积 | Im2Col + blocked GEMM | 暂不支持 |
| 池化 | MaxPool + AvgPool | 暂不支持 |
| BatchNorm/LayerNorm | 完整实现 | 暂不支持 |
| Softmax | 原位沿轴归约 | 主机端回退（需数据传输） |
| Add（广播） | 全广播索引计算 | 主机端回退 |
| 内核编译 | N/A | 每帧动态编译（开销较大） |
