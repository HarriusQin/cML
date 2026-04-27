# 张量运算法则 (Tensor Operations Rules)

## 1. 基本数据结构

```c
typedef struct {
    TensorDType dtype;     // 数据类型: F32, INT8, etc.
    TensorLayout layout;   // 内存布局: NCHW, NHWC, etc.
    uint8_t ndim;          // 维度数
    size_t* shape;         // 各维度大小 [d0, d1, ..., dn-1]
    size_t* strides;       // 各维度步长 (元素数，非字节)
    void* data;            // 数据指针
    size_t size;           // 总元素数
    QuantType quant_type;  // 量化类型
    QuantParams quant;     // 量化参数 (scale, zero_point)
} Tensor;
```

### 数据类型 (TensorDType)

| 类型 | 字节数 |
|------|--------|
| TENSOR_DTYPE_F32 | 4 |
| TENSOR_DTYPE_F16 | 2 |
| TENSOR_DTYPE_INT8 | 1 |
| TENSOR_DTYPE_UINT8 | 1 |
| TENSOR_DTYPE_INT32 | 4 |

### 量化类型 (QuantType)

| 类型 | 说明 |
|------|------|
| QUANT_NONE | 无量化 |
| QUANT_AFFINE | 有仿射量化: x_q = round(x/scale) + zero_point |
| QUANT_SYMMETRIC | 对称量化: zero_point = 0 |

---

## 2. 内存布局 (Memory Layout)

| 布局 | 格式 | 用途 |
|------|------|------|
| **NCHW** | [N, C, H, W] | PyTorch默认，卷积友好 |
| **NHWC** | [N, H, W, C] | TensorFlow默认，通道最后 |
| **CHWN** | [C, H, W, N] | 某些卷积实现 |
| **OWI** | [Out, Win, In] | 扁平化权重 |

### Stride计算 (Row-major / NCHW)

```
shape = [N, C, H, W]
stride = [C*H*W, H*W, W, 1]

元素索引 [n,c,h,w] -> flat = n*stride[0] + c*stride[1] + h*stride[2] + w*stride[3]
```

### Stride计算 (NHWC)

```
shape = [N, H, W, C]
stride = [H*W*C, W*C, C, 1]

元素索引 [n,h,w,c] -> flat = n*stride[0] + h*stride[1] + w*stride[2] + c*stride[3]
```

---

## 3. 广播法则 (Broadcasting Rules)

广播将不同形状的张量自动扩展到相同形状。

### 规则

从**右端(最后)**开始比较维度：

1. 如果两个维度相等，可以兼容
2. 如果其中一个维度是 `1`，可以扩展
3. 其他情况不兼容

### 示例

| A | B | 结果 |
|---|---|------|
| [3, 4] | [3, 4] | [3, 4] (相同) |
| [3, 1] | [1, 4] | [3, 4] (双向扩展) |
| [1, 4] | [3, 1] | [3, 4] |
| [3, 4, 1] | [1, 4, 5] | [3, 4, 5] |
| [3, 4] | [5] | **不兼容** |

---

## 4. 矩阵乘法 (GEMM)

### 数学定义

```
C = α * (A @ B) + β * C
```

- A: [M × K]
- B: [K × N]
- C: [M × N]
- α, β: 标量系数

### 分块乘法 (Blocked / Tiled GEMM)

将大矩阵分成能在L1缓存放下的小块:

```
A[M×K] 分成 ib × kb 块
B[K×N] 分成 kb × jb 块

每个块组合: A[i,j] @ B[j,k] -> C[i,k]
```

```
L1缓存 ~32KB
16×16 float = 1KB per block (安全)
16×16×4 bytes = 1KB
```

### 伪代码

```
for i in 0..M step BLOCK:
    for k in 0..K step BLOCK:
        for j in 0..N step BLOCK:
            // 计算子矩阵
            ib = min(BLOCK, M-i)
            kb = min(BLOCK, K-k)
            jb = min(BLOCK, N-j)

            C[i:i+ib, j:j+jb] += A[i:i+ib, k:k+kb] @ B[k:k+kb, j:j+jb]
```

### 缓存命中率优化

1. **内层循环连续访问**: `C[i, j+k]` 连续
2. **块重排**: 确保 `A[i,k]` 内层循环沿K方向
3. **预取**: 提前加载下一块数据

---

## 5. 卷积运算 (Conv2D)

### 数学定义

```
Output[n, oc, h, w] = Σ Σ Σ Input[n, ic, h', w'] * Weight[oc, ic, kh, kw]

其中:
h' = h * stride_h + kh - pad_h
w' = w * stride_w + kw - pad_w
```

### 参数

| 参数 | 说明 |
|------|------|
| stride | 步长，默认1 |
| padding | 填充，默认0 |
| dilation | 膨胀，默认1 |

### 输出尺寸

```
H_out = (H - KH + 2*pad_h) / stride_h + 1
W_out = (W - KW + 2*pad_w) / stride_w + 1
```

### Im2Col方法

将卷积转换为矩阵乘法:

1. 将每个感受野展平为列向量
2. 所有列组成矩阵 Col [KH×KW×C, H_out×W_out]
3. Weight [OutC, KH×KW×C]
4. GEMM: Weight @ Col -> [OutC, H_out×W_out]

---

## 6. 池化运算 (Pooling)

### Max Pooling

```
Output[h, w] = max(Input[h*stride + kh, w*stride + kw])
```

### Average Pooling

```
Output[h, w] = mean(Input[h*stride + kh, w*stride + kw])
```

### 输出尺寸

与Conv2D相同公式。

---

## 7. 激活函数

### ReLU

```
f(x) = max(0, x)
```

### Sigmoid

```
f(x) = 1 / (1 + e^(-x))
```

### Tanh

```
f(x) = tanh(x)
```

### Softmax

沿指定轴计算:

```
softmax(x_i) = e^(x_i) / Σ e^(x_j)
```

**数值稳定**: 减去最大值

```
x_i = e^(x_i - max) / Σ e^(x_j - max)
```

---

## 8. 规约运算 (Reduction)

沿指定轴求和/均值/最大:

```
Input: [N, C, H, W]
sum(axis=2) -> [N, C, W]
mean(axis=1) -> [N, H, W]
max(axis=3) -> [N, C, H]
```

---

## 9. 量化 (Quantization)

### Affine Quantization

```
x_q = round(x / scale) + zero_point

x ≈ (x_q - zero_point) * scale
```

### Symmetric Quantization

```
zero_point = 0
x_q = round(x / scale)
```

### 量化GEMM

```
// INT8 乘法
sum_int = Σ (a_q - a_zp) * (b_q - b_zp)

// 反量化
sum ≈ sum_int * scale_a * scale_b
```

---

## 10. 维度转换

### Reshape

元素总数不变，可以改变维度解释方式:

```
[2, 3, 4] -> [6, 4] -> [24]
```

### Transpose

交换两个轴:

```
[N, C, H, W] transposed [N, H, W, C] (NCHW->NHWC)
```

### Slice

沿一个轴取子集:

```
[:, :, 0:3, :] 取高度0-2
```

---

## 11. 运算复杂度

| 运算 | 时间复杂度 | 空间复杂度 |
|------|------------|------------|
| Element-wise | O(n) | O(n) 或 O(1) inplace |
| GEMM | O(M×N×K) | O(M×N) |
| Conv2D | O(N×OutC×H_out×W_out×C×KH×KW) | O(N×OutC×H_out×W_out) |
| MaxPool | O(N×H×W×pool_h×pool_w) | O(1) |
| Softmax | O(n) | O(1) |

---

## 12. 内存布局对性能的影响

### NCHW (PyTorch风格)

优点:
- 卷积层权重是连续的 [OutC, C, KH, KW]
- Batch matmul 效率高

缺点:
- NHWC在某些硬件(ARM NEON)上SIMD更高效

### NHWC (TensorFlow风格)

优点:
- 通道维度最后，SIMD一次处理多个通道
- 适合量化推理

缺点:
- 跨通道操作需要跳跃访问

### 选择建议

| 场景 | 推荐布局 |
|------|----------|
| PyTorch训练 | NCHW |
| ARM推理 | NHWC |
| 量化模型 | NHWC |
| 通用 | NCHW |

---

## 13. 量化计算 (Quantized Operations)

### INT8 量化流程

```
Float32 -> Quantize -> INT8 -> Compute -> Dequantize -> Float32
```

### 量化GEMM (INT8)

```c
// 输入: A_q[M×K], B_q[K×N] (int8)
// 输出: C_q[M×N] (int8, 重新量化)

void gemm_quantized(int32_t* C, int8_t* A, int8_t* B,
                    float scale_A, float scale_B,
                    size_t M, size_t N, size_t K) {
    // 1. 清零累加器
    for (size_t i = 0; i < M*N; i++) C[i] = 0;

    // 2. INT8 矩阵乘法
    for (size_t i = 0; i < M; i++)
        for (size_t k = 0; k < K; k++)
            for (size_t j = 0; j < N; j++)
                C[i*N + j] += A[i*K + k] * B[k*N + j];

    // 3. 反量化回 float 并缩放
    float scale = scale_A * scale_B;
    for (size_t i = 0; i < M*N; i++)
        C[i] = (int32_t)(C[i] * scale);
}
```

### 量化ReLU

```c
// INT8 ReLU: 负数置零
void relu_quantized(int8_t* data, size_t size) {
    for (size_t i = 0; i < size; i++)
        if (data[i] < 0) data[i] = 0;
}
```

### 量化ReLU6

```c
// INT8 ReLU6: 限制在 [0, 6] 范围
void relu6_quantized(int8_t* data, size_t size, float scale, int8_t zero_point) {
    int8_t min_val = -zero_point;
    int8_t max_val = (int8_t)(6.0f / scale + zero_point);
    for (size_t i = 0; i < size; i++) {
        if (data[i] < min_val) data[i] = min_val;
        if (data[i] > max_val) data[i] = max_val;
    }
}
```

### 量化Softmax

Softmax 需要浮点计算后再量化:

```c
void softmax_quantized(int8_t* output, float* input,
                      float scale, int8_t zero_point, size_t n) {
    // 1. 去量化
    for (size_t i = 0; i < n; i++)
        temp[i] = ((float)input[i] - zero_point) * scale;

    // 2. 找最大值 (数值稳定)
    float max_val = temp[0];
    for (size_t i = 1; i < n; i++)
        if (temp[i] > max_val) max_val = temp[i];

    // 3. Exp 并求和
    float sum = 0;
    for (size_t i = 0; i < n; i++) {
        temp[i] = exp(temp[i] - max_val);
        sum += temp[i];
    }

    // 4. 归一化并requantize
    for (size_t i = 0; i < n; i++) {
        float val = temp[i] / sum;
        output[i] = (int8_t)(val / scale + zero_point);
    }
}
```

### 量化池化

**MaxPool**: 直接比较 INT8 值，无需去量化

**AvgPool**: 需要去量化计算均值后再量化

```c
// INT8 AvgPool
int32_t sum = 0;
for (size_t ph = 0; ph < pool_h; ph++)
    for (size_t pw = 0; pw < pool_w; pw++)
        sum += input[...];

// 去量化求均值
float mean = (sum - zero_point * pool_size) * scale / pool_size;

// 再量化
output = (int8_t)(mean / scale + zero_point);
```

### BatchNorm 融合

推理时将 BatchNorm 参数融合到卷积权重:

```c
void fuse_batch_norm(int8_t* weight, float* bias,
                    float* mean, float* var,
                    float* gamma, float* beta,
                    float eps, size_t channels) {
    for (size_t c = 0; c < channels; c++) {
        float std = sqrt(var[c] + eps);
        float scale = gamma[c] / std;

        // 融合到权重
        for (size_t i = 0; i < kernel_size; i++)
            weight[c * kernel_size + i] *= scale;

        // 融合到偏置
        if (bias)
            bias[c] = scale * (bias[c] - mean[c]) + beta[c];
    }
}
```

### 量化推理流程

```
1. 量化输入图像: input_q = round(input / scale_input + zp_input)
2. 卷积 (INT8 GEMM)
3. ReLU / ReLU6 (INT8)
4. 池化 (INT8 比较或去量化计算)
5. 全连接 (INT8 GEMM)
6. Softmax (需要去量化)
7. 反量化输出
```

### 精度考虑

| 量化误差来源 | 影响 |
|-------------|------|
| 舍入误差 | 累积后可能显著 |
| 溢出 | INT8 乘法可能超范围 (-128~127) |
| ReLU6 截断 | 大值被截断 |

**缓解方法**:
- 使用 per-channel 量化
- 在关键层保留 FP32
- 混合精度量化

---

## 14. 常见实现问题与修复

### 问题 1: Transpose 只复制数据不重排

**错误实现**:
```c
// BUG: 只交换了 shape/stride，数据字节顺序不变
Tensor* tensor_transpose(const Tensor* t, uint8_t axis0, uint8_t axis1) {
    Tensor* r = tensor_create(t->dtype, t->layout, t->shape, t->ndim);
    memcpy(r->data, t->data, tensor_nbytes(t));  // 数据未转置！

    size_t tmp = r->shape[axis0]; r->shape[axis0] = r->shape[axis1]; r->shape[axis1] = tmp;
    tmp = r->strides[axis0]; r->strides[axis0] = r->strides[axis1]; r->strides[axis1] = tmp;
    return r;
}
```

**正确实现**:
```c
Tensor* tensor_transpose(const Tensor* t, uint8_t axis0, uint8_t axis1) {
    Tensor* r = tensor_create(t->dtype, t->layout, t->shape, t->ndim);

    // 正确重排数据元素
    size_t elem_size = tensor_dtype_size(t->dtype);
    for (size_t i = 0; i < r->size; i++) {
        // 计算输出索引对应的输入索引
        size_t idx[t->ndim];
        size_t remaining = i;
        for (uint8_t d = 0; d < t->ndim; d++) {
            idx[d] = remaining % r->shape[d];
            remaining /= r->shape[d];
        }
        // 交换轴
        size_t tmp = idx[axis0]; idx[axis0] = idx[axis1]; idx[axis1] = tmp;

        // 转换为 flat 索引
        size_t src_flat = 0;
        for (uint8_t d = 0; d < t->ndim; d++)
            src_flat += idx[d] * t->strides[d];

        memcpy(dst + i * elem_size, src + src_flat * elem_size, elem_size);
    }

    size_t tmp = r->shape[axis0]; r->shape[axis0] = r->shape[axis1]; r->shape[axis1] = tmp;
    tmp = r->strides[axis0]; r->strides[axis0] = r->strides[axis1]; r->strides[axis1] = tmp;
    return r;
}
```

### 问题 2: FC 层手动转置导致数据布局错误

**错误做法**:
```c
// BUG: w_t 的 shape 改变了，但数据布局没变
Tensor* w_t = tensor_create(TENSOR_DTYPE_F32, layout,
                             (size_t[]){weight->shape[1], weight->shape[0]}, 2);
memcpy(w_t->data, weight->data, tensor_nbytes(weight));  // 数据未转置！
```

**原因**: 2D tensor 的 `shape` 和 `strides` 相关联。创建新 shape 后，`strides` 会按新 shape 计算，导致逻辑布局错误。

**正确做法**: 直接计算矩阵乘法，避免转置
```c
// y[b,o] = sum_i x[b,i] * W[o,i]
for (b)
  for (o)
    for (i)
      y[b,o] += x[b,i] * W[o,i];
```

### 问题 3: Inplace 操作破坏后续计算

**错误**:
```c
// relu_backward 原地修改 grad，但 grad 后续还会用于 tensor_gemm
void relu_backward(Tensor* grad_output, const Tensor* mask) {
    for (size_t i = 0; i < grad_output->size; i++)
        grad_output->data[i] *= mask->data[i];  // 破坏原数据！
}
```

**正确**:
```c
// 返回新 tensor，不修改输入
Tensor* relu_backward(const Tensor* grad_output, const Tensor* mask) {
    Tensor* grad = tensor_create(grad_output->dtype, grad_output->layout,
                                  grad_output->shape, grad_output->ndim);
    for (size_t i = 0; i < grad_output->size; i++)
        grad->data[i] = grad_output->data[i] * mask->data[i];
    return grad;
}
```

### 问题 4: Stride 计算在不同 Layout 下的差异

**NCHW (row-major)**:
```
shape [N, C, H, W]
strides = [C*H*W, H*W, W, 1]
元素 [n,c,h,w] -> flat = n*CHW + c*HW + h*W + w
```

**NHWC**:
```
shape [N, H, W, C]
strides = [H*W*C, W*C, C, 1]
元素 [n,h,w,c] -> flat = n*HWC + h*WC + w*C + c
```

如果把 NHWC 数据按 NCHW stride 解读，会得到完全错误的结果。

### 问题 5: 内存泄漏

**错误** (resize 时):
```c
free(layer->relu_mask->shape);
free(layer->relu_mask->strides);
layer->relu_mask->shape = malloc(...);  // data 指针未释放！
```

**正确**:
```c
tensor_free(layer->relu_mask);  // 释放整个 tensor
layer->relu_mask = tensor_create(...);
```

### 问题 6: GEMM 维度顺序混淆

**C = A @ B**
- A: [M × K]
- B: [K × N]
- C: [M × N]

**错误**:
```c
// 维度混淆
tensor_gemm(C, A, B, M, K, N);  // B 的维度不对！
```

**正确**:
```c
tensor_gemm(C, A, B, M, N, K);  // A[M,K] @ B[K,N] = C[M,N]
```

### 调试建议

1. **打印 tensor 形状**: `tensor_print(t, "name");`
2. **打印部分数据**: 检查前几个元素的值
3. **用已知输入验证**: 用简单的 identity 矩阵测试
4. **梯度检查**: 对比数值梯度和解析梯度
5. **单步执行**: 先验证前向传播，再验证反向传播
