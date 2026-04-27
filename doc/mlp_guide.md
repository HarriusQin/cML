# 多层感知机 (MLP) 实现指南

## 目录

1. [MLP 架构概述](#1-mlp-架构概述)
2. [前向传播 (Forward Pass)](#2-前向传播-forward-pass)
3. [激活函数](#3-激活函数)
4. [损失函数 (Loss Function)](#4-损失函数-loss-function)
5. [反向传播 (Backward Pass)](#5-反向传播-backward-pass)
6. [参数更新 (SGD Optimizer)](#6-参数更新-sgd-optimizer)
7. [训练流程](#7-训练流程)
8. [MLP 实现细节](#8-mlp-实现细节)
9. [已知问题与修复](#9-已知问题与修复)
10. [使用示例](#10-使用示例)

---

## 1. MLP 架构概述

### 网络结构

```
输入层 (784) -> 隐藏层1 (256) -> 隐藏层2 (256) -> 输出层 (10)
```

- **输入层**: MNIST 图像 28×28=784 个像素值，归一化到 [0,1]
- **隐藏层**: 全连接层 + ReLU 激活
- **输出层**: 10 个类别，Softmax 归一化输出概率

### 数据结构

```c
// 单个全连接层
typedef struct {
    Tensor* weight;      // [out_features, in_features]
    Tensor* bias;        // [out_features]
    Tensor* grad_w;      // 权重梯度
    Tensor* grad_b;      // 偏置梯度
    Tensor* input_cache;      // 前向传播缓存
    Tensor* preact_cache;     // 激活前缓存
    Tensor* relu_mask;        // ReLU 反向传播用
} FCLayer;

// MLP 网络
typedef struct {
    FCLayer** layers;          // 层数组
    size_t num_layers;         // 层数
    size_t input_dim;          // 输入维度
    size_t output_dim;         // 输出维度
    Tensor** activations;       // 存储用于反向传播
} MLP;
```

### 权重初始化

使用 **Xavier 初始化**:

```c
// 方差 = sqrt(2.0 / (fan_in + fan_out))
float std = sqrtf(2.0f / (in_features + out_features));
tensor_fill_randn(weight, 0.0f, std);
```

偏置初始化为 0。

---

## 2. 前向传播 (Forward Pass)

### 全连接层 (Fully Connected Layer)

**数学公式**:
```
y = x @ W.T + b
```

- `x`: [batch, in_features] - 输入
- `W`: [out_features, in_features] - 权重
- `b`: [out_features] - 偏置
- `y`: [batch, out_features] - 输出

**实现** (注意矩阵乘法的正确性):

```c
// 错误做法: 简单 memcpy 转置 (数据布局不变!)
Tensor* w_t = tensor_create(...);
memcpy(w_t->data, w->data, ...);  // BUG: stride 信息丢失

// 正确做法: 直接计算 y[b,o] = sum_i x[b,i] * W[o,i]
for (size_t b = 0; b < batch; b++) {
    for (size_t o = 0; o < out_features; o++) {
        float sum = 0.0f;
        for (size_t i = 0; i < in_features; i++) {
            sum += x_data[b * in_features + i] * w_data[o * in_features + i];
        }
        preact_data[b * out_features + o] = sum + bias_data[o];
    }
}
```

### 矩阵乘法注意事项

权重矩阵 W 存储为 `[out_features, in_features]`，即行优先存储。
当执行 `x @ W.T` 时:

- W.T 的元素 `[i,o]` = W 的元素 `[o,i]`
- 不能简单用 `memcpy`，因为 stride 信息不同

---

## 3. 激活函数

### ReLU (Rectified Linear Unit)

**数学公式**:
```
f(x) = max(0, x)
```

**前向传播** (原地操作):

```c
for (size_t i = 0; i < size; i++) {
    mask_data[i] = (x_data[i] > 0) ? 1.0f : 0.0f;  // 保存 mask
    if (x_data[i] < 0) x_data[i] = 0;               // inplace
}
```

**反向传播**:
```
dL/dx = dL/dy * mask
```

其中 mask 在前向传播时保存。

### Softmax

**数学公式**:
```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

**数值稳定版本**:
```c
// 减去最大值防止溢出
float max_val = data[0];
for (size_t i = 1; i < n; i++)
    if (data[i] > max_val) max_val = data[i];

float sum = 0;
for (size_t i = 0; i < n; i++) {
    data[i] = expf(data[i] - max_val);
    sum += data[i];
}
for (size_t i = 0; i < n; i++)
    data[i] /= sum;
```

---

## 4. 损失函数 (Loss Function)

### Cross-Entropy Loss

**数学公式**:
```
L = -sum(y_true * log(y_pred)) / batch_size
```

**使用场景**: 多分类问题，Softmax 输出

**梯度** (对 softmax 输入):
```
dL/dx_c = pred_c - target_c
```

其中 target_c 是目标类别的 one-hot 编码 (正确类为 1，其他为 0)。

**实现**:

```c
// pred: [batch, num_classes] 概率
// targets: [batch] 类索引
for (size_t b = 0; b < batch; b++) {
    size_t target = (size_t)target_data[b];
    for (size_t c = 0; c < num_classes; c++) {
        grad[b * num_classes + c] = pred[b * num_classes + c];
        if (c == target)
            grad[b * num_classes + c] -= 1.0f;
    }
}
```

---

## 5. 反向传播 (Backward Pass)

### 计算图

```
输入 x
  |
  v
Layer0: y0 = ReLU(x @ W0.T + b0)
  |
  v
Layer1: y1 = ReLU(y0 @ W1.T + b1)
  |
  v
Layer2: y2 = Softmax(y1 @ W2.T + b2)
  |
  v
Loss = CrossEntropy(y2, target)
```

### 反向传播顺序

从输出层向输入层传播梯度:

```
dL/dW2, dL/db2, dL/dy1 = dL/dy2 @ W2
    |
    v
dL/dW1, dL/db1, dL/dy0 = dL/dy1 @ W1
    |
    v
dL/dW0, dL/db0, dL/dx   = dL/dy0 @ W0
```

### 梯度计算

**FC 层梯度**:

```c
// dL/dW = (dL/dy).T @ x
// grad_w[o,i] = sum_b grad[b,o] * x[b,i]

// dL/db = sum_b dL/dy[b,o]
grad_b[o] = sum_b grad[b * out_features + o] / batch_size;

// dL/dx = grad @ W
// dx[b,i] = sum_o grad[b,o] * W[o,i]
```

**ReLU 梯度**:

```c
// dL/dx = dL/dy * mask
// mask[i] = 1 if x[i] > 0 else 0
for (size_t i = 0; i < size; i++)
    grad[i] = grad[i] * mask[i];
```

### 关键实现细节

1. **缓存中间结果**: 前向传播时保存 input_cache, preact_cache, relu_mask
2. **原地修改问题**: `relu_backward` 不能原地修改 grad，需要创建新 tensor
3. **内存管理**: 每层反向传播后释放不再需要的 tensor

---

## 6. 参数更新 (SGD Optimizer)

### SGD with Momentum

**公式**:
```
v_t = momentum * v_{t-1} - lr * grad
w_t = w_{t-1} + v_t
```

**带 Weight Decay**:
```
v_t = momentum * v_{t-1} - lr * (grad + weight_decay * w)
w_t = w_{t-1} + v_t
```

### 梯度裁剪

防止梯度爆炸:

```c
void clip_gradients(Tensor* grad, float max_norm) {
    float norm_sq = 0;
    for (size_t i = 0; i < grad->size; i++)
        norm_sq += data[i] * data[i];
    float norm = sqrtf(norm_sq);

    if (norm > max_norm) {
        float scale = max_norm / norm;
        for (size_t i = 0; i < grad->size; i++)
            data[i] *= scale;
    }
}
```

---

## 7. 训练流程

### 单步训练

```c
float train_step(MLP* mlp, SGDOptimizer* opt,
                 const Tensor* input, const Tensor* targets) {
    // 1. 前向传播
    Tensor* output = mlp_forward(mlp, input);

    // 2. 计算损失
    float loss = cross_entropy_loss(output, targets);

    // 3. 计算损失函数梯度
    Tensor* grad = cross_entropy_grad(output, targets);

    // 4. 反向传播
    mlp_backward(mlp, grad);

    // 5. 更新参数
    mlp_update(mlp, opt);

    // 6. 释放内存
    tensor_free(output);
    tensor_free(grad);

    return loss;
}
```

### 完整训练循环

```c
for (size_t epoch = 0; epoch < epochs; epoch++) {
    for (size_t batch = 0; batch < batches_per_epoch; batch++) {
        // 获取 mini-batch
        Tensor* X = get_batch(images, batch * batch_size, batch_size);
        Tensor* y = get_labels(labels, batch * batch_size, batch_size);

        // 训练一步
        float loss = train_step(mlp, opt, X, y);

        tensor_free(X);
        tensor_free(y);
    }

    // 评估
    float acc = evaluate(mlp, test_images, test_labels);
    printf("Epoch %zu, Accuracy: %.2f%%\n", epoch, 100.0f * acc);
}
```

---

## 8. MLP 实现细节

### 层创建

```c
FCLayer* fc_layer_create(size_t in_features, size_t out_features) {
    FCLayer* layer = malloc(sizeof(FCLayer));

    // 权重 [out_features, in_features]
    layer->weight = tensor_create(...);
    tensor_fill_randn(layer->weight, 0.0f, sqrtf(2.0f / (in + out)));

    // 偏置 [out_features]
    layer->bias = tensor_create(...);
    tensor_fill_f32(layer->bias, 0.0f);

    // 梯度缓冲区
    layer->grad_w = tensor_create(...);
    layer->grad_b = tensor_create(...);

    // 缓存 (反向传播用)
    layer->input_cache = NULL;
    layer->preact_cache = NULL;
    layer->relu_mask = NULL;

    return layer;
}
```

### 缓存管理

```c
// relu_mask 每次前向传播时分配/重用
if (layer->relu_mask == NULL) {
    layer->relu_mask = tensor_create(dtype, layout, shape, ndim);
} else if (layer->relu_mask->size != x->size) {
    tensor_free(layer->relu_mask);
    layer->relu_mask = tensor_create(...);
}
```

### 内存释放

```c
void fc_layer_free(FCLayer* layer) {
    tensor_free(layer->weight);
    tensor_free(layer->bias);
    tensor_free(layer->grad_w);
    tensor_free(layer->grad_b);
    tensor_free(layer->input_cache);
    tensor_free(layer->preact_cache);
    tensor_free(layer->relu_mask);
    free(layer);
}
```

---

## 9. 已知问题与修复

### 问题 1: tensor_transpose 数据未重排

**现象**: MLP 准确率停留在 ~10% (随机猜测)

**原因**:
```c
// 错误实现 - 只复制数据，不重排
Tensor* tensor_transpose(const Tensor* t, ...) {
    Tensor* r = tensor_create(...);
    memcpy(r->data, t->data, ...);  // BUG: 数据顺序不变
    // 只交换了 shape/stride
    ...
}
```

**修复**: 正确实现转置，重新排列数据元素

### 问题 2: fc_layer_forward 转置错误

**原因**: 手动创建转置 tensor 时，数据布局不正确

**修复**: 直接计算矩阵乘法，避免转置:
```c
// y[b,o] = sum_i x[b,i] * W[o,i]
for (b)
  for (o)
    sum = 0;
    for (i)
      sum += x[b,i] * W[o,i];
```

### 问题 3: relu_backward 原地修改

**原因**:
```c
// 错误 - 原地修改 grad，但 grad 后续还会被使用
void relu_backward(Tensor* grad, const Tensor* mask) {
    for (i) grad->data[i] *= mask->data[i];  // 破坏原数据
}
```

**修复**: 返回新 tensor
```c
Tensor* relu_backward(const Tensor* grad, const Tensor* mask) {
    Tensor* new_grad = tensor_create(...);
    for (i) new_grad->data[i] = grad->data[i] * mask->data[i];
    return new_grad;
}
```

### 问题 4: relu_mask 内存泄漏

**原因**: resize 时只释放了 shape/strides，没有释放 data

**修复**: 使用 tensor_free 释放整个 tensor

---

## 10. 使用示例

### 创建网络

```c
// 创建 MLP: 784 -> 256 -> 256 -> 10
MLP* mlp = mlp_create(784, 256, 10, 3);

// 创建优化器: 学习率 0.01, momentum 0.9
SGDOptimizer* opt = sgd_create(mlp, 0.01f, 0.9f, 0.0001f);
```

### 训练

```c
// 加载 MNIST 数据
MNISTData* data = mnist_load("./data");

// 训练多个 epoch
for (size_t epoch = 0; epoch < 10; epoch++) {
    for (size_t batch = 0; batch < batches; batch++) {
        Tensor* X = get_batch(data->train_images, batch * 128, 128);
        Tensor* y = get_labels_batch(data->train_labels, batch * 128, 128);

        float loss = mlp_train_step(mlp, opt, X, y);

        tensor_free(X);
        tensor_free(y);
    }

    // 评估
    float acc = mlp_accuracy(mlp, data->test_images, data->test_labels);
    printf("Epoch %zu: Accuracy = %.2f%%\n", epoch, 100.0f * acc);
}
```

### 预测

```c
Tensor* pred = mlp_predict(mlp, input_image);

// 获取预测类别
size_t pred_class = 0;
float max_prob = ((float*)pred->data)[0];
for (size_t c = 1; c < 10; c++) {
    float p = ((float*)pred->data)[c];
    if (p > max_prob) {
        max_prob = p;
        pred_class = c;
    }
}
printf("Predicted class: %zu (prob: %.2f%%)\n", pred_class, 100.0f * max_prob);

tensor_free(pred);
```

### 清理

```c
sgd_free(opt);
mlp_free(mlp);
mnist_free(data);
```

---

## 11. Dataset 适配器 (CSV/Iris)

本节说明如何使用 `dataset` 结构将 CSV 数据转换为 MLP 可用的格式。

### Dataset 结构

```c
typedef struct {
    num_column* features;     // 特征列数组
    label_column* labels;     // 标签列数组
    size_t rows;              // 样本数
    size_t num_features;      // 特征数量
    size_t num_labels;        // 标签数量
} dataset;
```

### CSV 转 Tensor

```c
/**
 * 将 dataset 特征转换为 Tensor [n_samples, n_features]
 * 使用 min-max 归一化到 [0, 1]
 */
static Tensor* dataset_features_to_tensor(const dataset* ds, size_t feature_idx) {
    size_t n_samples = ds->rows;
    size_t n_features = ds->num_features;

    size_t shape[] = {n_samples, n_features};
    Tensor* t = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 2);
    float* data = (float*)t->data;

    // 找每列的 min/max
    double* min_vals = malloc(sizeof(double) * n_features);
    double* max_vals = malloc(sizeof(double) * n_features);

    for (size_t f = 0; f < n_features; f++) {
        min_vals[f] = ds->features[f].data[0];
        max_vals[f] = ds->features[f].data[0];
        for (size_t i = 1; i < n_samples; i++) {
            double val = ds->features[f].data[i];
            if (val < min_vals[f]) min_vals[f] = val;
            if (val > max_vals[f]) max_vals[f] = val;
        }
    }

    // 归一化到 [0, 1]
    for (size_t i = 0; i < n_samples; i++) {
        for (size_t f = 0; f < n_features; f++) {
            double range = max_vals[f] - min_vals[f];
            double val = ds->features[f].data[i];
            data[i * n_features + f] = (range > 0) ?
                (float)((val - min_vals[f]) / range) : 0.0f;
        }
    }

    free(min_vals);
    free(max_vals);
    return t;
}

/**
 * 将 dataset 标签转换为 Tensor [n_samples]
 * 标签存储为类别索引 (0, 1, 2, ...)
 */
static Tensor* dataset_labels_to_tensor(const dataset* ds, size_t label_idx) {
    size_t n_samples = ds->rows;
    size_t shape[] = {n_samples};
    Tensor* t = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 1);
    float* data = (float*)t->data;

    for (size_t i = 0; i < n_samples; i++) {
        data[i] = (float)ds->labels[label_idx].labels[i];
    }
    return t;
}
```

### 训练/测试集划分

```c
typedef struct {
    size_t* train_indices;
    size_t* test_indices;
    size_t train_size;
    size_t test_size;
} TrainTestSplit;

static void train_test_split_indices(size_t n_samples, float test_ratio,
                                     unsigned int seed,
                                     size_t** train_out, size_t** test_out,
                                     size_t* train_size_out, size_t* test_size_out) {
    size_t test_size = (size_t)(n_samples * test_ratio);
    size_t train_size = n_samples - test_size;

    size_t* indices = malloc(sizeof(size_t) * n_samples);
    for (size_t i = 0; i < n_samples; i++) indices[i] = i;

    // Fisher-Yates 洗牌
    srand(seed);
    for (size_t i = n_samples - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        size_t tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }

    *train_out = malloc(sizeof(size_t) * train_size);
    *test_out = malloc(sizeof(size_t) * test_size);
    *train_size_out = train_size;
    *test_size_out = test_size;

    memcpy(*train_out, indices, sizeof(size_t) * train_size);
    memcpy(*test_out, indices + train_size, sizeof(size_t) * test_size);
    free(indices);
}
```

### Mini-Batch 获取

```c
static Tensor* get_feature_batch(const Tensor* features,
                                const size_t* indices, size_t batch_size) {
    size_t n_features = features->shape[1];
    size_t shape[] = {batch_size, n_features};
    Tensor* batch = tensor_create(TENSOR_DTYPE_F32, features->layout, shape, 2);
    float* dst = (float*)batch->data;
    float* src = (float*)features->data;

    for (size_t i = 0; i < batch_size; i++) {
        memcpy(dst + i * n_features,
               src + indices[i] * n_features,
               sizeof(float) * n_features);
    }
    return batch;
}

static Tensor* get_label_batch(const Tensor* labels,
                             const size_t* indices, size_t batch_size) {
    size_t shape[] = {batch_size};
    Tensor* batch = tensor_create(TENSOR_DTYPE_F32, labels->layout, shape, 1);
    float* dst = (float*)batch->data;
    float* src = (float*)labels->data;

    for (size_t i = 0; i < batch_size; i++) {
        dst[i] = src[indices[i]];
    }
    return batch;
}
```

### 完整示例 (Iris 数据集)

```c
int main(int argc, char* argv[]) {
    const char* csv_path = "data/iris.csv";

    // 1. 加载 CSV 并转换为 dataset
    csv_t* csv = csv_load(csv_path);
    const char* labels[] = {"species"};
    dataset* ds = csv_to_dataset(csv, labels, 1);

    // 2. 转换为 Tensor
    Tensor* X = dataset_features_to_tensor(ds, 0);  // [150, 4]
    Tensor* y = dataset_labels_to_tensor(ds, 0);    // [150]

    // 3. 划分训练/测试集
    TrainTestSplit split;
    train_test_split_indices(ds->rows, 0.2f, 42,
                           &split.train_indices, &split.test_indices,
                           &split.train_size, &split.test_size);

    // 4. 创建 MLP (4 -> 16 -> 3)
    MLP* mlp = mlp_create(4, 16, 3, 2);
    SGDOptimizer* opt = sgd_create(mlp, 0.01f, 0.9f, 0.0001f);

    // 5. 训练
    for (size_t epoch = 0; epoch < 200; epoch++) {
        for (size_t b = 0; b < batches_per_epoch; b++) {
            Tensor* X_batch = get_feature_batch(X, split.train_indices + b * batch_size, batch_size);
            Tensor* y_batch = get_label_batch(y, split.train_indices + b * batch_size, batch_size);
            mlp_train_step(mlp, opt, X_batch, y_batch);
            tensor_free(X_batch);
            tensor_free(y_batch);
        }
    }

    // 6. 评估
    float acc = evaluate_accuracy(mlp, X, y, split.test_indices, split.test_size, 3);
    printf("Test Accuracy: %.2f%%\n", 100.0f * acc);

    // 7. 清理
    sgd_free(opt);
    mlp_free(mlp);
    tensor_free(X);
    tensor_free(y);
    free_dataset(ds);
    free_csv_data(csv);
    free(csv);

    return 0;
}
```

### 完整示例 (MNIST 数据集)

MNIST 有专用的 `MNISTData` 结构，无需手动转换:

```c
// 1. 加载 MNIST
MNISTData* data = mnist_load("./data");

// 2. 创建 MLP (784 -> 256 -> 256 -> 10)
MLP* mlp = mlp_create(784, 256, 10, 3);
SGDOptimizer* opt = sgd_create(mlp, 0.01f, 0.9f, 0.0001f);

// 3. 获取 batch
Tensor* X_batch = get_batch(data->train_images, start_idx, batch_size);
Tensor* y_batch = get_labels_batch(data->train_labels, start_idx, batch_size);

// 4. 训练
float loss = mlp_train_step(mlp, opt, X_batch, y_batch);

// 5. 评估
float acc = mlp_accuracy(mlp, data->test_images, data->test_labels);

// 6. 清理
sgd_free(opt);
mlp_free(mlp);
mnist_free(data);
```

---

## 附录: 超参数建议

| 参数 | MNIST | CIFAR-10 | 建议范围 |
|------|-------|----------|----------|
| 学习率 | 0.01 | 0.01 - 0.1 | 0.001 - 0.1 |
| Momentum | 0.9 | 0.9 | 0.5 - 0.99 |
| Weight Decay | 0.0001 | 0.0005 | 0 - 0.001 |
| Batch Size | 128 | 128 | 32 - 256 |
| 隐藏层维度 | 256 | 512 - 1024 | 输入的 0.5 - 4 倍 |
| 层数 | 2-3 | 3-5 | 2-5 |

---

## 附录: 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 准确率 = 10% | 前向传播错误 | 检查矩阵乘法 |
| 准确率不增加 | 梯度为 0 | 检查反向传播 |
| loss 爆炸 | 学习率太高 | 降低学习率或梯度裁剪 |
| loss 不下降 | 学习率太低 | 提高学习率 |
| 训练慢 | 矩阵乘法效率低 | 检查 GEMM 实现 |
