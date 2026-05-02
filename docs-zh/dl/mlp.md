# 多层感知机 (MLP)

带全连接层的前馈神经网络。

## 架构

```
输入 (784) -> Hidden1 (256) -> Hidden2 (128) -> 输出 (10)
                |                |
              ReLU             ReLU
```

## 前向传播

对于每个全连接层：
$$\mathbf{y} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$$

其中 $\sigma$ 通常是 ReLU: $\sigma(x) = \max(0, x)$

## 反向传播

1. 计算输出误差: $\delta^L = \nabla_a C \odot \sigma'(\mathbf{z}^L)$
2. 反向传播误差: $\delta^l = (\mathbf{W}^{l+1})^T \delta^{l+1} \odot \sigma'(\mathbf{z}^l)$
3. 计算梯度: $\frac{\partial C}{\partial \mathbf{W}^l} = \delta^l (\mathbf{a}^{l-1})^T$
4. 使用带动量的梯度下降更新权重

## 数据结构

### FCLayer（全连接层）

```c
typedef struct {
    Tensor* weights;              // [output_dim, input_dim]
    Tensor* bias;                 // [output_dim]
    Tensor* grad_w;              // 权重梯度（用于优化器）
    Tensor* grad_b;              // 偏置梯度
    Tensor* velocity_w;         // 动量速度（用于 SGD）
    Tensor* velocity_b;
    size_t input_dim;
    size_t output_dim;
    ActivationType activation;
} FCLayer;
```

### MLP

```c
typedef struct {
    FCLayer** layers;            // 层指针数组
    size_t num_layers;
    size_t input_dim;
    size_t output_dim;
} MLP;
```

### SGD 优化器

```c
typedef struct {
    MLP* mlp;
    float lr;                    // 学习率
    float momentum;              // 动量系数
    float weight_decay;          // L2 正则化
} SGDOptimizer;
```

## 函数

```c
MLP* mlp_create(size_t input_dim, size_t hidden_dim, size_t output_dim, size_t num_layers);
SGDOptimizer* sgd_create(MLP* mlp, float lr, float momentum, float weight_decay);
float mlp_train_step(MLP* mlp, SGDOptimizer* opt, Tensor* X, Tensor* y);
Tensor* mlp_predict(MLP* mlp, Tensor* input);
float mlp_accuracy(MLP* mlp, Tensor* input, Tensor* labels);
void mlp_free(MLP* mlp);
void sgd_free(SGDOptimizer* opt);
```

## 示例

```c
#define MLP_IMPLEMENTATION
#include "mlp.h"

// 创建 MLP: 784 -> 256 -> 128 -> 10
MLP* mlp = mlp_create(784, 256, 10, 3);

// 创建带动量的 SGD 优化器
SGDOptimizer* opt = sgd_create(mlp, 0.01f, 0.9f, 0.0001f);

// 训练循环
for (size_t epoch = 0; epoch < 10; epoch++) {
    for (size_t b = 0; b < n_batches; b++) {
        float loss = mlp_train_step(mlp, opt, X_batch, y_batch);
        printf("Loss: %.4f\n", loss);
    }
}

// 评估
float acc = mlp_accuracy(mlp, X_test, y_test);
printf("Accuracy: %.2f%%\n", 100.0f * acc);

// 清理
sgd_free(opt);
mlp_free(mlp);
```

## GPU 版本 (OpenCL)

```c
#define CL_MLP_IMPLEMENTATION
#include "opencl_mlp.h"

// 初始化 OpenCL
CLOpenCL cl;
cl_init(&cl, CL_DEVICE_TYPE_GPU);

// 创建 GPU 模型
CLOpenCLMLP* mlp = cl_mlp_create(&cl, 784, 256, 10, 3);

// 创建 GPU 优化器
CLSGDOptimizer* opt = cl_sgd_create(&cl, mlp, 0.01f, 0.9f, 0.0001f);

// 在 GPU 上训练
for (...) {
    float loss = cl_mlp_train_step(&cl, &cl.kernel_cache, mlp, opt, X_gpu, y_gpu);
}

// 释放
cl_mlp_free(&cl, mlp);
cl_release(&cl);
```

## 说明

- 权重使用 Xavier 初始化
- 使用交叉熵损失进行分类
- 小批量梯度下降
- 梯度裁剪以保证稳定性
