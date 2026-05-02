# LeNet-5

用于手写数字识别的经典卷积神经网络。

## 架构

```
输入: [N, 1, 32, 32]
    │
    ▼
C1: Conv(1, 6, 5×5) → [N, 6, 28, 28]
    │ 6 个滤波器, 步长 1
    ▼
S2: AvgPool(2×2) → [N, 6, 14, 14]
    │ 2×2 窗口, 步长 2
    ▼
C3: Conv(6, 16, 5×5) → [N, 16, 10, 10]
    │ 16 个滤波器
    ▼
S4: AvgPool(2×2) → [N, 16, 5, 5]
    ▼
C5: Conv(16, 120, 5×5) → [N, 120, 1, 1]
    ▼
展平 → [N, 120]
    │
    ▼
F6: FC(120, 84) → [N, 84]
    │
    ▼
输出: FC(84, 10) → [N, 10] (softmax)
```

## 超参数

| 层 | 参数 | 值 |
|-------|-----------|-------|
| C1 | 输入通道 | 1 |
| C1 | 输出通道 | 6 |
| C1 | 卷积核大小 | 5×5 |
| S2 | 池化大小 | 2×2 |
| S2 | 步长 | 2 |
| C3 | 输入通道 | 6 |
| C3 | 输出通道 | 16 |
| C3 | 卷积核大小 | 5×5 |
| S4 | 池化大小 | 2×2 |
| S4 | 步长 | 2 |
| C5 | 输出通道 | 120 |
| F6 | 隐藏单元 | 84 |
| 输出 | 类别数 | 10 |

## 数据结构

```c
typedef struct {
    ConvLayer* conv1;   // C1: 1→6 通道
    ConvLayer* conv2;   // C3: 6→16 通道
    ConvLayer* conv3;  // C5: 16→120 通道
    FCLayer* fc1;      // F6: 120→84
    FCLayer* fc2;      // 输出: 84→10
    bool training;
} LeNet5;
```

## 函数

```c
LeNet5* lenet5_create(void);
void lenet5_free(LeNet5* model);

Tensor* lenet5_forward(LeNet5* model, const Tensor* input);
float lenet5_train_step(LeNet5* model, float lr,
                         const Tensor* input, const Tensor* targets);
Tensor* lenet5_predict(LeNet5* model, const Tensor* input);
float lenet5_accuracy(LeNet5* model, const Tensor* input, const Tensor* targets);
```

## 示例

```c
#define LENET5_IMPLEMENTATION
#include "lenet5.h"

// 创建模型
LeNet5* model = lenet5_create();

// 训练循环
for (size_t epoch = 0; epoch < 10; epoch++) {
    float loss = lenet5_train_step(model, 0.01f, X_batch, y_batch);
    float acc = lenet5_accuracy(model, X_batch, y_batch);
    printf("Epoch %zu: loss=%.4f acc=%.2f%%\n", epoch, loss, 100*acc);
}

// 预测
Tensor* pred = lenet5_predict(model, X_test);

// 清理
tensor_free(pred);
lenet5_free(model);
```

## 说明

- 权重使用 Xavier 初始化
- 整个网络使用 tanh 激活（原始论文使用 sigmoid）
- 使用平均池化（不是最大池化）
- C3 和 S2 之间是稀疏连接（非全部 6→16 连接）
- 总共约 60K 参数
