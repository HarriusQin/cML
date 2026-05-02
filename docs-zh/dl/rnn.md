# 循环神经网络 (RNN)

用于序列数据处理的简单 Elman RNN。

## 概述

 vanilla RNN（Elman 网络）通过维护隐藏状态来处理序列，该隐藏状态捕获前面时间步的上下文。与 LSTM 不同，它具有简单的循环结构，容易出现梯度消失。

## 架构

```
输入: [batch, seq_len, input_size]
         │
         ▼
┌────────────────────────────────────────────┐
│  h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)  │
│                                            │
│  其中 h_{-1} = 0（初始隐藏状态）             │
└────────────────────────────────────────────┘
         │
         ▼
输出: [batch, seq_len, hidden_size]
```

## 前向传播

对于每个时间步：
$$\mathbf{h}_t = \tanh(\mathbf{W}_{ih}\mathbf{x}_t + \mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{b})$$

隐藏状态 $\mathbf{h}_t$ 既传递到输出，也传递到下一个时间步。

## 数据结构

```c
typedef struct {
    Tensor* W_ih;   // [hidden_size, input_size] 输入到隐藏
    Tensor* W_hh;   // [hidden_size, hidden_size] 隐藏到隐藏
    Tensor* b;      // [hidden_size] 偏置
    Tensor* grad_W_ih;
    Tensor* grad_W_hh;
    Tensor* grad_b;
    Tensor** h_cache;   // 每个时间步的隐藏状态
    Tensor** x_cache;   // 每个时间步的输入
    size_t seq_len;
} RNNLayer;

typedef struct {
    RNNLayer* layer;
    size_t input_size;
    size_t hidden_size;
    size_t num_layers;
    bool training;
} RNN;
```

## 函数

```c
RNN* rnn_create(size_t input_size, size_t hidden_size, size_t num_layers);
void rnn_free(RNN* model);

Tensor* rnn_forward(RNN* model, const Tensor* input);
void rnn_backward(RNN* model, const Tensor* grad_output);
float rnn_train_step(RNN* model, float lr,
                      const Tensor* input, const Tensor* targets);
Tensor* rnn_predict(RNN* model, const Tensor* input);
```

## 示例

```c
#define RNN_IMPLEMENTATION
#include "rnn.h"

// 创建 RNN: input=50, hidden=128, 1 层
RNN* model = rnn_create(50, 128, 1);

// 输入: batch=16, seq_len=30, input_size=50
size_t input_shape[] = {16, 30, 50};
Tensor* input = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW,
                              input_shape, 3);

// 使用 MSE 损失的训练循环
for (size_t epoch = 0; epoch < 50; epoch++) {
    float loss = rnn_train_step(model, 0.001f, input, targets);
    printf("Epoch %zu: loss=%.4f\n", epoch, loss);
}

// 预测
Tensor* output = rnn_predict(model, input);

rnn_free(model);
```

## 梯度消失问题

在 vanilla RNN 中，梯度在每个步骤乘以 $\tanh'(\mathbf{z}_t) \cdot \mathbf{W}_{hh}$：

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}} = \sum_{t=0}^{T-1} \left( \prod_{k=t+1}^{T-1} \tanh'(\mathbf{z}_k) \mathbf{W}_{hh} \right) \frac{\partial \mathcal{L}}{\partial \mathbf{h}_t}$$

由于 $\|\tanh'(\mathbf{z})\| \leq 1$，且 $\mathbf{W}_{hh}$ 的特征值通常 $< 1$，乘积呈指数衰减。

## 与 LSTM 的比较

| 方面 | RNN | LSTM |
|--------|-----|------|
| 记忆 | 单个隐藏状态 | 单元状态 + 门 |
| 梯度流 | 可能消失 | 通过单元状态不间断 |
| 门 | 无 | 遗忘、输入、输出 |
| 参数 | 更少 | 4 倍以上 |
| 训练难度 | 较难 | 较易 |
| 序列长度 | 短（< 50） | 长（< 1000） |

## 说明

- 梯度裁剪防止梯度爆炸
- Xavier 初始化: $\text{std} = \sqrt{2/(n_{in} + n_{out})}$
- tanh 激活将值保持在 $[-1, 1]$
- 对于深层 RNNs，堆叠多个 RNNLayer 对象
- 对于长序列，通常使用截断 BPTT（忽略长距离依赖）
