# 长短期记忆网络 (LSTM)

通过门控单元状态处理长距离依赖的循环网络。

## 概述

LSTM（Hochreiter & Schmidhuber，1997）通过门控内存单元解决 vanilla RNN 的梯度消失问题，这些单元可以有选择地记住或忘记很长时间序列的信息。

## 架构

```
输入: [batch, seq_len, input_size]
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  对于每个时间步 t:                                       │
│                                                         │
│  f_t = σ(W_f @ x_t + R_f @ h_{t-1} + b_f)  // 遗忘门    │
│  i_t = σ(W_i @ x_t + R_i @ h_{t-1} + b_i)  // 输入门    │
│  c_t = f_t * c_{t-1} + i_t * tanh(W_c @ x_t + R_c @ h_{t-1} + b_c) │
│  o_t = σ(W_o @ x_t + R_o @ h_{t-1} + b_o)  // 输出门    │
│  h_t = o_t * tanh(c_t)                                  │
└─────────────────────────────────────────────────────────┘
         │
         ▼
输出: [batch, seq_len, hidden_size]
```

## 门详解

### 遗忘门 ($f_t$)
决定从单元状态中丢弃哪些信息：
- $f_t = \sigma(\mathbf{W}_f \mathbf{x}_t + \mathbf{R}_f \mathbf{h}_{t-1} + \mathbf{b}_f)$
- 输出接近 1：保留信息
- 输出接近 0：忘记信息

### 输入门 ($i_t$)
决定存储哪些新信息：
- $i_t = \sigma(\mathbf{W}_i \mathbf{x}_t + \mathbf{R}_i \mathbf{h}_{t-1} + \mathbf{b}_i)$

### 单元候选 ($\tilde{c}_t$)
新的候选值：
- $\tilde{c}_t = \tanh(\mathbf{W}_c \mathbf{x}_t + \mathbf{R}_c \mathbf{h}_{t-1} + \mathbf{b}_c)$

### 输出门 ($o_t$)
决定输出什么：
- $o_t = \sigma(\mathbf{W}_o \mathbf{x}_t + \mathbf{R}_o \mathbf{h}_{t-1} + \mathbf{b}_o)$
- $\mathbf{h}_t = o_t \odot \tanh(c_t)$

## 数据结构

```c
typedef struct {
    // 输入权重 [hidden_size, input_size]
    Tensor* W_f, *W_i, *W_c, *W_o;

    // 循环权重 [hidden_size, hidden_size]
    Tensor* R_f, *R_i, *R_c, *R_o;

    // 偏置 [hidden_size]
    Tensor* b_f, *b_i, *b_c, *b_o;

    // ... 梯度和缓存 ...
} LSTMLayer;

typedef struct {
    LSTMLayer* layer;
    size_t input_size;
    size_t hidden_size;
    size_t num_layers;
    bool training;
    float clip_threshold;  // 梯度裁剪
} LSTM;
```

## 函数

```c
LSTM* lstm_create(size_t input_size, size_t hidden_size, size_t num_layers);
void lstm_free(LSTM* model);

Tensor* lstm_forward(LSTM* model, const Tensor* input);
void lstm_backward(LSTM* model, const Tensor* grad_output);
float lstm_train_step(LSTM* model, float lr,
                       const Tensor* input, const Tensor* targets);
Tensor* lstm_predict(LSTM* model, const Tensor* input);
```

## 示例

```c
#define LSTM_IMPLEMENTATION
#include "lstm.h"

// 创建 LSTM: input=100, hidden=256, 1 层
LSTM* model = lstm_create(100, 256, 1);

// 输入: batch=32, seq_len=50, input_size=100
size_t input_shape[] = {32, 50, 100};
Tensor* input = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW,
                              input_shape, 3);

// 训练
for (size_t epoch = 0; epoch < 20; epoch++) {
    float loss = lstm_train_step(model, 0.001f, input, targets);
    printf("Epoch %zu: loss=%.4f\n", epoch, loss);
}

// 预测
Tensor* output = lstm_predict(model, input);

lstm_free(model);
```

## 梯度流

LSTM 的关键是细胞状态的不间断梯度流：

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$

由于 $f_t \in [0, 1]$，梯度被缩放但不像 vanilla RNN 那样指数衰减。

## 说明

- 梯度裁剪防止梯度爆炸（阈值：5.0）
- 输入权重使用 Xavier 初始化，循环权重使用正交初始化
- Sigmoid 用于门，tanh 用于细胞状态和输出
- 单元状态使用阿达马乘积 (*)，不是矩阵乘法
