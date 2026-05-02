# 深度学习

带有 OpenCL GPU 加速的神经网络模型。

## 模型

| 模型 | 文件 | 描述 |
|-------|------|-------------|
| MLP | [mlp.md](mlp.md) | 多层感知机 |
| LeNet-5 | [lenet5.md](lenet5.md) | 用于图像分类的 CNN |
| RNN | [rnn.md](rnn.md) | 简单循环神经网络 |
| LSTM | [lstm.md](lstm.md) | 长短期记忆网络 |

## GPU 加速

关键运算的 OpenCL 版本：

| CPU | GPU (OpenCL) |
|-----|--------------|
| tensor.h | [opencl_tensor.h](opencl_tensor.md) |
| [mlp.h](mlp.md) | [opencl_mlp.h](opencl_mlp.md) |

## 核心概念

### 张量运算
参见 [tensor.md](../common/tensor.md) 了解张量数据结构和运算。

### 训练循环
```c
// 前向传播
Tensor* output = model_forward(model, input);

// 计算损失
float loss = cross_entropy(output, targets);

// 反向传播
model_backward(model, loss);

// 更新权重
optimizer_update(optimizer, model);
```

### 层
- **FCLayer**: 全连接层 (y = Wx + b)
- **Conv2D**: 2D 卷积
- **MaxPool2D**: 2D 最大池化
- **LSTM**: 长短期记忆单元
