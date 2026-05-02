# Deep Learning

Neural network models for deep learning, with optional OpenCL GPU acceleration.

## Models

| Model | File | Description |
|-------|------|-------------|
| MLP | [mlp.md](mlp.md) | Multi-Layer Perceptron |
| LeNet-5 | [lenet5.md](lenet5.md) | CNN for image classification |
| RNN | [rnn.md](rnn.md) | Vanilla Recurrent Neural Network |
| LSTM | [lstm.md](lstm.md) | Long Short-Term Memory |

## GPU Acceleration

OpenCL versions of key operations:

| CPU | GPU (OpenCL) |
|-----|--------------|
| tensor.h | [opencl_tensor.h](opencl_tensor.md) |
| [mlp.h](mlp.md) | [opencl_mlp.h](opencl_mlp.md) |

## Core Concepts

### Tensor Operations
See [tensor.md](../common/tensor.md) for tensor data structure and operations.

### Training Loop
```c
// Forward pass
Tensor* output = model_forward(model, input);

// Compute loss
float loss = cross_entropy(output, targets);

// Backward pass
model_backward(model, loss);

// Update weights
optimizer_update(optimizer, model);
```

### Layers
- **FCLayer**: Full connected layer (y = Wx + b)
- **Conv2D**: 2D convolution
- **MaxPool2D**: 2D max pooling
- **LSTM**: Long Short-Term Memory cell
