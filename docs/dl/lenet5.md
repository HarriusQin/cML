# LeNet-5

Classic convolutional neural network for handwritten digit recognition.

## Overview

LeNet-5 (LeCun et al., 1998) is a pioneering CNN architecture for handwritten digit recognition (MNIST). It demonstrated that convolutional layers could learn translation-invariant features.

## Architecture

```
Input: [N, 1, 32, 32]
    │
    ▼
C1: Conv(1, 6, 5×5) → [N, 6, 28, 28]
    │ 6 filters, stride 1
    ▼
S2: AvgPool(2×2) → [N, 6, 14, 14]
    │ 2×2 window, stride 2
    ▼
C3: Conv(6, 16, 5×5) → [N, 16, 10, 10]
    │ 16 filters
    ▼
S4: AvgPool(2×2) → [N, 16, 5, 5]
    ▼
C5: Conv(16, 120, 5×5) → [N, 120, 1, 1]
    ▼
Flatten → [N, 120]
    │
    ▼
F6: FC(120, 84) → [N, 84]
    │
    ▼
Output: FC(84, 10) → [N, 10] (softmax)
```

## Data Structures

### ConvLayer

```c
typedef struct {
    Tensor* weight;      // [out_channels, in_channels, kh, kw]
    Tensor* bias;       // [out_channels]
    Tensor* grad_w;     // Weight gradients
    Tensor* grad_b;     // Bias gradients
    Tensor* input_cache;   // Saved for backward
    Tensor* output_cache;   // Saved for backward
} ConvLayer;
```

### FCLayer

```c
typedef struct {
    Tensor* weight;      // [out_features, in_features]
    Tensor* bias;        // [out_features]
    Tensor* grad_w;      // Weight gradients
    Tensor* grad_b;      // Bias gradients
    Tensor* input_cache; // For backward pass
    Tensor* preact_cache; // Pre-activation cache
} FCLayer;
```

### LeNet5 Model

```c
typedef struct {
    ConvLayer* conv1;   // C1: 1→6 channels
    ConvLayer* conv2;   // C3: 6→16 channels
    ConvLayer* conv3;  // C5: 16→120 channels
    FCLayer* fc1;      // F6: 120→84
    FCLayer* fc2;      // Output: 84→10
    bool training;
} LeNet5;
```

## Layer Creation

```c
// Convolutional layer
static ConvLayer* conv_layer_create(size_t in_ch, size_t out_ch,
                                    size_t kh, size_t kw);

// Fully connected layer
static FCLayer* fc_layer_create(size_t in_features, size_t out_features);
```

## Model Functions

```c
// Create/destroy
LeNet5* lenet5_create(void);
void lenet5_free(LeNet5* model);

// Forward/backward
Tensor* lenet5_forward(LeNet5* model, const Tensor* input);
void lenet5_backward(LeNet5* model, const Tensor* grad_output);

// Training
float lenet5_train_step(LeNet5* model, float lr,
                         const Tensor* input, const Tensor* targets);

// Prediction
Tensor* lenet5_predict(LeNet5* model, const Tensor* input);
float lenet5_accuracy(LeNet5* model, const Tensor* input,
                      const Tensor* targets);
```

## Forward Pass

Convolution with bias and tanh activation:

```c
Tensor* lenet5_forward(LeNet5* model, const Tensor* input) {
    // C1: Conv + tanh
    Conv2DParams conv1_params = {1, 1, 0, 0, 1, 1};
    Tensor* c1_out = tensor_conv2d(input, model->conv1->weight, &conv1_params);
    // Add bias, apply tanh
    tensor_tanh(c1_out);

    // S2: AvgPool 2×2 stride 2
    Tensor* s2_out = tensor_avgpool2d(c1_out, 2, 2, 2, 2);

    // C3: Conv + tanh
    Tensor* c3_out = tensor_conv2d(s2_out, model->conv2->weight, &conv3_params);
    tensor_tanh(c3_out);

    // S4: AvgPool
    Tensor* s4_out = tensor_avgpool2d(c3_out, 2, 2, 2, 2);

    // C5: Conv + tanh
    Tensor* c5_out = tensor_conv2d(s4_out, model->conv3->weight, &conv5_params);
    tensor_tanh(c5_out);

    // Flatten: [N, 120, 1, 1] → [N, 120]
    size_t flat_shape[] = {N, 120};
    Tensor* flat = tensor_reshape(c5_out, flat_shape, 2);

    // FC layers with tanh
    Tensor* f6_out = fc_layer_forward(model->fc1, flat);
    tensor_tanh(f6_out);
    Tensor* output = fc_layer_forward(model->fc2, f6_out);

    return output;
}
```

## Training Step

```c
float lenet5_train_step(LeNet5* model, float lr,
                       const Tensor* input, const Tensor* targets) {
    Tensor* output = lenet5_forward(model, input);
    tensor_softmax(output, 1);

    float loss = lenet5_cross_entropy(output, targets);
    lenet5_backward(model, output);

    // Gradient descent update for FC layers
    // (simplified - full conv backward requires im2col transpose)

    tensor_free(output);
    return loss;
}
```

## Example

```c
#define LENET5_IMPLEMENTATION
#include "lenet5.h"

// Create model
LeNet5* model = lenet5_create();

// Training loop
for (size_t epoch = 0; epoch < 10; epoch++) {
    float loss = lenet5_train_step(model, 0.01f, X_batch, y_batch);
    float acc = lenet5_accuracy(model, X_batch, y_batch);
    printf("Epoch %zu: loss=%.4f acc=%.2f%%\n", epoch, loss, 100*acc);
}

// Prediction
Tensor* pred = lenet5_predict(model, X_test);

// Cleanup
tensor_free(pred);
lenet5_free(model);
```

## Hyperparameters

| Layer | Parameter | Value |
|-------|-----------|-------|
| C1 | Input channels | 1 |
| C1 | Output channels | 6 |
| C1 | Kernel size | 5×5 |
| S2 | Pool size | 2×2 |
| S2 | Stride | 2 |
| C3 | Input channels | 6 |
| C3 | Output channels | 16 |
| C3 | Kernel size | 5×5 |
| S4 | Pool size | 2×2 |
| S4 | Stride | 2 |
| C5 | Output channels | 120 |
| F6 | Hidden units | 84 |
| Output | Classes | 10 |

## Notes

- Xavier initialization for weights
- tanh activation throughout (original paper used sigmoid)
- Average pooling (not max pooling)
- Sparse connectivity between C3 and S2 (not all 6→16 connections)
- ~60K parameters total
- Full backward pass requires conv transpose operations (simplified here)
