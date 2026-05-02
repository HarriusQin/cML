# Multi-Layer Perceptron (MLP)

Feedforward neural network with fully connected layers.

## Architecture

```
Input (784) -> Hidden1 (256) -> Hidden2 (128) -> Output (10)
                |                |
              ReLU             ReLU
```

## Forward Pass

For each fully connected layer:
$$\mathbf{y} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$$

Where $\sigma$ is typically ReLU: $\sigma(x) = \max(0, x)$

## Backpropagation

1. Compute output error: $\delta^L = \nabla_a C \odot \sigma'(\mathbf{z}^L)$
2. Propagate error backward: $\delta^l = (\mathbf{W}^{l+1})^T \delta^{l+1} \odot \sigma'(\mathbf{z}^l)$
3. Compute gradients: $\frac{\partial C}{\partial \mathbf{W}^l} = \delta^l (\mathbf{a}^{l-1})^T$
4. Update weights using gradient descent with momentum

## Data Structures

### FCLayer (Fully Connected Layer)

```c
typedef struct {
    Tensor* weights;              // [output_dim, input_dim]
    Tensor* bias;                 // [output_dim]
    Tensor* grad_w;              // Weight gradient (for optimizer)
    Tensor* grad_b;              // Bias gradient
    Tensor* velocity_w;         // Momentum velocity (for SGD)
    Tensor* velocity_b;
    size_t input_dim;
    size_t output_dim;
    ActivationType activation;
} FCLayer;
```

### MLP

```c
typedef struct {
    FCLayer** layers;            // Array of layer pointers
    size_t num_layers;
    size_t input_dim;
    size_t output_dim;
} MLP;
```

### SGD Optimizer

```c
typedef struct {
    MLP* mlp;
    float lr;                    // Learning rate
    float momentum;              // Momentum coefficient
    float weight_decay;          // L2 regularization
} SGDOptimizer;
```

## Activation Types

```c
typedef enum {
    ACTIVATION_RELU,
    ACTIVATION_SIGMOID,
    ACTIVATION_TANH,
    ACTIVATION_SOFTMAX,
    ACTIVATION_NONE
} ActivationType;
```

## Functions

### Model Creation

```c
MLP* mlp_create(size_t input_dim, size_t hidden_dim, size_t output_dim, size_t num_layers);
```

### Training

```c
SGDOptimizer* sgd_create(MLP* mlp, float lr, float momentum, float weight_decay);

float mlp_train_step(MLP* mlp, SGDOptimizer* opt, Tensor* X, Tensor* y);
```

### Prediction

```c
Tensor* mlp_predict(MLP* mlp, Tensor* input);
float mlp_accuracy(MLP* mlp, Tensor* input, Tensor* labels);
```

### Memory Management

```c
void mlp_free(MLP* mlp);
void sgd_free(SGDOptimizer* opt);
```

## Example

```c
#define MLP_IMPLEMENTATION
#include "mlp.h"

// Create MLP: 784 -> 256 -> 128 -> 10
MLP* mlp = mlp_create(784, 256, 10, 3);

// Create SGD optimizer with momentum
SGDOptimizer* opt = sgd_create(mlp, 0.01f, 0.9f, 0.0001f);

// Training loop
for (size_t epoch = 0; epoch < 10; epoch++) {
    for (size_t b = 0; b < n_batches; b++) {
        float loss = mlp_train_step(mlp, opt, X_batch, y_batch);
        printf("Loss: %.4f\n", loss);
    }
}

// Evaluate
float acc = mlp_accuracy(mlp, X_test, y_test);
printf("Accuracy: %.2f%%\n", 100.0f * acc);

// Cleanup
sgd_free(opt);
mlp_free(mlp);
```

## GPU Version (OpenCL)

```c
#define CL_MLP_IMPLEMENTATION
#include "opencl_mlp.h"

// Initialize OpenCL
CLOpenCL cl;
cl_init(&cl, CL_DEVICE_TYPE_GPU);

// Create GPU model
CLOpenCLMLP* mlp = cl_mlp_create(&cl, 784, 256, 10, 3);

// Create GPU optimizer
CLSGDOptimizer* opt = cl_sgd_create(&cl, mlp, 0.01f, 0.9f, 0.0001f);

// Train on GPU
for (...) {
    float loss = cl_mlp_train_step(&cl, &cl.kernel_cache, mlp, opt, X_gpu, y_gpu);
}

// Free
cl_mlp_free(&cl, mlp);
cl_release(&cl);
```

## Notes

- Xavier initialization for weights
- Cross-entropy loss for classification
- Mini-batch gradient descent
- Gradient clipping for stability
