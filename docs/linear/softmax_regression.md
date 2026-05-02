# Softmax Regression

Multi-class classification using one-vs-all softmax. For each class $k$:
$$P(y=k|\mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}}}$$

## Algorithm

1. For K classes, train K binary classifiers (one-vs-all)
2. Each classifier $k$ solves: $P(y=k) = \sigma(\mathbf{w}_k^T \mathbf{x})$
3. At prediction: choose class with highest probability

## Two Training Methods

### Normal Equation (Closed-form)

```c
ML_Model_t create_softmax_model(void);
```
Uses SVD for stability. Fast but memory-intensive for large K.

### Gradient Descent

```c
ML_Model_t create_softmax_model_gd(void);
```
Iterative optimization. Memory-efficient for large K.

## Configuration (Gradient Descent)

```c
typedef struct {
    double learning_rate;     // Default: 0.1
    size_t max_iter;          // Default: 500
    double tolerance;         // Default: 1e-5
    int verbose;              // Print progress
} SoftmaxReg_Config;
```

## Data Structures

```c
typedef struct {
    double** weights;         // [K][n_features] class weights
    double* bias;             // [K] class biases
    size_t n_classes;
    double learning_rate;
} SoftmaxReg_Weights_t;
```

## Functions

```c
ML_Model_t create_softmax_model(void);
ML_Model_t create_softmax_model_gd(void);

// Probability prediction
int softmax_predict_proba(const SoftmaxReg_Weights_t* state,
                          const dataset* ds, size_t* feat_idx,
                          size_t n_features, size_t* indices,
                          size_t n_samples, float* probabilities);
```

## Example

```c
#define SOFTMAX_REGRESSION_IMPLEMENTATION
#include "softmax_regression.h"

// Create model (gradient descent variant)
ML_Model_t model = create_softmax_model_gd();

// Configure
SoftmaxReg_Config config = {
    .learning_rate = 0.1,
    .max_iter = 500,
    .tolerance = 1e-5,
    .verbose = 1
};
model.config.params = &config;
model.config.size = sizeof(config);

// Train
model.methods.fit(&model.config, &model.state,
                   train_ds, feat_idx, n_features, label_idx,
                   train_idx, train_size);

// Predict class labels
int predictions[100];
model.methods.predict(&model.state, test_ds, feat_idx, n_features,
                      test_idx, 100, predictions);

// Get probabilities
float probs[100 * 3];  // 100 samples, 3 classes
softmax_predict_proba(&model.state, test_ds, feat_idx, n_features,
                      test_idx, 100, probs);
```

## Notes

- Automatically adds bias term
- One-vs-all approach allows multi-class without modifying algorithm
- Gradient descent variant is more memory-efficient for many classes
- Feature scaling improves convergence for gradient descent
