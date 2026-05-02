# Ridge Regression

Linear regression with L2 regularization. Solves:
$$\mathbf{w} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$

## Algorithm

1. Add bias column to design matrix $\mathbf{X}$
2. Compute regularized normal equation: $\mathbf{w} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$
3. Use SVD for numerical stability

## Configuration

```c
typedef struct {
    double alpha;             // Regularization strength (default: 1.0)
    int verbose;
} Ridge_Config_t;
```

## Data Structures

```c
typedef struct {
    double* weights;          // [n_features + 1]
    double alpha;             // Regularization parameter
    double rss;
    double r_squared;
} Ridge_Weights_t;
```

## Functions

### Model Creation

```c
ML_Model_t create_ridge_regression_model(double alpha);
```

### Direct Usage

```c
int Ridge_fit(const ML_Model_Config_t* config, Ridge_Weights_t* state,
              const dataset* ds, size_t* feat_idx, size_t n_features,
              size_t label_idx, size_t* indices, size_t n_samples);

int Ridge_predict(const Ridge_Weights_t* state, const dataset* ds,
                  size_t* feat_idx, size_t n_features,
                  size_t* indices, size_t n_samples, double* predictions);
```

## Example

```c
#define RIDGE_REGRESSION_IMPLEMENTATION
#include "ridge_regression.h"

// Create with alpha=0.5
ML_Model_t model = create_ridge_regression_model(0.5);

Ridge_Config_t config = {.alpha = 0.5, .verbose = 1};
model.config.params = &config;
model.config.size = sizeof(config);

model.methods.fit(&model.config, &model.state, ds, ...);

// Get coefficients
double coef[5];
model.methods.get_coefficients(&model.state, coef);
```

## When to Use

- **Multicollinearity**: When features are correlated, vanilla MLR becomes unstable
- **Overfitting**: L2 regularization penalizes large weights
- **High dimensionality**: When n_features approaches or exceeds n_samples

## Notes

- α=0: Equivalent to ordinary least squares
- α→∞: All weights approach zero
- Features should be scaled for best results
