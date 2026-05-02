# Linear Models

Models that learn linear relationships between features and targets.

## Models

| Model | File | Type | Use Case |
|-------|------|------|----------|
| Multiple Linear Regression | [mlr.h](mlr.md) | Regression | Continuous targets |
| Ridge Regression | [ridge_regression.h](ridge_regression.md) | Regression | L2 regularized |
| Polynomial Regression | [polynomial_regression.h](polynomial_regression.md) | Regression | Non-linear relationships |
| Softmax Regression | [softmax_regression.md](softmax_regression.md) | Classification | Multi-class |
| Weighted Least Squares | [wls.md](wls.md) | Regression | Heteroscedastic data |

## Common Properties

- All solve variations of the normal equation: $\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$
- Feature scaling recommended for gradient descent variants
- Ridge adds L2 regularization: $\mathbf{w} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$

## Feature Scaling

Most linear models benefit from feature scaling:

```c
ML_ScalingParams_t* scaler = ml_fit_scaling(ds, feat_idx, n_features,
                                            train_idx, train_size,
                                            SCALING_STANDARD);
dataset* scaled_train = ml_transform_features(scaler, ds, feat_idx, n_features,
                                              train_idx, train_size);
```
