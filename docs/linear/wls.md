# Weighted Least Squares (WLS)

Regression with sample-specific weights for heteroscedastic data.

## Overview

Weighted Least Squares (WLS) extends ordinary least squares by assigning different weights to different samples. This is essential when observations have different variances or are measured with different precisions.

## Problem Formulation

Standard LS minimizes: $\sum_{i=1}^n (y_i - \mathbf{x}_i^T\boldsymbol{\theta})^2$

WLS minimizes: $\sum_{i=1}^n w_i (y_i - \mathbf{x}_i^T\boldsymbol{\theta})^2$

Where $w_i$ is the weight for sample $i$, often proportional to $1/\sigma_i^2$.

## Solution

The weighted normal equation:

$$\boldsymbol{\theta} = (\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^T\mathbf{W}\mathbf{y}$$

Where $\mathbf{W} = \text{diag}(w_1, w_2, \ldots, w_n)$.

This can be rewritten as solving OLS on the weighted problem:

Let $\mathbf{W}^{1/2}$ be the diagonal matrix with $\sqrt{w_i}$, then:

$$\tilde{\mathbf{y}} = \mathbf{W}^{1/2}\mathbf{y}$$
$$\tilde{\mathbf{X}} = \mathbf{W}^{1/2}\mathbf{X}$$

Solve: $\boldsymbol{\theta} = (\tilde{\mathbf{X}}^T\tilde{\mathbf{X}})^{-1}\tilde{\mathbf{X}}^T\tilde{\mathbf{y}}$

## Data Structure

```c
typedef struct WLS_State {
    Matrix* weights;   // Coefficient matrix [1, n_features]
    double* bias;     // Intercept term
} WLS_State;
```

## Functions

```c
// Fit weighted linear regression
static int wls_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
                   const dataset* ds, const size_t* feature_indices,
                   size_t n_features, size_t target_index,
                   const size_t* sample_indices, size_t n_samples,
                   const double* weights);  // Sample weights

// Predict using fitted model
static int wls_predict(const ML_Weights_t* state, const dataset* ds,
                        const size_t* feature_indices, size_t n_features,
                        const size_t* sample_indices, size_t n_samples,
                        void* output);
```

## Weight Matrix Construction

```c
// Compute sqrt of weights for W^{1/2}
static int weighted_X(const Matrix* X, const double* w_sqrt, Matrix* result) {
    // result[i,j] = X[i,j] * w_sqrt[i]
    for (size_t i = 0; i < X->rows; i++) {
        double wi = w_sqrt[i];
        for (size_t j = 0; j < X->cols; j++)
            result->mat[i * X->cols + j] = X->mat[i * X->cols + j] * wi;
    }
    return 0;
}
```

## Common Weight Schemes

### Inverse Variance Weighting

When $\sigma_i$ is known for each measurement:

$$w_i = \frac{1}{\sigma_i^2}$$

### Statistical Weighting

For count data with variance proportional to mean:

$$w_i = \frac{n_i}{\hat{y}_i}$$

### Custom Weights

Based on domain knowledge or measurement reliability.

## Example

```c
#define WLS_IMPLEMENTATION
#include "wls.h"

// Sample weights based on measurement uncertainty
double weights[] = {1.0, 1.0, 0.5, 0.5, 0.25, 0.25};  // Lower weight for noisy samples

ML_Weights_t state = {0};
wls_fit(NULL, &state, ds, feat_idx, n_features, target_idx,
        train_idx, n_samples, weights);

// Predict
double predictions[100];
wls_predict(&state, test_ds, feat_idx, n_features,
           test_idx, 100, predictions);
```

## Comparison with OLS

| Aspect | OLS | WLS |
|--------|-----|-----|
| Objective | $\sum (y - \hat{y})^2$ | $\sum w_i(y - \hat{y})^2$ |
| Variance assumption | Homoscedastic | Heteroscedastic |
| Efficient | When weights correct | Always with correct weights |
| Standard errors | Homoscedastic formula | Weighted formula |

## Heteroscedasticity Detection

Use the Breusch-Pagan test:

1. Fit OLS, compute residuals $e_i$
2. Regress $e_i^2$ on predictors
3. If $R^2 \times n \times 2 \sim \chi^2_1$, heteroscedasticity present

```c
double compute_breusch_pagan_residuals(const Tensor* residuals,
                                       const Tensor* X) {
    // Squared residuals regressed on features
    // Returns test statistic
}
```

## Notes

- Weights should be proportional to inverse variance for efficient estimates
- WLS is a special case of Generalized Least Squares (GLS)
- If weights are unknown, use feasible GLS (FGLS) with estimated weights
- SVD-based pseudoinverse used when $\mathbf{X}^T\mathbf{W}\mathbf{X}$ is singular
- Weights can be updated iteratively ( IRLS - Iteratively Reweighted Least Squares )
