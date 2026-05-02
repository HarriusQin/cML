# Polynomial Regression

Extends linear regression to capture non-linear relationships.

## Overview

Polynomial regression extends the linear model by adding powers of each feature as new features, enabling the model to fit curves while still using linear regression's normal equation.

## Algorithm

For a polynomial of degree $d$:

$$y_i = \theta_0 + \theta_1 x_i + \theta_2 x_i^2 + \cdots + \theta_d x_i^d + \epsilon_i$$

The design matrix is augmented with polynomial terms:

$$\mathbf{X} = \begin{bmatrix} 1 & x_1 & x_1^2 & \cdots & x_1^d \\ 1 & x_2 & x_2^2 & \cdots & x_2^d \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_n & x_n^2 & \cdots & x_n^d \end{bmatrix}$$

Then solve normal equation: $\boldsymbol{\theta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$

## Data Structure

```c
typedef struct PolyReg_State {
    Matrix* weights;    // Polynomial coefficients [degree]
    double* bias;       // Intercept term
    size_t degree;      // Polynomial degree
} PolyReg_State;
```

## Functions

```c
// Generate polynomial features up to specified degree
static int polynomial_features(
    const double* x,       // Input [n_samples]
    size_t n_samples,
    size_t degree,
    Matrix* result          // Output [n_samples, degree]
);

// Fit polynomial regression
static int poly_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
                    const double* x, const double* y,
                    size_t n_samples, size_t degree);

// Predict using fitted model
static int poly_predict(const ML_Weights_t* state,
                         const double* x, size_t n_samples,
                         double* output);
```

## Feature Generation

```c
static int polynomial_features(
    const double* x, size_t n_samples,
    size_t degree, Matrix* result) {
    // result: [n_samples, degree]
    // Each row: [x^1, x^2, ..., x^degree]
    for (size_t i = 0; i < n_samples; i++) {
        double x_pow = x[i];  // x^1
        for (size_t d = 0; d < degree; d++) {
            result->mat[i * degree + d] = x_pow;
            x_pow *= x[i];    // x^(d+2)
        }
    }
    return 0;
}
```

## Example

```c
#define POLYNOMIAL_REGRESSION_IMPLEMENTATION
#include "polynomial_regression.h"

// Generate data: y = 3 + 2x + x^2 + noise
double x[] = {0, 1, 2, 3, 4, 5};
double y[] = {3.1, 6.2, 13.0, 22.1, 35.0, 50.2};

// Fit degree-2 polynomial
PolyReg_State state = {0};
poly_fit(NULL, &state, x, y, 6, 2);

// Predict
double x_test[] = {1.5, 2.5, 3.5};
double y_pred[3];
poly_predict(&state, x_test, 3, y_pred);

// Free
poly_free(&state);
```

## Bias-Variance Tradeoff

| Degree | Bias | Variance | Overfitting Risk |
|--------|------|----------|-----------------|
| 1 (linear) | High | Low | Underfitting |
| 2-3 | Medium | Medium | Good balance |
| High (>10) | Low | High | Overfitting |

## Choosing Degree

Use cross-validation to select optimal degree:

```c
double best_score = -INFINITY;
size_t best_degree = 1;

for (size_t deg = 1; deg <= 10; deg++) {
    double cv_score = cross_validate(poly_fit, poly_predict,
                                    x_train, y_train, deg, k_folds);
    if (cv_score > best_score) {
        best_score = cv_score;
        best_degree = deg;
    }
}
```

## Notes

- Always normalize polynomial features for numerical stability
- Higher degrees increase overfitting risk dramatically
- Use regularization (ridge) for high-degree polynomials
- For multivariate data, polynomial features become: $1, x_1, x_2, x_1^2, x_1 x_2, x_2^2, \ldots$
- Normal equation is solved using SVD for ill-conditioned matrices
