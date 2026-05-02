# Multiple Linear Regression (MLR)

Solves the problem: $\mathbf{y} = \mathbf{X}\mathbf{w}$

## Algorithm

1. Compute design matrix $\mathbf{X}$ with bias column
2. Solve normal equation using SVD for numerical stability:
   - $\mathbf{U}, \mathbf{\Sigma}, \mathbf{V} = \text{SVD}(\mathbf{X})$
   - $\mathbf{w} = \mathbf{V}^T \mathbf{\Sigma}^{-1} \mathbf{U}^T \mathbf{y}$

## Data Structures

```c
typedef struct {
    double* weights;          // [n_features + 1] including bias
    double* residuals;
    double rss;               // Residual sum of squares
    double r_squared;
} MLR_Weights_t;
```

## Configuration

```c
typedef struct {
    int verbose;              // Print progress
} MLR_Config_t;
```

## Functions

### Model Creation

```c
ML_Model_t create_linear_regression_model(void);
```

### Direct Fitting

```c
int MLR_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
            const dataset* ds, size_t* feat_idx, size_t n_features,
            size_t label_idx, size_t* indices, size_t n_samples);
```

### Prediction

```c
int MLR_predict(const ML_Weights_t* state, const dataset* ds,
               size_t* feat_idx, size_t n_features,
               size_t* indices, size_t n_samples, int* predictions);
```

## Example

```c
#define CSV_IMPLEMENTATION
#include "csv.h"
#define DATASET_IMPLEMENTATION
#include "dataset.h"
#define LINEAR_ALGEBRA_IMPLEMENTATION
#include "linear_algebra.h"
#define MLR_IMPLEMENTATION
#include "mlr.h"

// Load and prepare data
csv_t* csv = csv_load("housing.csv");
const char* labels[] = {"MEDV"};  // Target: median value
dataset* ds = csv_to_dataset(csv, labels, 1);

// Create and train model
ML_Model_t model = create_linear_regression_model();
model.methods.fit(&model.config, &model.state,
                  ds, NULL, ds->num_features, 0,
                  split.train_indices, split.train_size);

// Predict
int predictions[100];
model.methods.predict(&model.state, test_ds, NULL, n_features,
                      test_idx, 100, predictions);

// Free
model.methods.free_state(&model.state);
```

## Notes

- Uses SVD for numerical stability (handles multicollinearity)
- Automatically adds bias term
- Computes R-squared during training
- For classification, use [Softmax Regression](softmax_regression.md)
