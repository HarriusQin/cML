# cML - C Machine Learning Library

A pure C implementation of machine learning algorithms, including deep learning with OpenCL GPU acceleration.

## Project Structure

```
cML/
├── dl/                    # Deep Learning
│   ├── mlp.h             # Multi-Layer Perceptron
│   ├── tensor.h          # Tensor operations
│   ├── opencl_mlp.h      # OpenCL-accelerated MLP
│   ├── opencl_tensor.h    # OpenCL tensor operations
│   ├── lenet5.h          # LeNet-5 CNN
│   ├── lstm.h            # Long Short-Term Memory
│   ├── rnn.h             # Recurrent Neural Network
│   └── test/             # DL tests
├── ensemble/             # Ensemble Methods
│   ├── adaboost.h        # AdaBoost
│   ├── randomforest.h    # Random Forest
│   ├── xgboost.h        # XGBoost
│   └── catboost.h        # CatBoost
├── linear/               # Linear Models
│   ├── mlr.h             # Multiple Linear Regression
│   ├── ridge_regression.h # Ridge Regression (L2)
│   ├── polynomial_regression.h # Polynomial Regression
│   ├── softmax_regression.h # Softmax Regression
│   ├── linear_algebra.h  # Matrix operations
│   └── wls.h             # Weighted Least Squares
├── test/                 # General ML tests
├── common/               # Utilities
│   ├── dataset.h         # Dataset structure
│   ├── csv.h             # CSV parsing
│   ├── machine_learning.h # Unified ML interface
│   └── utilities.h       # Helper functions
├── idx.h                 # MNIST idx file format
├── gnb.h                 # Gaussian Naive Bayes
└── decision_tree.h       # Decision Tree
```

## Features

- **Deep Learning**: MLP, CNN (LeNet-5), RNN, LSTM with OpenCL GPU acceleration
- **Ensemble Methods**: AdaBoost, Random Forest, XGBoost, CatBoost
- **Linear Models**: MLR, Ridge, Polynomial, Softmax, WLS
- **Pure C**: No external dependencies beyond standard C library
- **OpenCL Support**: GPU acceleration for tensor operations

## Quick Start

```c
#define CSV_IMPLEMENTATION
#include "csv.h"

#define DATASET_IMPLEMENTATION
#include "dataset.h"

#define SOFTMAX_REGRESSION_IMPLEMENTATION
#include "softmax_regression.h"

// Load data
csv_t* csv = csv_load("data.csv");
const char* labels[] = {"label_col"};
dataset* ds = csv_to_dataset(csv, labels, 1);

// Create and train model
ML_Model_t model = create_softmax_model();
model.methods.fit(&model.config, &model.state, ds, ...);

// Predict
int predictions[100];
model.methods.predict(&model.state, test_ds, ..., predictions);
```

## Building

```bash
make all          # Build all targets
make test         # Run basic tests
make test_dl      # Run deep learning tests
```

## Documentation

- [Getting Started](getting-started.md)
- [Common Utilities](common/)
  - [Dataset](common/dataset.md)
  - [Tensor](common/tensor.md)
  - [IDX Format](common/idx.md)
  - [Utilities](common/utilities.md)
- [Linear Models](linear/)
  - [MLR](linear/mlr.md)
  - [Ridge Regression](linear/ridge_regression.md)
  - [Polynomial Regression](linear/polynomial_regression.md)
  - [Softmax Regression](linear/softmax_regression.md)
  - [Weighted Least Squares](linear/wls.md)
- [Ensemble Methods](ensemble/)
  - [AdaBoost](ensemble/adaboost.md)
  - [Random Forest](ensemble/randomforest.md)
  - [XGBoost](ensemble/xgboost.md)
  - [CatBoost](ensemble/catboost.md)
- [Deep Learning](dl/)
  - [MLP](dl/mlp.md)
  - [OpenCL Tensor](dl/opencl_tensor.md)
  - [LeNet-5](dl/lenet5.md)
  - [LSTM](dl/lstm.md)
  - [RNN](dl/rnn.md)
- [API Reference](api/full_api_reference.md)
