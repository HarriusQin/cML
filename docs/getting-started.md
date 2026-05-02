# Getting Started

## Building the Library

```bash
# Build all targets
make all

# Build specific target
make bin/test_mlp

# Run tests
make test         # Basic ML tests
make test_dl     # Deep learning tests

# Clean build artifacts
make clean
```

## Dependencies

- GCC with C99 support
- OpenCL (optional, for GPU acceleration)
- Math library (-lm)
- OpenCL library (-lOpenCL, optional)

## Basic Usage

### 1. Load Data with CSV

```c
#define CSV_IMPLEMENTATION
#include "csv.h"

csv_t* csv = csv_load("data.csv");
if (!csv) {
    fprintf(stderr, "Failed to load CSV\n");
    return 1;
}
```

### 2. Convert to Dataset

```c
#define DATASET_IMPLEMENTATION
#include "dataset.h"

const char* labels[] = {"species"};
dataset* ds = csv_to_dataset(csv, labels, 1);
```

### 3. Train a Model

```c
#define SOFTMAX_REGRESSION_IMPLEMENTATION
#include "softmax_regression.h"

// Create model
ML_Model_t model = create_softmax_model();

// Train
int train_idx[] = {0, 1, 2, ...};  // Training indices
model.methods.fit(&model.config, &model.state,
                  train_ds, feat_idx, n_features, label_idx,
                  train_idx, train_size);

// Predict
int predictions[100];
model.methods.predict(&model.state, test_ds, feat_idx, n_features,
                      test_idx, test_size, predictions);
```

### 4. Evaluate

```c
double accuracy = compute_accuracy(&model.state, test_ds, ...);
printf("Accuracy: %.2f%%\n", accuracy * 100);
```

## Deep Learning Example (MNIST)

```c
#define MLP_IMPLEMENTATION
#include "mlp.h"

// Create MLP: 784 -> 256 -> 128 -> 10
MLP* mlp = mlp_create(784, 256, 10, 3);

// Train with SGD
SGDOptimizer* opt = sgd_create(mlp, 0.01f, 0.9f, 0.0001f);
float loss = mlp_train_step(mlp, opt, X_batch, y_batch);

// Predict
Tensor* pred = mlp_predict(mlp, X_batch);
```

## OpenCL Example

```c
#define CL_MLP_IMPLEMENTATION
#include "opencl_mlp.h"

CLOpenCL cl;
cl_init(&cl, CL_DEVICE_TYPE_GPU);

// Create OpenCL MLP
CLOpenCLMLP* mlp = cl_mlp_create(&cl, 784, 256, 10, 3);

// Train on GPU
float loss = cl_mlp_train_step(&cl, &cl.kernel_cache, mlp, opt, X_gpu, y_gpu);
```

## Data Format

### CSV Format
```csv
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
```

### Dataset Structure
- `dataset.features[]`: Array of `feature_column` (column-oriented)
- `dataset.labels[]`: Array of `label_column`
- `dataset.rows`: Number of samples

## Common Patterns

### Feature Scaling
```c
ML_ScalingParams_t* scaler = ml_fit_scaling(ds, feat_idx, n_features,
                                            train_idx, train_size,
                                            SCALING_STANDARD);
dataset* scaled = ml_transform_features(scaler, ds, feat_idx, n_features,
                                       test_idx, test_size);
```

### Train/Test Split
```c
Dataset_Split_t split;
train_test_split(ds, 0.2, 42, &split);  // 80/20 split
```

### Cross-Validation
```c
CV_Result_t cv = kfold_cross_validate(&model, ds, 5);  // 5-fold CV
```
