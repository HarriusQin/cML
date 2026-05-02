# Dataset

Column-oriented data structure for machine learning.

## Overview

The `dataset` structure organizes data by columns rather than rows, which is more efficient for ML operations where you typically access all values of a specific feature.

## Data Structures

### feature_column

```c
typedef struct {
    size_t num_samples;
    double* data;           // Raw data values
    double min_val;
    double max_val;
    double mean;
    double std_dev;
} feature_column;
```

### label_column

```c
typedef struct {
    size_t num_samples;
    size_t* labels;         // Integer class indices
    size_t classes;         // Number of unique classes
    char** value_map;       // Maps index to string label
    size_t* class_counts;   // Count per class
} label_column;
```

### dataset

```c
typedef struct {
    size_t rows;                     // Number of samples
    size_t num_features;             // Number of feature columns
    size_t num_labels;              // Number of label columns
    feature_column* features;       // [num_features]
    label_column* labels;           // [num_labels]
} dataset;
```

## Functions

### Creation and Destruction

```c
dataset* create_dataset(size_t num_samples, size_t num_features, size_t num_labels);
void free_dataset(dataset* ds);
```

### CSV Conversion

```c
dataset* csv_to_dataset(csv_t* csv, const char** label_cols, size_t num_labels);
```

### Feature Access

```c
double dataset_get_feature(const dataset* ds, size_t feature_idx, size_t sample_idx);
void dataset_set_feature(dataset* ds, size_t feature_idx, size_t sample_idx, double value);
```

### Label Access

```c
size_t dataset_get_label(const dataset* ds, size_t label_idx, size_t sample_idx);
const char* dataset_get_label_string(const dataset* ds, size_t label_idx, size_t sample_idx);
```

### Train/Test Split

```c
typedef struct {
    size_t* train_indices;
    size_t* test_indices;
    size_t train_size;
    size_t test_size;
} Dataset_Split_t;

void train_test_split(const dataset* ds, float test_ratio, unsigned int seed,
                      Dataset_Split_t* split);
```

### Feature Scaling

```c
typedef enum {
    SCALING_STANDARD,    // (x - mean) / std_dev
    SCALING_MINMAX,      // (x - min) / (max - min)
    SCALING_NONE
} ScalingType;

ML_ScalingParams_t* ml_fit_scaling(const dataset* ds, size_t* feat_idx,
                                    size_t n_features, size_t* indices,
                                    size_t n_samples, ScalingType type);

dataset* ml_transform_features(const ML_ScalingParams_t* params,
                              const dataset* ds, size_t* feat_idx,
                              size_t n_features, size_t* indices,
                              size_t n_samples);

void ml_free_scaling_params(ML_ScalingParams_t* params);
```

## Example

```c
#define CSV_IMPLEMENTATION
#include "csv.h"

#define DATASET_IMPLEMENTATION
#include "dataset.h"

// Load CSV
csv_t* csv = csv_load("iris.csv");

// Convert to dataset
const char* labels[] = {"species"};
dataset* ds = csv_to_dataset(csv, labels, 1);

// Access features
printf("Sample 0, Feature 0: %f\n",
       dataset_get_feature(ds, 0, 0));

// Split data
Dataset_Split_t split;
train_test_split(ds, 0.2, 42, &split);

// Scale features
ML_ScalingParams_t* scaler = ml_fit_scaling(ds, NULL, ds->num_features,
                                            split.train_indices,
                                            split.train_size,
                                            SCALING_STANDARD);
```

## Notes

- Features are stored column-wise for efficient access
- Labels are converted to integer indices automatically
- String labels are stored in `value_map` for interpretation
