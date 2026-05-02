# AdaBoost

Adaptive Boosting - combines weak classifiers (decision stumps) via reweighted sampling.

## Algorithm

1. Initialize weights $w_i = 1/N$ for all samples
2. For each boosting round $t = 1, \ldots, T$:
   - Train weak classifier on weighted samples
   - Compute weighted error: $\epsilon_t = \sum_{i: h_t(x_i) \neq y_i} w_i$
   - Compute stage weight: $\alpha_t = \frac{1}{2}\ln\frac{1-\epsilon_t}{\epsilon_t}$
   - Update weights: $w_i \leftarrow w_i e^{-\alpha_t y_i h_t(x_i)}$
   - Normalize weights
3. Final prediction: $H(x) = \text{sign}\left(\sum_t \alpha_t h_t(x)\right)$

## Data Structures

```c
#define ADABOOST_MAX_ESTIMATORS 50

typedef struct {
    int n_estimators;
    double* estimator_weights;     // [n_estimators] stage weights
    DecisionStump** estimators;    // Weak learners
} Adaboost_Weights_t;
```

## Configuration

```c
typedef struct {
    int n_estimators;             // Max weak learners (default: 50)
    double learning_rate;          // Shrinkage (default: 1.0)
} Adaboost_Config_t;
```

## Functions

```c
ML_Model_t create_adaboost_model(void);

int Adaboost_fit(const ML_Model_Config_t* config, Adaboost_Weights_t* state,
                const dataset* ds, size_t* feat_idx, size_t n_features,
                size_t label_idx, size_t* indices, size_t n_samples);

int Adaboost_predict(const Adaboost_Weights_t* state, const dataset* ds,
                     size_t* feat_idx, size_t n_features,
                     size_t* indices, size_t n_samples, int* predictions);
```

## Example

```c
#define ADABOOST_IMPLEMENTATION
#include "adaboost.h"

ML_Model_t model = create_adaboost_model();

Adaboost_Config_t config = {
    .n_estimators = 100,
    .learning_rate = 0.5
};
model.config.params = &config;
model.config.size = sizeof(config);

model.methods.fit(&model.config, &model.state,
                  train_ds, feat_idx, n_features, label_idx,
                  train_idx, train_size);

int predictions[100];
model.methods.predict(&model.state, test_ds, feat_idx, n_features,
                     test_idx, 100, predictions);
```

## Decision Stump

A depth-1 decision tree that splits on one feature:

```c
typedef struct {
    size_t feature_idx;           // Which feature to split on
    double threshold;             // Split threshold
    int left_label;              // Label for values < threshold
    int right_label;             // Label for values >= threshold
    double gain;                 // Information gain from split
} DecisionStump;
```

## Notes

- Uses decision stumps as weak learners (configurable)
- Sensitive to noise and outliers (all misclassified samples get increased weight)
- Learning rate < 1.0 reduces overfitting but requires more estimators
- For high-dimensional data, consider Random Forest instead
