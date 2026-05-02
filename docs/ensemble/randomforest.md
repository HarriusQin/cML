# Random Forest

Bootstrap aggregating (bagging) of decision trees with random feature selection.

## Algorithm

1. For $b = 1, \ldots, B$ trees:
   - Draw bootstrap sample of size $N$ (sampling with replacement)
   - Train decision tree on bootstrap sample
   - At each node, randomly select $m$ features
   - Split on best feature among the $m$ selected
2. Aggregate predictions:
   - **Classification**: Majority vote
   - **Regression**: Average prediction

## Data Structures

```c
#define RANDOMFOREST_MAX_ESTIMATORS 200

typedef struct {
    int n_estimators;
    DecisionTree** trees;         // Bootstrap trees
    double* feature_importance;   // [n_features] importance scores
} RandomForest_Weights_t;
```

## Configuration

```c
typedef struct {
    int n_estimators;             // Number of trees (default: 100)
    int max_depth;                // Max tree depth (default: unlimited)
    size_t min_samples_split;     // Min samples to split (default: 2)
    size_t n_features_subset;      // Features per split (default: sqrt(n))
} RandomForest_Config_t;
```

## Functions

```c
ML_Model_t create_random_forest_model(void);

int RandomForest_fit(const ML_Model_Config_t* config, RandomForest_Weights_t* state,
                      const dataset* ds, size_t* feat_idx, size_t n_features,
                      size_t label_idx, size_t* indices, size_t n_samples);

int RandomForest_predict(const RandomForest_Weights_t* state, const dataset* ds,
                          size_t* feat_idx, size_t n_features,
                          size_t* indices, size_t n_samples, int* predictions);
```

## Feature Importance

After training, feature importance scores are computed:

```c
double* importance = forest->feature_importance;
for (size_t i = 0; i < n_features; i++) {
    printf("Feature %zu: %.4f\n", i, importance[i]);
}
```

## Example

```c
#define RANDOMFOREST_IMPLEMENTATION
#include "randomforest.h"

ML_Model_t model = create_random_forest_model();

RandomForest_Config_t config = {
    .n_estimators = 200,
    .max_depth = 20,
    .min_samples_split = 5,
    .n_features_subset = 0  // 0 = use sqrt(n)
};
model.config.params = &config;
model.config.size = sizeof(config);

model.methods.fit(&model.config, &model.state,
                  train_ds, NULL, n_features, label_idx,
                  train_idx, train_size);

int predictions[100];
model.methods.predict(&model.state, test_ds, NULL, n_features,
                     test_idx, 100, predictions);
```

## Advantages over AdaBoost

- More robust to noise (averaging reduces variance)
- Parallel training (trees are independent)
- Feature importance estimation
- Less prone to overfitting

## Notes

- `n_features_subset = 0` defaults to $\sqrt{n\_features}$
- Higher `n_estimators` generally better but slower
- For imbalanced data, consider class weighting
