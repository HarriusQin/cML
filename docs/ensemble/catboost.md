# CatBoost

Gradient boosting with ordered target statistics for reduced overfitting.

## Overview

CatBoost (Categorical Boosting) uses a novel target-based encoding for categorical features and an ordered boosting algorithm to reduce overfitting and prediction shift.

## Key Features

- **Ordered Target Statistics**: Reduces overfitting from target leakage
- **Automatic Categorical Handling**: Processes categorical features without explicit encoding
- **Ordered Boosting**: Trains trees on different subsets to prevent prediction shift
- **Symmetric Trees**: Uses oblivious decision trees for faster inference

## Algorithm

### Ordered Target Statistics

For categorical feature $k$, compute target statistics:

$$\text{enc}_k(x) = \frac{\sum_{i=1}^{n} 1_{x_k = x} \cdot \text{target}_i + a \cdot P}{\sum_{i=1}^{n} 1_{x_k = x} + a}$$

Where:
- $a$ is the prior (smoothed average)
- $P$ is the global prior (mean target)

### Ordered Boosting

1. Generate random permutation $\sigma$ of training samples
2. For each sample $i$, compute residual using only samples preceding $i$ in permutation
3. Train tree on these residuals
4. Make predictions using trees trained on samples that do not include the current sample

## Data Structure

```c
typedef struct CatBoost_TreeNode {
    size_t feature_index;
    double threshold;
    bool is_leaf;
    double weight;
    struct CatBoost_TreeNode *left, *right;
} CatBoost_TreeNode;

typedef struct CatBoost_Tree {
    CatBoost_TreeNode* root;
    size_t depth;
    size_t n_leaves;
} CatBoost_Tree;

typedef struct CatBoost_State {
    CatBoost_Tree** trees;
    double* tree_weights;
    size_t n_trees;
    double learning_rate;
    double l2_leaf_reg;
    double random_strength;
    size_t max_depth;
    size_t border_count;
    double min_data_in_leaf;
} CatBoost_State;
```

## Configuration

```c
typedef struct {
    size_t iterations;           // Number of boosting iterations
    double learning_rate;        // Step size
    size_t depth;                // Max tree depth
    double l2_leaf_reg;          // L2 regularization
    double random_strength;      // Randomness for splits
    size_t border_count;         // Number of splits for numerical features
    size_t min_data_in_leaf;     // Minimum samples per leaf
    double bagging_temperature;   // Bayesian bootstrap temperature
} CatBoost_Config;
```

## Functions

```c
int CatBoost_fit(const ML_Model_Config_t* config, CatBoost_State* state,
                 const dataset* ds, size_t* feat_idx, size_t n_features,
                 size_t target_idx, size_t* indices, size_t n_samples);

int CatBoost_predict(const CatBoost_State* state, const dataset* ds,
                     size_t* feat_idx, size_t n_features,
                     size_t* indices, size_t n_samples, int* predictions);

int CatBoost_predict_proba(const CatBoost_State* state, const dataset* ds,
                           size_t* feat_idx, size_t n_features,
                           size_t* indices, size_t n_samples,
                           size_t n_classes, double* probabilities);

void CatBoost_free(CatBoost_State* state);
```

## Categorical Feature Handling

CatBoost handles categorical features differently from other gradient boosting libraries:

```c
// For each categorical feature, compute ordered target statistics
static double compute_category_statistics(
    const dataset* ds,
    size_t feature_idx,
    size_t category_value,
    const double* targets,
    const size_t* indices,
    size_t n_samples,
    double prior) {
    double sum = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < n_samples; i++) {
        if (ds->features[feature_idx].data[indices[i]] == category_value) {
            sum += targets[indices[i]];
            count++;
        }
    }
    double a = 1.0;  // smoothing factor
    return (sum + a * prior) / (count + a);
}
```

## Example

```c
#define CATBOOST_IMPLEMENTATION
#include "catboost.h"

CatBoost_Config config = {
    .iterations = 500,
    .learning_rate = 0.05,
    .depth = 6,
    .l2_leaf_reg = 3.0,
    .random_strength = 1.0,
    .border_count = 254,
    .min_data_in_leaf = 1
};

ML_Model_t model = create_catboost_model();
model.config.params = &config;

model.methods.fit(&model.config, &model.state,
                 train_ds, categorical_idx, n_features, label_idx,
                 train_idx, train_size);

// Predict probabilities
double probs[100 * n_classes];
CatBoost_predict_proba(&model.state, test_ds, NULL, n_features,
                      test_idx, 100, n_classes, probs);
```

## Differences from XGBoost

| Feature | XGBoost | CatBoost |
|---------|---------|----------|
| Categorical handling | Requires encoding | Native ordered statistics |
| Tree structure | Asymmetric | Symmetric (oblivious) |
| Overfitting prevention | Row/column sampling | Ordered boosting |
| Prediction shift | May occur | Mitigated by ordering |
| Missing values | Needs handling | Native handling |

## Notes

- CatBoost uses ordered boosting to prevent prediction shift
- Symmetric (oblivious) trees are faster for inference
- Target statistics are computed on-the-fly during training
- The random permutation is different for each iteration
