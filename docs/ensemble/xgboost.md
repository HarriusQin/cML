# XGBoost

Gradient boosted decision trees with regularization.

## Overview

XGBoost (eXtreme Gradient Boosting) is an optimized gradient boosting implementation with built-in regularization (L1/L2) to prevent overfitting. It sequentially trains decision trees to correct the errors of previous trees.

## Algorithm

1. For each iteration $t$:
   - Compute gradients and hessians of the loss function
   - Find optimal split by maximizing gain:
   $$\text{gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$
   - Create new leaf weights using: $w = -G / (H + \lambda)$
   - Update predictions

## Data Structure

```c
typedef struct XGBoost_TreeNode {
    size_t feature_index;
    double threshold;
    bool is_leaf;
    double weight;
    struct XGBoost_TreeNode *left, *right;
} XGBoost_TreeNode;

typedef struct XGBoost_Tree {
    XGBoost_TreeNode* root;
    size_t depth;
    size_t n_leaves;
} XGBoost_Tree;

typedef struct XGBoost_State {
    XGBoost_Tree** trees;
    double tree_weights;
    size_t n_trees;
    double learning_rate;
    double reg_lambda;        // L2 regularization
    double reg_alpha;         // L1 regularization
    double min_split_gain;
    double min_child_weight;
    size_t max_depth;
    double subsample;        // Row sampling
    double colsample_bytree;  // Column sampling
} XGBoost_State;
```

## Configuration

```c
typedef struct {
    size_t n_estimators;      // Number of trees
    double learning_rate;     // Shrinkage (default: 0.1)
    size_t max_depth;         // Max tree depth
    double reg_lambda;        // L2 regularization
    double reg_alpha;         // L1 regularization
    double min_child_weight;  // Minimum hessian
    double subsample;         // Subsample ratio
    double colsample_bytree;  // Feature sampling
} XGBoost_Config;
```

## Functions

```c
int XGBoost_fit(const ML_Model_Config_t* config, XGBoost_State* state,
                const dataset* ds, size_t* feat_idx, size_t n_features,
                size_t label_idx, size_t* indices, size_t n_samples);

int XGBoost_predict(const XGBoost_State* state, const dataset* ds,
                    size_t* feat_idx, size_t n_features,
                    size_t* indices, size_t n_samples, int* predictions);

int XGBoost_predict_proba(const XGBoost_State* state, const dataset* ds,
                          size_t* feat_idx, size_t n_features,
                          size_t* indices, size_t n_samples,
                          size_t n_classes, double* probabilities);

void XGBoost_free(XGBoost_State* state);
```

## Loss Functions

For binary classification, XGBoost uses logistic loss:

```c
// Gradient of logistic loss
double xgb_logistic_grad(double pred, int label) {
    double p = 1.0 / (1.0 + exp(-pred));
    return p - label;
}

// Hessian of logistic loss
double xgb_logistic_hess(double pred, int label) {
    double p = 1.0 / (1.0 + exp(-pred));
    return p * (1.0 - p);
}
```

## Regularization

XGBoost adds regularization terms to the objective:

$$\mathcal{L}^{(t)} = \sum_{i=1}^n \left[ g_i w_{q(x_i)} + \frac{1}{2} h_i w_{q(x_i)}^2 \right] + \Omega(w)$$

Where:
$$\Omega(w) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2 + \alpha \sum_{j=1}^T |w_j|$$

- $T$ = number of leaves
- $\lambda$ = L2 regularization
- $\alpha$ = L1 regularization

## Example

```c
#define XGBOOST_IMPLEMENTATION
#include "xgboost.h"

// Configure model
XGBoost_Config config = {
    .n_estimators = 100,
    .learning_rate = 0.1,
    .max_depth = 6,
    .reg_lambda = 1.0,
    .reg_alpha = 0.0,
    .min_child_weight = 1,
    .subsample = 1.0,
    .colsample_bytree = 1.0
};

ML_Model_t model = create_xgboost_model();
model.config.params = &config;

// Train
model.methods.fit(&model.config, &model.state,
                  train_ds, NULL, n_features, label_idx,
                  train_idx, train_size);

// Predict
int predictions[100];
model.methods.predict(&model.state, test_ds, NULL, n_features,
                     test_idx, 100, predictions);
```

## Notes

- XGBoost uses pre-sorting based split finding (exact algorithm)
- Gradient clipping helps stability for logistic regression
- subsample and colsample_bytree add stochasticity for regularization
- Leaf weight formula: $w_j = -G_j / (H_j + \lambda)$
