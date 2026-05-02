#ifndef __C_XGBOOST_H__
#define __C_XGBOOST_H__

#include "machine_learning.h"

#include <math.h>

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
    double reg_lambda;
    double reg_alpha;
    double min_split_gain;
    double min_child_weight;
    size_t max_depth;
    double subsample;
    double colsample_bytree;
} XGBoost_State;

static int XGBoost_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
                       const dataset* ds, const size_t* feature_indices,
                       size_t n_features, size_t target_index,
                       const size_t* sample_indices, size_t n_samples);

static int XGBoost_predict(const ML_Weights_t* state, const dataset* ds,
                           const size_t* feature_indices, size_t n_features,
                           const size_t* sample_indices, size_t n_samples,
                           void* output);

static int XGBoost_predict_proba(const ML_Weights_t* state, const dataset* ds,
                                 const size_t* feature_indices,
                                 size_t n_features,
                                 const size_t* sample_indices, size_t n_samples,
                                 size_t n_classes, void* output);

static void XGBoost_free(ML_Weights_t* state);

#ifdef XGBOOST_IMPLEMENTATION

inline static double xgb_logistic_grad(double pred, int label) {
    double p = 1.0 / (1.0 + exp(-pred));
    return p - label;
}

inline static double xgb_logistic_hess(double pred, int label) {
    double p = 1.0 / (1.0 + exp(-pred));
    return p * (1.0 - p);
}

inline static void xgb_compute_gradients_classifier(
    double* g, double* h, const double* predictions, const int* labels,
    size_t n_samples, double (*loss_grad)(double, int),
    double (*loss_hess)(double, int)) {
    for (size_t i = 0; i < n_samples; i++) {
        g[i] = loss_grad(predictions[i], labels[i]);
        h[i] = loss_hess(predictions[i], labels[i]);
    }
}

inline static double xgb_split_gain_classifier(double G_L, double H_L,
                                               double G_R, double H_R,
                                               double lambda, double gamma) {
    return 0.5 * (G_L * G_L / (H_L + lambda) + G_R * G_R / (H_R + lambda) -
                  (G_L + G_R) * (G_L + G_R) / (H_L + H_R + lambda)) -
           gamma;
}

static double xgb_leaf_weight(const double* g, const double* h,
                              const size_t* indices, size_t n_samples,
                              double lambda) {
    double G = 0.0, H = 0.0;
    for (size_t i = 0; i < n_samples; i++) {
        G += g[indices[i]];
        H += h[indices[i]];
    }

    if (H + lambda == 0.0) {
        return 0.0
    }
    return -G / (H + lambda);
}

static XGBoost_TreeNode* xgb_create_leaf_node(const double* g, const double* h,
                                              const size_t* indices,
                                              size_t n_samples,
                                              const XGBoost_State* params) {
    XGBoost_TreeNode* node =
        (XGBoost_TreeNode*)malloc(sizeof(XGBoost_TreeNode));
    if (!node)
        return NULL;

    node->is_leaf = true;
    node->feature_index = (size_t)-1;
    node->threshold = 0.0;
    node->left = NULL;
    node->right = NULL;

    node->weight =
        xgb_leaf_weight(g, h, indices, n_samples, params->reg_lambda);

    return node;
}

static XGBoost_TreeNode*
xgb_build_tree_classifier(double* g, double* h, const dataset* ds,
                          const size_t* indices, size_t n_samples,
                          const XGBoost_State* params, size_t depth) {
    if (depth >= params->max_depth) {
        return xgb_create_leaf_node(g, h, indices, n_samples, params);
    }
}

// static double mse_grad(double pred, double true)

static int XGBoost_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
                       const dataset* ds, const size_t* feature_indices,
                       size_t n_features, size_t target_index,
                       const size_t* sample_indices, size_t n_samples) {}

static int XGBoost_predict(const ML_Weights_t* state, const dataset* ds,
                           const size_t* feature_indices, size_t n_features,
                           const size_t* sample_indices, size_t n_samples,
                           void* output) {}

static int XGBoost_predict_proba(const ML_Weights_t* state, const dataset* ds,
                                 const size_t* feature_indices,
                                 size_t n_features,
                                 const size_t* sample_indices, size_t n_samples,
                                 size_t n_classes, void* output) {}

static void XGBoost_free(ML_Weights_t* state) {}

#endif // XGBOOST_IMPLEMENTATION

#endif // !__C_XGBOOST_H__
