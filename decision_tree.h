#ifndef __C_DECISION_TREE_H__
#define __C_DECISION_TREE_H__

#include <stddef.h>
#include <stdlib.h>
#include <math.h>

#include "machine_learning.h"

/* ============================================================================
 * Decision Tree
 * ============================================================================ */

typedef struct DT_Node {
    int is_leaf;
    int class_label;
    size_t feature_index;
    double threshold;
    struct DT_Node* left;
    struct DT_Node* right;
} DT_Node;

typedef struct {
    DT_Node* root;
    size_t n_classes;
} DT_State;

static double dt_entropy(int* labels, size_t n) {
    if (n == 0) return 0.0;
    size_t* counts = (size_t*)calloc(256, sizeof(size_t));
    for (size_t i = 0; i < n; i++) counts[labels[i]]++;
    double ent = 0.0;
    for (size_t i = 0; i < 256 && counts[i] > 0; i++) {
        double p = (double)counts[i] / n;
        if (p > 0) ent -= p * log(p);
    }
    free(counts);
    return ent;
}

static double dt_info_gain(const dataset* ds, size_t target_index,
                           size_t feature_idx, double threshold,
                           const size_t* indices, size_t n) {
    int* parent_labels = (int*)malloc(sizeof(int) * n);
    for (size_t i = 0; i < n; i++) parent_labels[i] = (int)ds->labels[target_index].labels[indices[i]];
    double parent_entropy = dt_entropy(parent_labels, n);
    free(parent_labels);

    size_t n_left = 0, n_right = 0;
    int* left_labels = (int*)malloc(sizeof(int) * n);
    int* right_labels = (int*)malloc(sizeof(int) * n);

    for (size_t i = 0; i < n; i++) {
        double val = ds->features[feature_idx].data[indices[i]];
        if (val <= threshold) left_labels[n_left++] = (int)ds->labels[target_index].labels[indices[i]];
        else right_labels[n_right++] = (int)ds->labels[target_index].labels[indices[i]];
    }

    double child_entropy = (n_left * dt_entropy(left_labels, n_left) +
                           n_right * dt_entropy(right_labels, n_right)) / n;
    free(left_labels); free(right_labels);
    return parent_entropy - child_entropy;
}

static DT_Node* dt_build_tree(const dataset* ds, size_t target_index,
                              const size_t* indices, size_t n,
                              size_t n_features, const size_t* feature_indices,
                              size_t max_depth, size_t min_samples_split) {
    DT_Node* node = (DT_Node*)malloc(sizeof(DT_Node));
    node->is_leaf = 0; node->left = NULL; node->right = NULL;

    int first_label = (int)ds->labels[target_index].labels[indices[0]];
    int all_same = 1;
    for (size_t i = 1; i < n; i++)
        if ((int)ds->labels[target_index].labels[indices[i]] != first_label) { all_same = 0; break; }

    if (all_same || n < min_samples_split || max_depth == 0) {
        node->is_leaf = 1; node->class_label = first_label;
        return node;
    }

    double best_gain = -1.0; size_t best_feature = 0; double best_threshold = 0.0;
    for (size_t f = 0; f < n_features; f++) {
        size_t feat_idx = feature_indices[f];
        double min_val = ds->features[feat_idx].data[indices[0]];
        double max_val = min_val;
        for (size_t i = 1; i < n; i++) {
            double val = ds->features[feat_idx].data[indices[i]];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
        for (int t = 1; t <= 5; t++) {
            double threshold = min_val + (max_val - min_val) * t / 6.0;
            double gain = dt_info_gain(ds, target_index, feat_idx, threshold, indices, n);
            if (gain > best_gain) { best_gain = gain; best_feature = feat_idx; best_threshold = threshold; }
        }
    }

    if (best_gain <= 0.0) { node->is_leaf = 1; node->class_label = first_label; return node; }

    node->feature_index = best_feature;
    node->threshold = best_threshold;

    size_t* left_idx = (size_t*)malloc(sizeof(size_t) * n);
    size_t* right_idx = (size_t*)malloc(sizeof(size_t) * n);
    size_t n_left = 0, n_right = 0;
    for (size_t i = 0; i < n; i++) {
        double val = ds->features[best_feature].data[indices[i]];
        if (val <= best_threshold) left_idx[n_left++] = indices[i];
        else right_idx[n_right++] = indices[i];
    }

    node->left = dt_build_tree(ds, target_index, left_idx, n_left, n_features, feature_indices, max_depth - 1, min_samples_split);
    node->right = dt_build_tree(ds, target_index, right_idx, n_right, n_features, feature_indices, max_depth - 1, min_samples_split);
    free(left_idx); free(right_idx);
    return node;
}

static int dt_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
                  const dataset* ds, const size_t* feature_indices, size_t n_features,
                  size_t target_index, const size_t* sample_indices, size_t n_samples) {
    (void)config; (void)n_features;
    DT_State* dt = (DT_State*)malloc(sizeof(DT_State));
    dt->n_classes = ds->labels[target_index].classes;
    dt->root = dt_build_tree(ds, target_index, sample_indices, n_samples, n_features, feature_indices, 10, 2);
    state->weights = dt; state->size = sizeof(DT_State);
    return 0;
}

static int dt_predict_single(const DT_Node* node, const dataset* ds, const size_t* feature_indices, size_t sample_idx) {
    if (node->is_leaf) return node->class_label;
    double val = ds->features[node->feature_index].data[sample_idx];
    return (val <= node->threshold) ? dt_predict_single(node->left, ds, feature_indices, sample_idx)
                                     : dt_predict_single(node->right, ds, feature_indices, sample_idx);
}

static int dt_predict(const ML_Weights_t* state, const dataset* ds,
                      const size_t* feature_indices, size_t n_features,
                      const size_t* sample_indices, size_t n_samples, void* output) {
    (void)n_features;
    DT_State* dt = (DT_State*)state->weights;
    int* predictions = (int*)output;
    for (size_t s = 0; s < n_samples; s++)
        predictions[s] = dt_predict_single(dt->root, ds, feature_indices, sample_indices[s]);
    return 0;
}

static void dt_free_node(DT_Node* node) {
    if (!node) return;
    dt_free_node(node->left); dt_free_node(node->right); free(node);
}

static void dt_free(ML_Weights_t* state) {
    if (state->weights) { DT_State* dt = (DT_State*)state->weights; dt_free_node(dt->root); free(dt); state->weights = NULL; }
}

static ML_Model_t create_dt_model(void) {
    return (ML_Model_t){
        .type = ML_CLASSIFICATION,
        .config = {NULL, 0}, .state = {NULL, 0},
        .methods = { .fit = dt_fit, .predict = dt_predict,
                     .predict_proba = NULL, .get_coefficients = NULL,
                     .serialize = NULL, .deserialize = NULL, .free_state = dt_free }
    };
}

#endif /* __C_DECISION_TREE_H__ */