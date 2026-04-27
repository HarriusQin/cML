#ifndef __C_RANDOMFOREST_H__
#define __C_RANDOMFOREST_H__

#define MAX_CLASSES 256
#define RANDOMFOREST_MAX_ESTIMATORS 200

#include <stddef.h>
#include <stdlib.h>

#include "machine_learning.h"
#include "utilities.h"

typedef struct TreeNode {
    bool is_leaf;
    int class_label;
    size_t feature_index;
    double threshold;
    struct TreeNode* left;
    struct TreeNode* right;
} TreeNode;

typedef struct DecisionTree_State {
    TreeNode* root;
    size_t n_classes;
} DecisionTree_State;

typedef struct RandomForest_State {
    TreeNode** trees;
    size_t n_trees;
    size_t n_classes;
    size_t max_features;
} RandomForest_State;

static int majority_vote(const dataset* ds, size_t target_index,
                         const size_t* indices, size_t n);

static TreeNode* build_tree(const dataset* ds, size_t target_index,
                            const size_t* indices, size_t n, size_t n_features,
                            const size_t* feature_indices, size_t max_depth,
                            size_t min_samples);

static int tree_predict_single(const TreeNode* node, const dataset* ds,
                               size_t sample_idx);

static size_t* bootstrap(size_t n, unsigned int seed);

static void shuffle_array(size_t* array, size_t n);

static int RandomForest_fit(const ML_Model_Config_t* config,
                            ML_Weights_t* state, const dataset* ds,
                            const size_t* feature_indices, size_t n_features,
                            size_t target_index, const size_t* sample_indices,
                            size_t n_samples);

static int RandomForest_predict(const ML_Weights_t* state, const dataset* ds,
                                const size_t* feature_indices,
                                size_t n_features, const size_t* sample_indices,
                                size_t n_samples, void* output);

static void free_tree(TreeNode* node);

static void RandomForest_free(ML_Weights_t* state);

static ML_Model_t create_randomforest_model(void);

#ifdef RANDOMFOREST_IMPLEMENTATION

static int majority_vote(const dataset* ds, size_t target_index,
                         const size_t* indices, size_t n) {
    size_t* counts = (size_t*)calloc(MAX_CLASSES, sizeof(size_t));
    for (int i = 0; i < n; ++i)
        ++counts[ds->labels[target_index].labels[indices[i]]];

    int majority = 0;
    for (int i = 1; i < MAX_CLASSES; ++i)
        if (counts[i] > counts[majority])
            majority = i;

    free(counts);
    return majority;
}

static TreeNode* build_tree(const dataset* ds, size_t target_index,
                            const size_t* indices, size_t n, size_t n_features,
                            const size_t* feature_indices, size_t max_depth,
                            size_t min_samples) {
    TreeNode* node = (TreeNode*)malloc(sizeof(TreeNode));
    node->is_leaf = false;
    node->left = NULL;
    node->right = NULL;

    if (n < min_samples || max_depth == 0) {
        node->is_leaf = 1;
        node->class_label = majority_vote(ds, target_index, indices, n);
        return node;
    }

    int first_label = (int)ds->labels[target_index].labels[indices[0]];
    bool identical = true;
    for (size_t i = 0; i < n; ++i) {
        if ((int)ds->labels[target_index].labels[indices[i]] != first_label) {
            identical = false;
            break;
        }
    }

    if (identical) {
        node->is_leaf = true;
        node->class_label = first_label;
        return node;
    }

    double best_gain = -0.1;
    size_t best_feature = 0;
    double best_threshold = 0.0;

    for (size_t f = 0; f < n_features; ++f) {
        size_t feat_idx = feature_indices[f];

        FeatureSample* sorted =
            (FeatureSample*)malloc(sizeof(FeatureSample) * n);
        for (size_t i = 0; i < n; ++i) {
            sorted[i].idx = indices[i];
            sorted[i].val = ds->features[feat_idx].data[indices[i]];
        }
        qsort(sorted, n, sizeof(FeatureSample), cmp_FeatureSample);

        size_t* left_counts = (size_t*)calloc(MAX_CLASSES, sizeof(size_t));
        size_t* right_counts = (size_t*)calloc(MAX_CLASSES, sizeof(size_t));

        for (size_t i = 0; i < n; ++i)
            right_counts[ds->labels[target_index].labels[sorted[i].idx]]++;

        double parent_gini = 1.0;
        for (int c = 0; c < MAX_CLASSES; ++c)
            if (right_counts[c] > 0) {
                double p = (double)right_counts[c] / n;
                parent_gini -= p * p;
            }

        for (size_t spilt = 0; spilt < n - 1; ++spilt) {
            double v_curr = sorted[spilt].val;
            double v_next = sorted[spilt + 1].val;

            size_t label = ds->labels[target_index].labels[sorted[spilt].idx];
            --right_counts[label];
            ++left_counts[label];

            if (v_curr == v_next)
                continue;

            double threshold = (v_curr + v_next) / 2.0;
            double n_left = (double)(spilt + 1);
            double n_right = (double)(n - spilt - 1);
            double l_gini = 1.0, r_gini = 1.0;
            for (int c = 0; c < MAX_CLASSES; ++c) {
                if (left_counts[c] > 0) {
                    double p = (double)left_counts[c] / n_left;
                    l_gini -= p * p;
                }
                if (right_counts[c] > 0) {
                    double p = (double)right_counts[c] / n_right;
                    r_gini -= p * p;
                }
            }

            double child_gini = (n_left * l_gini + n_right * r_gini) / n;
            double gini_gain = parent_gini - child_gini;
            if (gini_gain > best_gain) {
                best_gain = gini_gain;
                best_feature = feat_idx;
                best_threshold = threshold;
            }
        }
        free(left_counts);
        free(right_counts);
        free(sorted);
    }
    if (best_gain <= 0.0) {
        node->is_leaf = 1;
        node->class_label = majority_vote(ds, target_index, indices, n);
        return node;
    }

    node->feature_index = best_feature;
    node->threshold = best_threshold;

    size_t* left_idx = (size_t*)malloc(sizeof(size_t) * n);
    size_t* right_idx = (size_t*)malloc(sizeof(size_t) * n);
    size_t n_left = 0, n_right = 0;

    for (size_t i = 0; i < n; ++i) {
        double val = ds->features[best_feature].data[indices[i]];
        if (val <= best_threshold)
            left_idx[n_left++] = indices[i];
        else
            right_idx[n_right++] = indices[i];
    }

    node->left = build_tree(ds, target_index, left_idx, n_left, n_features,
                            feature_indices, max_depth - 1, min_samples);
    node->right = build_tree(ds, target_index, right_idx, n_right, n_features,
                             feature_indices, max_depth - 1, min_samples);

    free(left_idx);
    free(right_idx);

    return node;
}

static int tree_predict_single(const TreeNode* node, const dataset* ds,
                               size_t sample_idx) {
    if (node->is_leaf)
        return node->class_label;

    double val = ds->features[node->feature_index].data[sample_idx];
    if (val <= node->threshold) {
        return tree_predict_single(node->left, ds, sample_idx);
    } else {
        return tree_predict_single(node->right, ds, sample_idx);
    }
}

static size_t* bootstrap(size_t n, unsigned int seed) {
    size_t* indices = (size_t*)malloc(sizeof(size_t) * n);
    srand(seed);
    for (size_t i = 0; i < n; ++i)
        indices[i] = (size_t)(rand() % n);
    return indices;
}

static void shuffle_array(size_t* array, size_t n) {
    for (size_t i = n - 1; i > 0; --i) {
        size_t j = rand() % (i + 1);
        size_t tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
    }
}

static int RandomForest_fit(const ML_Model_Config_t* config,
                            ML_Weights_t* state, const dataset* ds,
                            const size_t* feature_indices, size_t n_features,
                            size_t target_index, const size_t* sample_indices,
                            size_t n_samples) {
    size_t max_trees = RANDOMFOREST_MAX_ESTIMATORS;
    size_t max_features = (size_t)sqrt(n_features);
    if (max_features < 2)
        max_features = 2;

    RandomForest_State* rf =
        (RandomForest_State*)malloc(sizeof(RandomForest_State));
    rf->trees = (TreeNode**)malloc(sizeof(TreeNode*) * max_trees);
    rf->n_trees = max_trees;
    rf->n_classes = ds->labels[target_index].classes;
    rf->max_features = max_features;

    size_t* selected_features = (size_t*)malloc(sizeof(size_t) * n_features);
    for (size_t i = 0; i < n_features; ++i) {
        selected_features[i] = feature_indices[i];
    }

    /* OOB tracking: accumulate votes and count how many trees voted per sample
     */
    int* oob_votes = (int*)calloc(ds->rows * rf->n_classes, sizeof(int));
    int* oob_count = (int*)calloc(ds->rows, sizeof(int));

    size_t patience = 30;
    size_t min_trees = 20;
    double oob_error_threshold =
        0.10; /* 90% accuracy threshold for early stop */
    size_t best_error_tree = 0;
    double best_oob_error = 1.0;

    for (size_t t = 0; t < max_trees; ++t) {
        size_t* bag = bootstrap(n_samples, (unsigned)t * 12345 + t);

        /* Compute OOB indices (samples NOT in bag) */
        int* in_bag = (int*)calloc(n_samples, sizeof(int));
        for (size_t i = 0; i < n_samples; ++i)
            in_bag[bag[i]] = 1;
        size_t* oob_idx = (size_t*)malloc(sizeof(size_t) * n_samples);
        size_t n_oob = 0;
        for (size_t i = 0; i < n_samples; ++i) {
            if (!in_bag[i])
                oob_idx[n_oob++] = sample_indices[i];
        }
        free(in_bag);

        /* Randomly select max_features from all features */
        shuffle_array(selected_features, n_features);

        rf->trees[t] = build_tree(ds, target_index, bag, n_samples,
                                  max_features, selected_features, 20, 2);

        free(bag);

        /* Evaluate on OOB samples and accumulate votes */
        for (size_t i = 0; i < n_oob; ++i) {
            size_t idx = oob_idx[i];
            int pred = tree_predict_single(rf->trees[t], ds, idx);
            oob_votes[idx * rf->n_classes + pred]++;
            oob_count[idx]++;
        }
        free(oob_idx);

        /* Early stopping check after min_trees */
        if (t >= min_trees - 1) {
            /* Compute OOB error from accumulated votes */
            size_t correct = 0, total = 0;
            for (size_t i = 0; i < n_samples; ++i) {
                size_t idx = sample_indices[i];
                if (oob_count[idx] == 0)
                    continue;
                /* Find majority vote prediction */
                int best_class = 0;
                int best_votes = oob_votes[idx * rf->n_classes];
                for (size_t c = 1; c < rf->n_classes; ++c) {
                    if (oob_votes[idx * rf->n_classes + c] > best_votes) {
                        best_votes = oob_votes[idx * rf->n_classes + c];
                        best_class = (int)c;
                    }
                }
                int true_label = ds->labels[target_index].labels[idx];
                if (best_class == true_label)
                    correct++;
                total++;
            }
            double oob_error =
                (total > 0) ? 1.0 - (double)correct / total : 1.0;

            /* Track best error and stopping condition */
            int improved = 0;
            if (oob_error < best_oob_error - 0.001) {
                best_oob_error = oob_error;
                best_error_tree = t;
                improved = 1;
            }

            /* Stop if OOB error below threshold and no improvement for patience
             * trees */
            if (oob_error < oob_error_threshold && !improved &&
                t >= best_error_tree + patience - 1) {
                rf->n_trees = t + 1;
                free(selected_features);
                free(oob_votes);
                free(oob_count);
                state->weights = rf;
                state->size = sizeof(RandomForest_State);
                return 0;
            }
        }
    }

    free(selected_features);
    free(oob_votes);
    free(oob_count);
    state->weights = rf;
    state->size = sizeof(RandomForest_State);
    return 0;
}

static int RandomForest_predict(const ML_Weights_t* state, const dataset* ds,
                                const size_t* feature_indices,
                                size_t n_features, const size_t* sample_indices,
                                size_t n_samples, void* output) {
    RandomForest_State* rf = (RandomForest_State*)state->weights;
    int* predictions = (int*)output;

    for (size_t s = 0; s < n_samples; ++s) {
        size_t idx = sample_indices[s];
        size_t* votes = (size_t*)calloc(rf->n_classes, sizeof(size_t));

        for (size_t t = 0; t < rf->n_trees; ++t) {
            int pred = tree_predict_single(rf->trees[t], ds, idx);
            votes[pred]++;
        }

        size_t best_vote = 0;
        for (size_t c = 1; c < rf->n_classes; ++c)
            if (votes[c] > votes[best_vote])
                best_vote = c;

        predictions[s] = (int)best_vote;
        free(votes);
    }
    return 0;
}

static void free_tree(TreeNode* node) {
    if (!node) {
        return;
    }
    free_tree(node->left);
    free_tree(node->right);
    free(node);
}

static void RandomForest_free(ML_Weights_t* state) {
    if (state->weights) {
        RandomForest_State* rf = (RandomForest_State*)state->weights;
        for (size_t t = 0; t < rf->n_trees; ++t) {
            free_tree(rf->trees[t]);
        }
        free(rf->trees);
        free(rf);
        state->weights = NULL;
    }
}

static ML_Model_t create_randomforest_model(void) {
    return (ML_Model_t){.type = ML_CLASSIFICATION,
                        .config = {NULL, 0},
                        .state = {NULL, 0},
                        .methods = {.fit = RandomForest_fit,
                                    .predict = RandomForest_predict,
                                    .predict_proba = NULL,
                                    .get_coefficients = NULL,
                                    .serialize = NULL,
                                    .deserialize = NULL,
                                    .free_state = RandomForest_free}};
}

#endif /* RANDOMFOREST_IMPLEMENTATION */

#endif /* __C_RANDOMFOREST_H__ */
