/**
 * @file ml_samples.h
 * @brief Machine Learning Models Library - Sample Implementation
 *
 * =============================================================================
 * 机器学习模型示例库 - 纯C语言实现
 * =============================================================================
 *
 * 本库提供以下机器学习模型（示例实现）：
 *
 * 分类模型:
 *   - 高斯朴素贝叶斯 (Gaussian Naive Bayes)
 *   - 决策树 (Decision Tree)
 *   - 随机森林 (Random Forest)
 *   - AdaBoost
 *   - K近邻 (K-Nearest Neighbors)
 *
 * 回归模型:
 *   - 随机森林回归 (Random Forest Regression)
 *   - K近邻回归 (KNN Regression)
 *
 * =============================================================================
 *
 * This library provides the following machine learning models:
 *
 * Classification:
 *   - Gaussian Naive Bayes (GNB)
 *   - Decision Tree (DT)
 *   - Random Forest (RF)
 *   - AdaBoost
 *   - K-Nearest Neighbors (KNN)
 *
 * Regression:
 *   - Random Forest Regression
 *   - KNN Regression
 *
 * @license MIT
 */

#ifndef ML_SAMPLES_H
#define ML_SAMPLES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Include Dataset types from dataset.h */
#ifndef DATASET_H_INCLUDED
#include "dataset.h"
#endif

/* ============================================================================
 * DATA STRUCTURES 数据结构
 * ============================================================================ */

/**
 * @brief Supported model types
 *
 * 支持的模型类型枚举
 */
typedef enum {
    MODEL_GNB,                  /**< 高斯朴素贝叶斯 */
    MODEL_DECISION_TREE,        /**< 决策树 */
    MODEL_RANDOM_FOREST,        /**< 随机森林 */
    MODEL_ADABOOST,            /**< AdaBoost */
    MODEL_KNN                  /**< K近邻 */
} model_type_t;

/**
 * @brief Model configuration
 *
 * 模型训练配置参数
 */
typedef struct {
    double learning_rate;       /**< 学习率/子采样比例 */
    size_t max_iter;           /**< 最大迭代次数 */
    double tol;                 /**< 收敛容差 */
    bool verbose;              /**< 详细输出 */
    size_t n_estimators;       /**< 树的数量 (RF/AdaBoost) */
    size_t max_depth;          /**< 树的最大深度 (0=无限制) */
    size_t min_samples_split;   /**< 分裂所需最小样本数 */
    size_t min_samples_leaf;    /**< 叶节点最小样本数 */
    size_t k_value;            /**< KNN的K值 */
} model_config;

/** 默认配置 */
static const model_config default_config = {
    .learning_rate = 1.0,
    .max_iter = 1000,
    .tol = 1e-6,
    .verbose = false,
    .n_estimators = 100,
    .max_depth = 10,
    .min_samples_split = 2,
    .min_samples_leaf = 1,
    .k_value = 5
};

/* ============================================================================
 * DECISION TREE NODE 决策树节点
 * ============================================================================ */

/**
 * @brief Decision Tree Node
 *
 * 决策树节点结构
 */
typedef struct dt_node {
    bool is_leaf;              /**< 是否为叶节点 */
    size_t feature_index;      /**< 分裂特征索引 */
    double threshold;          /**< 分裂阈值 */
    struct dt_node* left;      /**< 左子节点 (<=threshold) */
    struct dt_node* right;     /**< 右子节点 (>threshold) */
    int class_label;           /**< 类别标签 (分类) */
    double value;              /**< 节点值 (回归) */
    size_t samples;            /**< 样本数量 */
} dt_node;

/* ============================================================================
 * GAUSSIAN NAIVE BAYES 高斯朴素贝叶斯
 * ============================================================================ */

/**
 * @brief Gaussian Naive Bayes Model Data
 *
 * 高斯朴素贝叶斯模型数据
 */
typedef struct {
    double** mean;              /**< 均值 [n_classes][n_features] */
    double** var;               /**< 方差 [n_classes][n_features] */
    double* class_prior;        /**< 类先验概率 [n_classes] */
    size_t n_classes;           /**< 类别数量 */
    size_t n_features;          /**< 特征数量 */
    double epsilon;             /**< 正则化项 */
} gnb_data;

/* ============================================================================
 * RANDOM FOREST 随机森林
 * ============================================================================ */

/**
 * @brief Random Forest Model Data
 *
 * 随机森林模型数据
 */
typedef struct {
    dt_node** trees;            /**< 决策树数组 */
    size_t n_estimators;       /**< 树的数量 */
    size_t max_depth;          /**< 最大深度 */
    size_t max_features;        /**< 每树特征数 */
    double subsample_ratio;     /**< 子采样比例 */
    size_t n_classes;           /**< 类别数 */
    bool is_regression;         /**< 是否为回归 */
} rf_data;

/* ============================================================================
 * ADABOOST AdaBoost
 * ============================================================================ */

/**
 * @brief AdaBoost Weak Learner (Decision Stump)
 *
 * AdaBoost弱分类器 - 决策树桩
 */
typedef struct {
    size_t feature_index;       /**< 分裂特征 */
    double threshold;           /**< 分裂阈值 */
    int left_class;            /**< 左类标签 */
    int right_class;           /**< 右类标签 */
    double alpha;              /**< 权重系数 */
} adaboost_stump;

/**
 * @brief AdaBoost Model Data
 *
 * AdaBoost模型数据
 */
typedef struct {
    adaboost_stump* stumps;   /**< 弱分类器数组 */
    double* weights;           /**< 样本权重 */
    size_t n_stumps;           /**< 弱分类器数量 */
    size_t n_classes;          /**< 类别数 */
} adaboost_data;

/* ============================================================================
 * KNN K近邻
 * ============================================================================ */

/**
 * @brief KNN Model Data
 *
 * K近邻模型数据
 */
typedef struct {
    double* X;                 /**< 训练数据特征 */
    double* y;                 /**< 训练数据标签/值 */
    size_t n_samples;           /**< 样本数量 */
    size_t n_features;          /**< 特征数量 */
    size_t k;                   /**< K值 */
    bool is_regression;         /**< 是否为回归 */
} knn_data;

/* ============================================================================
 * GENERIC MODEL 通用模型
 * ============================================================================ */

/**
 * @brief Generic Model Handle
 *
 * 通用模型句柄
 */
typedef struct model {
    model_type_t type;          /**< 模型类型 */
    void* data;                /**< 内部数据 */
} model;

/* ============================================================================
 * FUNCTION DECLARATIONS 函数声明
 * ============================================================================ */

/* --- Gaussian Naive Bayes --- */
model* train_gnb(const dataset* ds, size_t feature_idx, size_t label_idx,
                 model_config config);
int predict_gnb(const model* m, const double* x, size_t n);
void free_gnb(model* m);

/* --- Decision Tree --- */
model* train_decision_tree(const dataset* ds, size_t feature_idx, size_t label_idx,
                          model_config config);
double predict_decision_tree(const model* m, const double* x, size_t n);
void free_decision_tree(model* m);

/* --- Random Forest --- */
model* train_random_forest(const dataset* ds, size_t feature_idx, size_t label_idx,
                          model_config config);
double predict_random_forest(const model* m, const double* x, size_t n);
void free_random_forest(model* m);

/* --- AdaBoost --- */
model* train_adaboost(const dataset* ds, size_t feature_idx, size_t label_idx,
                     model_config config);
int predict_adaboost(const model* m, const double* x, size_t n);
void free_adaboost(model* m);

/* --- KNN --- */
model* train_knn(const dataset* ds, size_t feature_idx, size_t label_idx,
                model_config config);
double predict_knn(const model* m, const double* x, size_t n);
void free_knn(model* m);

/* --- Evaluation --- */
double accuracy(const model* m, const dataset* ds, size_t feature_idx, size_t label_idx);
double r2_score(const model* m, const dataset* ds, size_t feature_idx, size_t label_idx);
double mse(const model* m, const dataset* ds, size_t feature_idx, size_t label_idx);
void model_free(model* m);

/* ============================================================================
 * IMPLEMENTATION 实现
 * ============================================================================ */

#ifdef ML_IMPLEMENTATION

/* ============================================================================
 * UTILITY FUNCTIONS 工具函数
 * ============================================================================ */

/* 计算基尼不纯度 */
static double gini_impurity(int* labels, size_t n) {
    if (n == 0) return 0.0;
    size_t max_class = 0;
    for (size_t i = 0; i < n; i++) {
        if ((size_t)labels[i] > max_class) max_class = (size_t)labels[i];
    }

    size_t* counts = (size_t*)calloc(max_class + 1, sizeof(size_t));
    if (!counts) return 0.0;

    for (size_t i = 0; i < n; i++) counts[labels[i]]++;

    double gini = 1.0;
    for (size_t c = 0; c <= max_class; c++) {
        if (counts[c] > 0) {
            double p = (double)counts[c] / n;
            gini -= p * p;
        }
    }
    free(counts);
    return gini;
}

/* 计算方差 */
static double variance(double* values, size_t n) {
    if (n == 0) return 0.0;
    double sum = 0.0, sum_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += values[i];
        sum_sq += values[i] * values[i];
    }
    double mean = sum / n;
    return (sum_sq / n) - (mean * mean);
}

/* ============================================================================
 * GAUSSIAN NAIVE BAYES IMPLEMENTATION
 * ============================================================================ */

static double log_gaussian_pdf(double x, double mean, double var, double epsilon) {
    double variance = var + epsilon;
    double coeff = -0.5 * log(2.0 * M_PI * variance);
    double exponent = -(x - mean) * (x - mean) / (2.0 * variance);
    return coeff + exponent;
}

model* train_gnb(const dataset* ds, size_t feature_idx, size_t label_idx,
                 model_config config) {
    (void)config;

    if (!ds || !ds->features || !ds->labels) return NULL;

    size_t n_samples = ds->rows;
    size_t n_features = ds->num_features;
    size_t n_classes = ds->labels[label_idx].classes;

    model* m = (model*)malloc(sizeof(model));
    if (!m) return NULL;

    gnb_data* data = (gnb_data*)malloc(sizeof(gnb_data));
    if (!data) { free(m); return NULL; }

    m->type = MODEL_GNB;
    m->data = data;

    data->n_classes = n_classes;
    data->n_features = n_features;
    data->epsilon = 1e-9;

    data->mean = (double**)malloc(sizeof(double*) * n_classes);
    data->var = (double**)malloc(sizeof(double*) * n_classes);
    data->class_prior = (double*)malloc(sizeof(double) * n_classes);

    if (!data->mean || !data->var || !data->class_prior) {
        free(data->mean); free(data->var); free(data->class_prior);
        free(data); free(m);
        return NULL;
    }

    size_t* class_counts = (size_t*)calloc(n_classes, sizeof(size_t));

    for (size_t c = 0; c < n_classes; c++) {
        data->mean[c] = (double*)calloc(n_features, sizeof(double));
        data->var[c] = (double*)malloc(sizeof(double) * n_features);
        if (!data->mean[c] || !data->var[c]) {
            for (size_t i = 0; i <= c; i++) {
                free(data->mean[i]); free(data->var[i]);
            }
            free(class_counts);
            free(data->mean); free(data->var); free(data->class_prior);
            free(data); free(m);
            return NULL;
        }
    }

    /* First pass: sum features */
    for (size_t i = 0; i < n_samples; i++) {
        int class_label = ds->labels[label_idx].labels[i];
        class_counts[class_label]++;
        for (size_t f = 0; f < n_features; f++) {
            data->mean[class_label][f] += ds->features[feature_idx + f].data[i];
        }
    }

    /* Compute means and priors */
    for (size_t c = 0; c < n_classes; c++) {
        if (class_counts[c] > 0) {
            for (size_t f = 0; f < n_features; f++) {
                data->mean[c][f] /= class_counts[c];
            }
            data->class_prior[c] = log((double)class_counts[c] / n_samples);
        }
    }

    /* Second pass: compute variances */
    for (size_t i = 0; i < n_samples; i++) {
        int class_label = ds->labels[label_idx].labels[i];
        for (size_t f = 0; f < n_features; f++) {
            double diff = ds->features[feature_idx + f].data[i] - data->mean[class_label][f];
            data->var[class_label][f] += diff * diff;
        }
    }

    for (size_t c = 0; c < n_classes; c++) {
        size_t n = class_counts[c];
        if (n > 1) {
            for (size_t f = 0; f < n_features; f++) {
                data->var[c][f] /= (n - 1);
            }
        }
    }

    free(class_counts);
    return m;
}

int predict_gnb(const model* m, const double* x, size_t n) {
    if (!m || m->type != MODEL_GNB || !x) return -1;

    gnb_data* data = (gnb_data*)m->data;
    double best_log_prob = -INFINITY;
    int best_class = 0;

    for (size_t c = 0; c < data->n_classes; c++) {
        double log_prob = data->class_prior[c];
        for (size_t f = 0; f < data->n_features && f < n; f++) {
            log_prob += log_gaussian_pdf(x[f], data->mean[c][f], data->var[c][f], data->epsilon);
        }
        if (log_prob > best_log_prob) {
            best_log_prob = log_prob;
            best_class = (int)c;
        }
    }
    return best_class;
}

void free_gnb(model* m) {
    if (!m || m->type != MODEL_GNB) return;
    gnb_data* data = (gnb_data*)m->data;
    if (data) {
        for (size_t c = 0; c < data->n_classes; c++) {
            free(data->mean[c]);
            free(data->var[c]);
        }
        free(data->mean);
        free(data->var);
        free(data->class_prior);
        free(data);
    }
    m->data = NULL;
}

/* ============================================================================
 * DECISION TREE IMPLEMENTATION
 * ============================================================================ */

static double find_best_split(double* X, int* y, double* y_reg, size_t n,
                              size_t n_features, size_t max_features,
                              size_t* best_feat, bool is_regression) {
    double best_score = is_regression ? -1e30 : 1.0;
    *best_feat = 0;
    double best_thresh = X[0];

    size_t features_to_try = max_features > 0 && max_features < n_features ?
                             max_features : n_features;

    for (size_t f = 0; f < features_to_try; f++) {
        double min_val = X[f], max_val = X[f];
        for (size_t i = 1; i < n; i++) {
            double v = X[i * n_features + f];
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }

        if (fabs(max_val - min_val) < 1e-10) continue;

        for (int t = 1; t <= 5; t++) {
            double thresh = min_val + (max_val - min_val) * t / 5.0;

            size_t n_left = 0, n_right = 0;
            for (size_t i = 0; i < n; i++) {
                if (X[i * n_features + f] <= thresh) n_left++;
                else n_right++;
            }

            if (n_left == 0 || n_right == 0) continue;

            double score;
            if (is_regression) {
                double var_parent = variance(y_reg, n);
                double* left_vals = (double*)malloc(sizeof(double) * n_left);
                double* right_vals = (double*)malloc(sizeof(double) * n_right);
                size_t li = 0, ri = 0;

                for (size_t i = 0; i < n; i++) {
                    if (X[i * n_features + f] <= thresh)
                        left_vals[li++] = y_reg[i];
                    else
                        right_vals[ri++] = y_reg[i];
                }

                double var_left = variance(left_vals, n_left);
                double var_right = variance(right_vals, n_right);
                free(left_vals); free(right_vals);

                double var_reduction = var_parent -
                    (double)n_left / n * var_left -
                    (double)n_right / n * var_right;
                score = var_reduction;
            } else {
                int* left_labels = (int*)malloc(sizeof(int) * n_left);
                int* right_labels = (int*)malloc(sizeof(int) * n_right);
                size_t li = 0, ri = 0;

                for (size_t i = 0; i < n; i++) {
                    if (X[i * n_features + f] <= thresh)
                        left_labels[li++] = y[i];
                    else
                        right_labels[ri++] = y[i];
                }

                double gini_parent = gini_impurity(y, n);
                double gini_left = gini_impurity(left_labels, n_left);
                double gini_right = gini_impurity(right_labels, n_right);
                free(left_labels); free(right_labels);

                double info_gain = gini_parent -
                    (double)n_left / n * gini_left -
                    (double)n_right / n * gini_right;
                score = info_gain;
            }

            if (score > best_score) {
                best_score = score;
                *best_feat = f;
                best_thresh = thresh;
            }
        }
    }
    return best_thresh;
}

static dt_node* build_tree(double* X, int* y, double* y_reg, size_t n,
                           size_t n_features, size_t max_features,
                           size_t max_depth, size_t min_samples_split,
                           size_t min_samples_leaf, size_t current_depth,
                           bool is_regression) {
    dt_node* node = (dt_node*)malloc(sizeof(dt_node));
    if (!node) return NULL;

    node->is_leaf = false;
    node->left = NULL;
    node->right = NULL;
    node->samples = n;
    node->feature_index = 0;
    node->threshold = 0.0;
    node->class_label = 0;
    node->value = 0.0;

    /* Stopping conditions */
    if (n < min_samples_split || (max_depth > 0 && current_depth >= max_depth)) {
        node->is_leaf = true;
        if (is_regression) {
            double sum = 0.0;
            for (size_t i = 0; i < n; i++) sum += y_reg[i];
            node->value = sum / n;
        } else {
            size_t* votes = (size_t*)calloc(256, sizeof(size_t));
            if (votes) {
                size_t max_votes = 0;
                for (size_t i = 0; i < n; i++) {
                    int c = y[i] >= 0 ? y[i] : 0;
                    votes[c]++;
                    if (votes[c] > max_votes) {
                        max_votes = votes[c];
                        node->class_label = c;
                    }
                }
                free(votes);
            }
        }
        return node;
    }

    /* Check purity */
    bool same = true;
    if (!is_regression) {
        for (size_t i = 1; i < n; i++) {
            if (y[i] != y[0]) { same = false; break; }
        }
    } else {
        for (size_t i = 1; i < n; i++) {
            if (fabs(y_reg[i] - y_reg[0]) > 1e-10) { same = false; break; }
        }
    }
    if (same) {
        node->is_leaf = true;
        if (is_regression) node->value = y_reg[0];
        else node->class_label = y[0];
        return node;
    }

    /* Find best split */
    double thresh = find_best_split(X, y, y_reg, n, n_features, max_features,
                                    &node->feature_index, is_regression);

    /* Split data */
    size_t n_left = 0, n_right = 0;
    for (size_t i = 0; i < n; i++) {
        if (X[i * n_features + node->feature_index] <= thresh) n_left++;
        else n_right++;
    }

    if (n_left < min_samples_leaf || n_right < min_samples_leaf) {
        node->is_leaf = true;
        if (is_regression) {
            double sum = 0.0;
            for (size_t i = 0; i < n; i++) sum += y_reg[i];
            node->value = sum / n;
        } else {
            size_t* votes = (size_t*)calloc(256, sizeof(size_t));
            if (votes) {
                size_t max_votes = 0;
                for (size_t i = 0; i < n; i++) {
                    int c = y[i] >= 0 ? y[i] : 0;
                    votes[c]++;
                    if (votes[c] > max_votes) {
                        max_votes = votes[c];
                        node->class_label = c;
                    }
                }
                free(votes);
            }
        }
        return node;
    }

    node->threshold = thresh;

    double* X_left = (double*)malloc(sizeof(double) * n_left * n_features);
    double* X_right = (double*)malloc(sizeof(double) * n_right * n_features);
    int* y_left = (int*)malloc(sizeof(int) * n_left);
    int* y_right = (int*)malloc(sizeof(int) * n_right);
    double* y_reg_left = is_regression ? (double*)malloc(sizeof(double) * n_left) : NULL;
    double* y_reg_right = is_regression ? (double*)malloc(sizeof(double) * n_right) : NULL;

    if (!X_left || !X_right || !y_left || !y_right ||
        (is_regression && (!y_reg_left || !y_reg_right))) {
        free(X_left); free(X_right); free(y_left); free(y_right);
        free(y_reg_left); free(y_reg_right);
        free(node);
        return NULL;
    }

    size_t li = 0, ri = 0;
    for (size_t i = 0; i < n; i++) {
        if (X[i * n_features + node->feature_index] <= thresh) {
            for (size_t f = 0; f < n_features; f++)
                X_left[li * n_features + f] = X[i * n_features + f];
            y_left[li] = y[i];
            if (is_regression) y_reg_left[li] = y_reg[i];
            li++;
        } else {
            for (size_t f = 0; f < n_features; f++)
                X_right[ri * n_features + f] = X[i * n_features + f];
            y_right[ri] = y[i];
            if (is_regression) y_reg_right[ri] = y_reg[i];
            ri++;
        }
    }

    node->left = build_tree(X_left, y_left, y_reg_left, n_left, n_features,
                            max_features, max_depth, min_samples_split,
                            min_samples_leaf, current_depth + 1, is_regression);
    node->right = build_tree(X_right, y_right, y_reg_right, n_right, n_features,
                             max_features, max_depth, min_samples_split,
                             min_samples_leaf, current_depth + 1, is_regression);

    free(X_left); free(X_right); free(y_left); free(y_right);
    free(y_reg_left); free(y_reg_right);

    return node;
}

static void free_tree(dt_node* node) {
    if (!node) return;
    if (node->left) free_tree(node->left);
    if (node->right) free_tree(node->right);
    free(node);
}

static int predict_tree_dt(const dt_node* node, const double* x) {
    if (!node || node->is_leaf) return node->class_label;
    if (x[node->feature_index] <= node->threshold)
        return predict_tree_dt(node->left, x);
    else
        return predict_tree_dt(node->right, x);
}

static double predict_tree_reg(const dt_node* node, const double* x) {
    if (!node || node->is_leaf) return node->value;
    if (x[node->feature_index] <= node->threshold)
        return predict_tree_reg(node->left, x);
    else
        return predict_tree_reg(node->right, x);
}

model* train_decision_tree(const dataset* ds, size_t feature_idx, size_t label_idx,
                          model_config config) {
    if (!ds || !ds->features || !ds->labels) return NULL;

    size_t n = ds->rows;
    size_t n_features = ds->num_features;
    size_t n_classes = ds->labels[label_idx].classes;
    bool is_regression = (n_classes == 0);

    model* m = (model*)malloc(sizeof(model));
    if (!m) return NULL;

    rf_data* data = (rf_data*)malloc(sizeof(rf_data));
    if (!data) { free(m); return NULL; }

    m->type = MODEL_DECISION_TREE;
    m->data = data;

    data->n_estimators = 1;
    data->max_depth = config.max_depth;
    data->n_classes = n_classes;
    data->is_regression = is_regression;
    data->max_features = n_features;

    double* X = (double*)malloc(sizeof(double) * n * n_features);
    int* y = (int*)malloc(sizeof(int) * n);
    double* y_reg = is_regression ? (double*)malloc(sizeof(double) * n) : NULL;

    if (!X || !y || (is_regression && !y_reg)) {
        free(X); free(y); free(y_reg); free(data); free(m);
        return NULL;
    }

    for (size_t i = 0; i < n; i++) {
        for (size_t f = 0; f < n_features; f++)
            X[i * n_features + f] = ds->features[feature_idx + f].data[i];
        y[i] = ds->labels[label_idx].labels[i];
        if (is_regression) y_reg[i] = (double)y[i];
    }

    data->trees = (dt_node**)malloc(sizeof(dt_node*));
    if (!data->trees) {
        free(X); free(y); free(y_reg); free(data); free(m);
        return NULL;
    }

    data->trees[0] = build_tree(X, y, y_reg, n, n_features, n_features,
                                config.max_depth, config.min_samples_split,
                                config.min_samples_leaf, 0, is_regression);

    free(X); free(y); free(y_reg);

    if (!data->trees[0]) {
        free(data->trees); free(data); free(m);
        return NULL;
    }

    return m;
}

double predict_decision_tree(const model* m, const double* x, size_t n) {
    if (!m || m->type != MODEL_DECISION_TREE || !x) return 0.0;
    rf_data* data = (rf_data*)m->data;
    (void)n;

    if (data->is_regression)
        return predict_tree_reg(data->trees[0], x);
    else
        return (double)predict_tree_dt(data->trees[0], x);
}

void free_decision_tree(model* m) {
    if (!m || m->type != MODEL_DECISION_TREE) return;
    rf_data* data = (rf_data*)m->data;
    if (data) {
        if (data->trees) {
            free_tree(data->trees[0]);
            free(data->trees);
        }
        free(data);
    }
    m->data = NULL;
}

/* ============================================================================
 * RANDOM FOREST IMPLEMENTATION
 * ============================================================================ */

model* train_random_forest(const dataset* ds, size_t feature_idx, size_t label_idx,
                          model_config config) {
    if (!ds || !ds->features || !ds->labels) return NULL;

    size_t n_samples = ds->rows;
    size_t n_features = ds->num_features;
    size_t n_classes = ds->labels[label_idx].classes;
    bool is_regression = (n_classes == 0);

    model* m = (model*)malloc(sizeof(model));
    if (!m) return NULL;

    rf_data* data = (rf_data*)malloc(sizeof(rf_data));
    if (!data) { free(m); return NULL; }

    m->type = MODEL_RANDOM_FOREST;
    m->data = data;

    data->n_estimators = config.n_estimators > 0 ? config.n_estimators : 100;
    data->max_depth = config.max_depth;
    data->subsample_ratio = config.learning_rate > 0 ? config.learning_rate : 1.0;
    data->n_classes = n_classes;
    data->is_regression = is_regression;
    data->max_features = is_regression ? n_features :
                          (size_t)(sqrt((double)n_features) + 0.5);
    if (data->max_features < 1) data->max_features = 1;

    data->trees = (dt_node**)malloc(sizeof(dt_node*) * data->n_estimators);
    if (!data->trees) { free(data); free(m); return NULL; }

    double* X = (double*)malloc(sizeof(double) * n_samples * n_features);
    int* y = (int*)malloc(sizeof(int) * n_samples);
    double* y_reg = is_regression ? (double*)malloc(sizeof(double) * n_samples) : NULL;

    if (!X || !y || (is_regression && !y_reg)) {
        free(X); free(y); free(y_reg); free(data->trees); free(data); free(m);
        return NULL;
    }

    for (size_t i = 0; i < n_samples; i++) {
        for (size_t f = 0; f < n_features; f++)
            X[i * n_features + f] = ds->features[feature_idx + f].data[i];
        y[i] = ds->labels[label_idx].labels[i];
        if (is_regression) y_reg[i] = (double)y[i];
    }

    size_t subsample_size = (size_t)(n_samples * data->subsample_ratio);
    if (subsample_size < 1) subsample_size = 1;

    unsigned int seed = (unsigned int)time(NULL);

    for (size_t t = 0; t < data->n_estimators; t++) {
        srand(seed + (unsigned int)t);

        size_t* indices = (size_t*)malloc(sizeof(size_t) * subsample_size);
        if (!indices) { data->trees[t] = NULL; continue; }

        for (size_t i = 0; i < subsample_size; i++)
            indices[i] = rand() % n_samples;

        double* X_boot = (double*)malloc(sizeof(double) * subsample_size * n_features);
        int* y_boot = (int*)malloc(sizeof(int) * subsample_size);
        double* y_reg_boot = is_regression ? (double*)malloc(sizeof(double) * subsample_size) : NULL;

        if (!X_boot || !y_boot || (is_regression && !y_reg_boot)) {
            free(indices); free(X_boot); free(y_boot); free(y_reg_boot);
            data->trees[t] = NULL;
            continue;
        }

        for (size_t i = 0; i < subsample_size; i++) {
            size_t idx = indices[i];
            for (size_t f = 0; f < n_features; f++)
                X_boot[i * n_features + f] = X[idx * n_features + f];
            y_boot[i] = y[idx];
            if (is_regression) y_reg_boot[i] = y_reg[idx];
        }

        data->trees[t] = build_tree(X_boot, y_boot, y_reg_boot, subsample_size,
                                    n_features, data->max_features,
                                    data->max_depth, config.min_samples_split,
                                    config.min_samples_leaf, 0, is_regression);

        free(indices); free(X_boot); free(y_boot); free(y_reg_boot);
    }

    free(X); free(y); free(y_reg);
    return m;
}

double predict_random_forest(const model* m, const double* x, size_t n) {
    if (!m || m->type != MODEL_RANDOM_FOREST || !x) return 0.0;
    rf_data* data = (rf_data*)m->data;
    (void)n;

    if (data->is_regression) {
        double sum = 0.0;
        size_t valid_trees = 0;
        for (size_t t = 0; t < data->n_estimators; t++) {
            if (data->trees[t]) {
                sum += predict_tree_reg(data->trees[t], x);
                valid_trees++;
            }
        }
        return valid_trees > 0 ? sum / valid_trees : 0.0;
    } else {
        size_t* votes = (size_t*)calloc(data->n_classes > 256 ? data->n_classes : 256,
                                        sizeof(size_t));
        if (!votes) return 0.0;

        for (size_t t = 0; t < data->n_estimators; t++) {
            if (data->trees[t]) {
                int pred = predict_tree_dt(data->trees[t], x);
                if (pred >= 0 && (size_t)pred < data->n_classes)
                    votes[pred]++;
            }
        }

        size_t best_class = 0;
        size_t max_votes = 0;
        for (size_t c = 0; c < data->n_classes; c++) {
            if (votes[c] > max_votes) {
                max_votes = votes[c];
                best_class = c;
            }
        }
        free(votes);
        return (double)best_class;
    }
}

void free_random_forest(model* m) {
    if (!m || m->type != MODEL_RANDOM_FOREST) return;
    rf_data* data = (rf_data*)m->data;
    if (data) {
        if (data->trees) {
            for (size_t t = 0; t < data->n_estimators; t++)
                if (data->trees[t]) free_tree(data->trees[t]);
            free(data->trees);
        }
        free(data);
    }
    m->data = NULL;
}

/* ============================================================================
 * ADABOOST IMPLEMENTATION
 * ============================================================================ */

static int predict_stump(const adaboost_stump* stump, const double* x) {
    if (x[stump->feature_index] <= stump->threshold)
        return stump->left_class;
    else
        return stump->right_class;
}

static adaboost_stump* train_stump(double* X, int* y, double* weights,
                                   size_t n, size_t n_features, size_t n_classes) {
    adaboost_stump* stump = (adaboost_stump*)malloc(sizeof(adaboost_stump));
    if (!stump) return NULL;

    double best_error = 1e30;
    stump->feature_index = 0;
    stump->threshold = X[0];
    stump->left_class = 0;
    stump->right_class = 0;
    stump->alpha = 1.0;

    for (size_t f = 0; f < n_features; f++) {
        double min_val = X[f], max_val = X[f];
        for (size_t i = 1; i < n; i++) {
            double v = X[i * n_features + f];
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }

        if (fabs(max_val - min_val) < 1e-10) continue;

        for (int t = 1; t <= 10; t++) {
            double thresh = min_val + (max_val - min_val) * t / 10.0;

            for (size_t c = 0; c < n_classes; c++) {
                int left_class = (int)c;
                int right_class = (int)((c + 1) % n_classes);

                double error = 0.0;
                for (size_t i = 0; i < n; i++) {
                    int pred = (X[i * n_features + f] <= thresh) ? left_class : right_class;
                    if (pred != y[i]) error += weights[i];
                }

                if (error < best_error) {
                    best_error = error;
                    stump->feature_index = f;
                    stump->threshold = thresh;
                    stump->left_class = left_class;
                    stump->right_class = right_class;
                }
            }
        }
    }

    if (best_error < 1e-10) best_error = 1e-10;
    if (best_error > 1.0 - 1e-10) best_error = 1.0 - 1e-10;
    stump->alpha = 0.5 * log((1.0 - best_error) / best_error);

    return stump;
}

model* train_adaboost(const dataset* ds, size_t feature_idx, size_t label_idx,
                     model_config config) {
    if (!ds || !ds->features || !ds->labels) return NULL;

    size_t n_samples = ds->rows;
    size_t n_features = ds->num_features;
    size_t n_classes = ds->labels[label_idx].classes;
    size_t n_stumps = config.n_estimators > 0 ? config.n_estimators : 50;

    model* m = (model*)malloc(sizeof(model));
    if (!m) return NULL;

    adaboost_data* data = (adaboost_data*)malloc(sizeof(adaboost_data));
    if (!data) { free(m); return NULL; }

    m->type = MODEL_ADABOOST;
    m->data = data;

    data->n_stumps = n_stumps;
    data->n_classes = n_classes;
    data->stumps = (adaboost_stump*)malloc(sizeof(adaboost_stump) * n_stumps);
    data->weights = (double*)malloc(sizeof(double) * n_samples);

    if (!data->stumps || !data->weights) {
        free(data->stumps); free(data->weights); free(data); free(m);
        return NULL;
    }

    double* X = (double*)malloc(sizeof(double) * n_samples * n_features);
    int* y = (int*)malloc(sizeof(int) * n_samples);

    if (!X || !y) {
        free(X); free(y); free(data->stumps); free(data->weights);
        free(data); free(m);
        return NULL;
    }

    for (size_t i = 0; i < n_samples; i++) {
        for (size_t f = 0; f < n_features; f++)
            X[i * n_features + f] = ds->features[feature_idx + f].data[i];
        y[i] = ds->labels[label_idx].labels[i];
    }

    for (size_t i = 0; i < n_samples; i++)
        data->weights[i] = 1.0 / n_samples;

    for (size_t t = 0; t < n_stumps; t++) {
        adaboost_stump* stump = train_stump(X, y, data->weights, n_samples,
                                            n_features, n_classes);
        if (!stump) {
            data->stumps[t].alpha = 0.0;
            continue;
        }

        data->stumps[t] = *stump;
        free(stump);

        double error = 0.0;
        for (size_t i = 0; i < n_samples; i++) {
            int pred = predict_stump(&data->stumps[t], &X[i * n_features]);
            if (pred != y[i]) error += data->weights[i];
        }

        if (error < 1e-10) error = 1e-10;
        if (error > 1.0 - 1e-10) error = 1.0 - 1e-10;

        double alpha = data->stumps[t].alpha;
        double weight_sum = 0.0;
        for (size_t i = 0; i < n_samples; i++) {
            int pred = predict_stump(&data->stumps[t], &X[i * n_features]);
            double y_i = (pred == y[i]) ? 1.0 : -1.0;
            data->weights[i] *= exp(-alpha * y_i);
            weight_sum += data->weights[i];
        }

        if (weight_sum > 0) {
            for (size_t i = 0; i < n_samples; i++)
                data->weights[i] /= weight_sum;
        }
    }

    free(X); free(y);
    return m;
}

int predict_adaboost(const model* m, const double* x, size_t n) {
    (void)n;
    if (!m || m->type != MODEL_ADABOOST || !x) return 0;

    adaboost_data* data = (adaboost_data*)m->data;

    double* class_scores = (double*)calloc(data->n_classes, sizeof(double));
    if (!class_scores) return 0;

    for (size_t t = 0; t < data->n_stumps; t++) {
        if (data->stumps[t].alpha <= 0) continue;
        int pred = predict_stump(&data->stumps[t], x);
        if (pred >= 0 && (size_t)pred < data->n_classes) {
            class_scores[pred] += data->stumps[t].alpha;
        }
    }

    int best_class = 0;
    double max_score = class_scores[0];
    for (size_t c = 1; c < data->n_classes; c++) {
        if (class_scores[c] > max_score) {
            max_score = class_scores[c];
            best_class = (int)c;
        }
    }

    free(class_scores);
    return best_class;
}

void free_adaboost(model* m) {
    if (!m || m->type != MODEL_ADABOOST) return;
    adaboost_data* data = (adaboost_data*)m->data;
    if (data) {
        free(data->stumps);
        free(data->weights);
        free(data);
    }
    m->data = NULL;
}

/* ============================================================================
 * KNN IMPLEMENTATION
 * ============================================================================ */

model* train_knn(const dataset* ds, size_t feature_idx, size_t label_idx,
                model_config config) {
    if (!ds || !ds->features || !ds->labels) return NULL;

    size_t n_samples = ds->rows;
    size_t n_features = ds->num_features;
    size_t k = config.k_value > 0 ? config.k_value : 5;
    if (k > n_samples) k = n_samples;

    model* m = (model*)malloc(sizeof(model));
    if (!m) return NULL;

    knn_data* data = (knn_data*)malloc(sizeof(knn_data));
    if (!data) { free(m); return NULL; }

    m->type = MODEL_KNN;
    m->data = data;

    data->is_regression = (ds->labels[label_idx].classes == 0 ||
                          ds->labels[label_idx].value_map == NULL);
    data->n_samples = n_samples;
    data->n_features = n_features;
    data->k = k;

    data->X = (double*)malloc(sizeof(double) * n_samples * n_features);
    data->y = (double*)malloc(sizeof(double) * n_samples);

    if (!data->X || !data->y) {
        free(data->X); free(data->y); free(data); free(m);
        return NULL;
    }

    for (size_t i = 0; i < n_samples; i++) {
        for (size_t f = 0; f < n_features; f++)
            data->X[i * n_features + f] = ds->features[feature_idx + f].data[i];
        data->y[i] = (double)ds->labels[label_idx].labels[i];
    }

    return m;
}

double predict_knn(const model* m, const double* x, size_t n) {
    if (!m || m->type != MODEL_KNN || !x) return 0.0;

    knn_data* data = (knn_data*)m->data;

    double* distances = (double*)malloc(sizeof(double) * data->n_samples);
    if (!distances) return 0.0;

    for (size_t i = 0; i < data->n_samples; i++) {
        double dist = 0.0;
        for (size_t f = 0; f < data->n_features && f < n; f++) {
            double diff = x[f] - data->X[i * data->n_features + f];
            dist += diff * diff;
        }
        distances[i] = sqrt(dist);
    }

    size_t k = data->k < data->n_samples ? data->k : data->n_samples;

    size_t* nearest_indices = (size_t*)malloc(sizeof(size_t) * k);
    double* nearest_distances = (double*)malloc(sizeof(double) * k);

    if (!nearest_indices || !nearest_distances) {
        free(distances); free(nearest_indices); free(nearest_distances);
        return 0.0;
    }

    for (size_t i = 0; i < k; i++) {
        nearest_indices[i] = i;
        nearest_distances[i] = distances[i];
    }

    for (size_t i = k; i < data->n_samples; i++) {
        size_t max_idx = 0;
        double max_dist = nearest_distances[0];
        for (size_t j = 1; j < k; j++) {
            if (nearest_distances[j] > max_dist) {
                max_dist = nearest_distances[j];
                max_idx = j;
            }
        }
        if (distances[i] < max_dist) {
            nearest_distances[max_idx] = distances[i];
            nearest_indices[max_idx] = i;
        }
    }

    double result;
    if (data->is_regression) {
        double sum = 0.0;
        for (size_t i = 0; i < k; i++)
            sum += data->y[nearest_indices[i]];
        result = sum / k;
    } else {
        size_t* class_counts = (size_t*)calloc(256, sizeof(size_t));
        if (!class_counts) {
            free(distances); free(nearest_indices); free(nearest_distances);
            return 0.0;
        }

        for (size_t i = 0; i < k; i++)
            class_counts[(size_t)data->y[nearest_indices[i]]]++;

        size_t best_class = 0;
        size_t max_votes = 0;
        for (size_t c = 0; c < 256; c++) {
            if (class_counts[c] > max_votes) {
                max_votes = class_counts[c];
                best_class = c;
            }
        }
        result = (double)best_class;
        free(class_counts);
    }

    free(distances);
    free(nearest_indices);
    free(nearest_distances);

    return result;
}

void free_knn(model* m) {
    if (!m || m->type != MODEL_KNN) return;
    knn_data* data = (knn_data*)m->data;
    if (data) {
        free(data->X);
        free(data->y);
        free(data);
    }
    m->data = NULL;
}

/* ============================================================================
 * EVALUATION 评估
 * ============================================================================ */

double accuracy(const model* m, const dataset* ds, size_t feature_idx, size_t label_idx) {
    if (!m || !ds) return 0.0;
    size_t n = ds->rows, n_feat = ds->num_features;
    size_t correct = 0;

    for (size_t i = 0; i < n; i++) {
        double* x = (double*)malloc(sizeof(double) * n_feat);
        if (!x) continue;
        for (size_t f = 0; f < n_feat; f++)
            x[f] = ds->features[feature_idx + f].data[i];

        int pred = -1;
        if (m->type == MODEL_GNB)
            pred = predict_gnb(m, x, n_feat);
        else if (m->type == MODEL_DECISION_TREE)
            pred = (int)predict_decision_tree(m, x, n_feat);
        else if (m->type == MODEL_RANDOM_FOREST)
            pred = (int)predict_random_forest(m, x, n_feat);
        else if (m->type == MODEL_ADABOOST)
            pred = predict_adaboost(m, x, n_feat);
        else if (m->type == MODEL_KNN)
            pred = (int)predict_knn(m, x, n_feat);

        if ((int)ds->labels[label_idx].labels[i] == pred) correct++;
        free(x);
    }
    return (double)correct / n;
}

double mse(const model* m, const dataset* ds, size_t feature_idx, size_t label_idx) {
    if (!m || !ds) return 0.0;
    size_t n = ds->rows, n_feat = ds->num_features;
    double sum_sq_err = 0.0;

    for (size_t i = 0; i < n; i++) {
        double* x = (double*)malloc(sizeof(double) * n_feat);
        if (!x) continue;
        for (size_t f = 0; f < n_feat; f++)
            x[f] = ds->features[feature_idx + f].data[i];

        double pred = 0.0;
        if (m->type == MODEL_DECISION_TREE)
            pred = predict_decision_tree(m, x, n_feat);
        else if (m->type == MODEL_RANDOM_FOREST)
            pred = predict_random_forest(m, x, n_feat);
        else if (m->type == MODEL_KNN)
            pred = predict_knn(m, x, n_feat);

        double actual = (double)ds->labels[label_idx].labels[i];
        double err = pred - actual;
        sum_sq_err += err * err;
        free(x);
    }
    return sum_sq_err / n;
}

double r2_score(const model* m, const dataset* ds, size_t feature_idx, size_t label_idx) {
    if (!m || !ds) return 0.0;
    size_t n = ds->rows;

    double y_mean = 0.0;
    double* y_true = (double*)malloc(sizeof(double) * n);
    double* y_pred = (double*)malloc(sizeof(double) * n);

    if (!y_true || !y_pred) { free(y_true); free(y_pred); return 0.0; }

    size_t n_feat = ds->num_features;
    for (size_t i = 0; i < n; i++) {
        y_true[i] = (double)ds->labels[label_idx].labels[i];
        y_mean += y_true[i];

        double* x = (double*)malloc(sizeof(double) * n_feat);
        if (x) {
            for (size_t f = 0; f < n_feat; f++)
                x[f] = ds->features[feature_idx + f].data[i];

            if (m->type == MODEL_DECISION_TREE)
                y_pred[i] = predict_decision_tree(m, x, n_feat);
            else if (m->type == MODEL_RANDOM_FOREST)
                y_pred[i] = predict_random_forest(m, x, n_feat);
            else if (m->type == MODEL_KNN)
                y_pred[i] = predict_knn(m, x, n_feat);

            free(x);
        }
    }
    y_mean /= n;

    double ss_res = 0.0, ss_tot = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff_pred = y_pred[i] - y_true[i];
        double diff_mean = y_true[i] - y_mean;
        ss_res += diff_pred * diff_pred;
        ss_tot += diff_mean * diff_mean;
    }

    free(y_true); free(y_pred);
    if (ss_tot < 1e-10) return 0.0;
    return 1.0 - ss_res / ss_tot;
}

void model_free(model* m) {
    if (!m) return;
    switch (m->type) {
        case MODEL_GNB: free_gnb(m); break;
        case MODEL_DECISION_TREE: free_decision_tree(m); break;
        case MODEL_RANDOM_FOREST: free_random_forest(m); break;
        case MODEL_ADABOOST: free_adaboost(m); break;
        case MODEL_KNN: free_knn(m); break;
    }
}

#endif /* ML_IMPLEMENTATION */

#endif /* ML_SAMPLES_H */
