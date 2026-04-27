# C Machine Learning 框架指南

## 概述

`machine_learning.h` 是一个单头文件的 C 语言机器学习框架，参考 stb 风格设计。

### 特性

- 特征缩放（Standard、MinMax、MaxAbs）
- 训练/测试集分割
- K 折交叉验证
- 模型评估指标
- 统一模型接口

---

## 快速开始

### 1. 引入头文件

```c
#define C_MACHINE_LEARNING_IMPLEMENTATION
#include "machine_learning.h"
```

将 `C_MACHINE_LEARNING_IMPLEMENTATION` 宏定义在 include 之前，可以在一个 .c 文件中包含实现。

### 2. 基本使用流程

```c
// 1. 加载数据
csv_t* csv = csv_load("data.csv");
const char* labels[] = {"target_column"};
dataset* ds = csv_to_dataset(csv, labels, 1);

// 2. 分割数据
Dataset_Split_t split;
train_test_split(ds, 0.2, 42, &split);

// 3. 特征缩放
ML_ScalingParams_t* params = ml_fit_scaling(
    ds, feature_indices, n_features,
    split.train_indices, split.train_size,
    SCALING_STANDARD
);
dataset* scaled_ds = ml_transform_features(
    params, ds, feature_indices, n_features,
    split.train_indices, split.train_size
);

// 4. 创建并训练模型
ML_Model_t model = {
    .type = ML_CLASSIFICATION,
    .methods = { my_fit, my_predict, NULL, NULL, NULL, NULL, my_free }
};
model.methods.fit(&model.config, &model.state, scaled_ds,
                  feature_indices, n_features, target_index,
                  split.train_indices, split.train_size);

// 5. 评估模型
ML_Classification_Metrics_t metrics;
model_evaluate(&model, scaled_ds, feature_indices, n_features,
               target_index, split.test_indices, split.test_size,
               &metrics);
printf("Accuracy: %.2f\n", metrics.Accuracy);

// 6. 释放资源
ml_free_scaling_params(params);
free_dataset(scaled_ds);
free(split.train_indices);
free(split.test_indices);
free_dataset(ds);
```

---

## 核心数据结构

### ML_ModelType_t

模型类型枚举：

```c
typedef enum {
    ML_REGRESSION = 0x1,     // 回归模型
    ML_CLASSIFICATION = 0x2  // 分类模型
} ML_ModelType_t;
```

### ML_Model_Impl_t

模型方法接口，包含以下函数指针：

| 方法 | 说明 |
|------|------|
| `fit` | 训练模型 |
| `predict` | 预测类别/数值 |
| `predict_proba` | 预测概率（分类） |
| `get_coefficients` | 获取模型系数 |
| `serialize` | 序列化模型 |
| `deserialize` | 反序列化模型 |
| `free_state` | 释放模型状态 |

### ML_ScalingParams_t

特征缩放参数结构：

```c
typedef struct ML_ScalingParams_t {
    FeatureScaling_t type;      // 缩放类型
    double* mean;                // 均值（Standard）
    double* std;                 // 标准差（Standard）
    double* min;                 // 最小值（MinMax）
    double* max;                 // 最大值（MinMax）
    double* maxabs;              // 最大绝对值（MaxAbs）
    size_t n_features;
    size_t* sample_indices;       // 拟合时使用的样本索引
} ML_ScalingParams_t;
```

---

## 方法检查接口

框架提供了检查模型方法是否已实现的功能：

```c
// 检查单个方法
if (ml_method_is_implemented(&model.methods, ML_METHOD_PREDICT)) {
    // predict 方法已实现
}

// 获取已实现方法的位掩码
unsigned int mask = ml_methods_get_implemented(&model.methods);
// mask 的第 i 位对应 ML_METHOD_FIT + i
```

---

## 实现自定义模型

### 模型接口函数签名

```c
// fit - 训练模型
int my_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
           const dataset* ds, const size_t* feature_indices, size_t n_features,
           size_t target_index, const size_t* sample_indices, size_t n_samples);

// predict - 预测
int my_predict(const ML_Weights_t* state, const dataset* ds,
               const size_t* feature_indices, size_t n_features,
               const size_t* sample_indices, size_t n_samples, void* output);

// free_state - 释放状态
void my_free_state(ML_Weights_t* state);
```

### 实现示例：Gaussian Naive Bayes (GNB)

```c
/*
 * Gaussian Naive Bayes 分类器
 *
 * 原理：假设每个类别的特征服从正态分布，
 *       P(x|y) = N(mu, sigma)
 */

#define C_MACHINE_LEARNING_IMPLEMENTATION
#include "machine_learning.h"
#include <math.h>
#include <stdlib.h>

typedef struct {
    double* means;      // 每类的特征均值 [n_classes * n_features]
    double* variances;  // 每类的特征方差 [n_classes * n_features]
    size_t n_classes;
    size_t n_features;
    double* class_priors;  // 类先验概率 [n_classes]
} GNB_State;

static int gnb_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
                   const dataset* ds, const size_t* feature_indices, size_t n_features,
                   size_t target_index, const size_t* sample_indices, size_t n_samples) {
    (void)config;

    size_t n_classes = ds->labels[target_index].classes;
    GNB_State* gnb = (GNB_State*)malloc(sizeof(GNB_State));
    if (!gnb) return -1;

    gnb->n_classes = n_classes;
    gnb->n_features = n_features;
    gnb->means = (double*)calloc(n_classes * n_features, sizeof(double));
    gnb->variances = (double*)calloc(n_classes * n_features, sizeof(double));
    gnb->class_priors = (double*)calloc(n_classes, sizeof(double));

    if (!gnb->means || !gnb->variances || !gnb->class_priors) {
        free(gnb->means); free(gnb->variances); free(gnb->class_priors); free(gnb);
        return -1;
    }

    // 计算每个类的样本数
    size_t* class_counts = (size_t*)calloc(n_classes, sizeof(size_t));
    for (size_t i = 0; i < n_samples; i++) {
        int label = ds->labels[target_index].labels[sample_indices[i]];
        class_counts[label]++;
    }

    // 计算类先验
    for (size_t c = 0; c < n_classes; c++) {
        gnb->class_priors[c] = (double)class_counts[c] / n_samples;
    }

    // 计算每个类的每个特征的均值
    for (size_t i = 0; i < n_samples; i++) {
        int label = ds->labels[target_index].labels[sample_indices[i]];
        for (size_t f = 0; f < n_features; f++) {
            double val = ds->features[feature_indices[f]].data[sample_indices[i]];
            gnb->means[label * n_features + f] += val;
        }
    }
    for (size_t c = 0; c < n_classes; c++) {
        if (class_counts[c] > 0) {
            for (size_t f = 0; f < n_features; f++) {
                gnb->means[c * n_features + f] /= class_counts[c];
            }
        }
    }

    // 计算每个类的每个特征的方差
    for (size_t i = 0; i < n_samples; i++) {
        int label = ds->labels[target_index].labels[sample_indices[i]];
        for (size_t f = 0; f < n_features; f++) {
            double val = ds->features[feature_indices[f]].data[sample_indices[i]];
            double mean = gnb->means[label * n_features + f];
            double diff = val - mean;
            gnb->variances[label * n_features + f] += diff * diff;
        }
    }
    for (size_t c = 0; c < n_classes; c++) {
        if (class_counts[c] > 0) {
            for (size_t f = 0; f < n_features; f++) {
                gnb->variances[c * n_features + f] /= class_counts[c];
                // 添加小的平滑项防止除零
                if (gnb->variances[c * n_features + f] < 1e-10) {
                    gnb->variances[c * n_features + f] = 1.0;
                }
            }
        }
    }

    free(class_counts);
    state->weights = gnb;
    state->size = sizeof(GNB_State);
    return 0;
}

static double gnb_gaussian_pdf(double x, double mean, double variance) {
    double coeff = 1.0 / sqrt(2.0 * 3.14159265358979 * variance);
    double exponent = -((x - mean) * (x - mean)) / (2.0 * variance);
    return coeff * exp(exponent);
}

static int gnb_predict(const ML_Weights_t* state, const dataset* ds,
                       const size_t* feature_indices, size_t n_features,
                       const size_t* sample_indices, size_t n_samples,
                       void* output) {
    GNB_State* gnb = (GNB_State*)state->weights;
    int* predictions = (int*)output;

    for (size_t s = 0; s < n_samples; s++) {
        size_t idx = sample_indices[s];
        double best_log_prob = -1e300;
        int best_class = 0;

        for (size_t c = 0; c < gnb->n_classes; c++) {
            double log_prob = log(gnb->class_priors[c] + 1e-10);

            for (size_t f = 0; f < n_features; f++) {
                double x = ds->features[feature_indices[f]].data[idx];
                double mean = gnb->means[c * n_features + f];
                double var = gnb->variances[c * n_features + f];
                log_prob += log(gnb_gaussian_pdf(x, mean, var) + 1e-10);
            }

            if (log_prob > best_log_prob) {
                best_log_prob = log_prob;
                best_class = (int)c;
            }
        }
        predictions[s] = best_class;
    }
    return 0;
}

static void gnb_free_state(ML_Weights_t* state) {
    if (state->weights) {
        GNB_State* gnb = (GNB_State*)state->weights;
        free(gnb->means);
        free(gnb->variances);
        free(gnb->class_priors);
        free(gnb);
        state->weights = NULL;
    }
}

static ML_Model_t create_gnb_model(void) {
    return (ML_Model_t){
        .type = ML_CLASSIFICATION,
        .config = {NULL, 0},
        .state = {NULL, 0},
        .methods = {
            .fit = gnb_fit,
            .predict = gnb_predict,
            .predict_proba = NULL,
            .get_coefficients = NULL,
            .serialize = NULL,
            .deserialize = NULL,
            .free_state = gnb_free_state
        }
    };
}
```

### 实现示例：决策树分类器（简版）

```c
/*
 * 决策树分类器（ID3 算法的简化版本）
 *
 * 使用信息增益（熵减）来选择最佳分割特征和阈值
 */

#define C_MACHINE_LEARNING_IMPLEMENTATION
#include "machine_learning.h"
#include <math.h>
#include <stdlib.h>

typedef struct DT_Node {
    int is_leaf;
    int class_label;           // 叶子节点的类别
    size_t feature_index;      // 分割特征
    double threshold;           // 分割阈值
    struct DT_Node* left;      // 左子节点 (<= threshold)
    struct DT_Node* right;     // 右子节点 (> threshold)
} DT_Node;

typedef struct {
    DT_Node* root;
    size_t n_classes;
} DT_State;

static double dt_entropy(int* labels, size_t n) {
    if (n == 0) return 0.0;
    size_t* counts = (size_t*)calloc(256, sizeof(size_t));  // 假设类别数 < 256
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
    // 计算父节点熵
    int* parent_labels = (int*)malloc(sizeof(int) * n);
    for (size_t i = 0; i < n; i++)
        parent_labels[i] = (int)ds->labels[target_index].labels[indices[i]];
    double parent_entropy = dt_entropy(parent_labels, n);
    free(parent_labels);

    // 分割
    size_t* left_idx = (size_t*)malloc(sizeof(size_t) * n);
    size_t* right_idx = (size_t*)malloc(sizeof(size_t) * n);
    size_t n_left = 0, n_right = 0;

    for (size_t i = 0; i < n; i++) {
        double val = ds->features[feature_idx].data[indices[i]];
        if (val <= threshold) {
            left_idx[n_left++] = indices[i];
        } else {
            right_idx[n_right++] = indices[i];
        }
    }

    // 子节点熵
    int* left_labels = (int*)malloc(sizeof(int) * n_left);
    int* right_labels = (int*)malloc(sizeof(int) * n_right);
    for (size_t i = 0; i < n_left; i++)
        left_labels[i] = (int)ds->labels[target_index].labels[left_idx[i]];
    for (size_t i = 0; i < n_right; i++)
        right_labels[i] = (int)ds->labels[target_index].labels[right_idx[i]];

    double child_entropy = (n_left * dt_entropy(left_labels, n_left) +
                           n_right * dt_entropy(right_labels, n_right)) / n;

    free(left_idx); free(right_idx);
    free(left_labels); free(right_labels);

    return parent_entropy - child_entropy;
}

static DT_Node* dt_build_tree(const dataset* ds, size_t target_index,
                              const size_t* indices, size_t n,
                              size_t n_features, const size_t* feature_indices,
                              size_t max_depth, size_t min_samples_split) {
    DT_Node* node = (DT_Node*)malloc(sizeof(DT_Node));
    node->is_leaf = 0;
    node->left = NULL;
    node->right = NULL;

    // 检查是否所有样本属于同一类
    int first_label = (int)ds->labels[target_index].labels[indices[0]];
    int all_same = 1;
    for (size_t i = 1; i < n; i++) {
        if ((int)ds->labels[target_index].labels[indices[i]] != first_label) {
            all_same = 0;
            break;
        }
    }

    if (all_same || n < min_samples_split || max_depth == 0) {
        node->is_leaf = 1;
        node->class_label = first_label;
        return node;
    }

    // 找最佳分割
    double best_gain = -1.0;
    size_t best_feature = 0;
    double best_threshold = 0.0;

    for (size_t f = 0; f < n_features; f++) {
        size_t feat_idx = feature_indices[f];
        double min_val = ds->features[feat_idx].data[indices[0]];
        double max_val = min_val;

        for (size_t i = 1; i < n; i++) {
            double val = ds->features[feat_idx].data[indices[i]];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }

        // 尝试多个候选阈值
        for (int t = 1; t <= 5; t++) {
            double threshold = min_val + (max_val - min_val) * t / 6.0;
            double gain = dt_info_gain(ds, target_index, feat_idx, threshold, indices, n);
            if (gain > best_gain) {
                best_gain = gain;
                best_feature = feat_idx;
                best_threshold = threshold;
            }
        }
    }

    if (best_gain <= 0.0) {
        node->is_leaf = 1;
        node->class_label = first_label;
        return node;
    }

    node->feature_index = best_feature;
    node->threshold = best_threshold;

    // 分割样本
    size_t* left_idx = (size_t*)malloc(sizeof(size_t) * n);
    size_t* right_idx = (size_t*)malloc(sizeof(size_t) * n);
    size_t n_left = 0, n_right = 0;

    for (size_t i = 0; i < n; i++) {
        double val = ds->features[best_feature].data[indices[i]];
        if (val <= best_threshold) {
            left_idx[n_left++] = indices[i];
        } else {
            right_idx[n_right++] = indices[i];
        }
    }

    node->left = dt_build_tree(ds, target_index, left_idx, n_left,
                               n_features, feature_indices, max_depth - 1,
                               min_samples_split);
    node->right = dt_build_tree(ds, target_index, right_idx, n_right,
                                n_features, feature_indices, max_depth - 1,
                                min_samples_split);

    free(left_idx);
    free(right_idx);
    return node;
}

static int dt_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
                  const dataset* ds, const size_t* feature_indices, size_t n_features,
                  size_t target_index, const size_t* sample_indices, size_t n_samples) {
    (void)config;
    (void)n_features;

    DT_State* dt = (DT_State*)malloc(sizeof(DT_State));
    if (!dt) return -1;

    dt->n_classes = ds->labels[target_index].classes;
    dt->root = dt_build_tree(ds, target_index, sample_indices, n_samples,
                             n_features, feature_indices, 10, 2);

    state->weights = dt;
    state->size = sizeof(DT_State);
    return 0;
}

static int dt_predict_single(const DT_Node* node, const dataset* ds,
                             const size_t* feature_indices,
                             size_t sample_idx) {
    if (node->is_leaf) return node->class_label;

    double val = ds->features[node->feature_index].data[sample_idx];
    if (val <= node->threshold) {
        return dt_predict_single(node->left, ds, feature_indices, sample_idx);
    } else {
        return dt_predict_single(node->right, ds, feature_indices, sample_idx);
    }
}

static int dt_predict(const ML_Weights_t* state, const dataset* ds,
                      const size_t* feature_indices, size_t n_features,
                      const size_t* sample_indices, size_t n_samples,
                      void* output) {
    (void)n_features;
    DT_State* dt = (DT_State*)state->weights;
    int* predictions = (int*)output;

    for (size_t s = 0; s < n_samples; s++) {
        predictions[s] = dt_predict_single(dt->root, ds, feature_indices,
                                           sample_indices[s]);
    }
    return 0;
}

static void dt_free_node(DT_Node* node) {
    if (!node) return;
    dt_free_node(node->left);
    dt_free_node(node->right);
    free(node);
}

static void dt_free_state(ML_Weights_t* state) {
    if (state->weights) {
        DT_State* dt = (DT_State*)state->weights;
        dt_free_node(dt->root);
        free(dt);
        state->weights = NULL;
    }
}

static ML_Model_t create_dt_model(void) {
    return (ML_Model_t){
        .type = ML_CLASSIFICATION,
        .config = {NULL, 0},
        .state = {NULL, 0},
        .methods = {
            .fit = dt_fit,
            .predict = dt_predict,
            .predict_proba = NULL,
            .get_coefficients = NULL,
            .serialize = NULL,
            .deserialize = NULL,
            .free_state = dt_free_state
        }
    };
}
```

---

## 实现示例：AdaBoost（自适应提升）

### 算法原理

AdaBoost 通过组合多个弱分类器（通常是决策树桩）来构建强分类器。每个样本有权重，错误分类的样本权重会增加，最终预测是所有弱分类器的加权投票。

### 步骤分解

1. **初始化样本权重**：每个样本初始权重为 `1/n`
2. **对每个弱分类器**：
   - 使用加权样本训练弱分类器
   - 计算弱分类器的加权错误率
   - 计算弱分类器的权重 `alpha = 0.5 * log((1 - error) / error)`
   - 更新样本权重：被正确分类的权重减小，被错误分类的权重增大
3. **最终预测**：所有弱分类器的加权投票

### 实现代码

```c
/*
 * AdaBoost 分类器
 *
 * 使用决策树桩作为弱分类器
 */

#define C_MACHINE_LEARNING_IMPLEMENTATION
#include "machine_learning.h"
#include <math.h>
#include <stdlib.h>

/* 决策树桩结构 - 只有一个分裂节点 */
typedef struct {
    size_t feature_index;  /* 分裂特征 */
    double threshold;       /* 分裂阈值 */
    int left_class;        /* 左子树类别 */
    int right_class;       /* 右子树类别 */
} Stump;

/* AdaBoost 状态 */
typedef struct {
    Stump* stumps;         /* 弱分类器数组 */
    double* alphas;        /* 弱分类器权重 */
    size_t n_estimators;   /* 弱分类器数量 */
    size_t n_classes;
} AdaBoost_State;

/* 辅助函数：构建决策树桩 */
static Stump build_stump(const dataset* ds, size_t target_index,
                        const size_t* indices, size_t n,
                        const size_t* feature_indices, size_t n_features,
                        const double* weights) {
    Stump stump = {0};
    double best_weighted_error = 1e300;

    for (size_t f = 0; f < n_features; f++) {
        size_t feat_idx = feature_indices[f];

        /* 找特征值的最小/最大值 */
        double min_val = ds->features[feat_idx].data[indices[0]];
        double max_val = min_val;
        for (size_t i = 1; i < n; i++) {
            double v = ds->features[feat_idx].data[indices[i]];
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }

        /* 尝试不同阈值 */
        for (int t = 0; t < 10; t++) {
            double threshold = min_val + (max_val - min_val) * (t + 1) / 11.0;

            /* 计算加权错误率 */
            double weighted_error = 0.0;
            int n_left = 0, n_right = 0;
            int* left_labels = (int*)calloc(n, sizeof(int));
            int* right_labels = (int*)calloc(n, sizeof(int));
            double* left_weights = (double*)malloc(sizeof(double) * n);
            double* right_weights = (double*)malloc(sizeof(double) * n);

            for (size_t i = 0; i < n; i++) {
                double v = ds->features[feat_idx].data[indices[i]];
                int label = ds->labels[target_index].labels[indices[i]];
                if (v <= threshold) {
                    left_labels[n_left] = label;
                    left_weights[n_left++] = weights[i];
                } else {
                    right_labels[n_right] = label;
                    right_weights[n_right++] = weights[i];
                }
            }

            /* 多数表决确定左右子树类别 */
            int left_class = 0, right_class = 0;
            if (n_left > 0) {
                size_t* counts = (size_t*)calloc(256, sizeof(size_t));
                for (int i = 0; i < n_left; i++) counts[left_labels[i]]++;
                left_class = 0; for (int i = 1; i < 256; i++) if (counts[i] > counts[left_class]) left_class = i;
                free(counts);
            }
            if (n_right > 0) {
                size_t* counts = (size_t*)calloc(256, sizeof(size_t));
                for (int i = 0; i < n_right; i++) counts[right_labels[i]]++;
                right_class = 0; for (int i = 1; i < 256; i++) if (counts[i] > counts[right_class]) right_class = i;
                free(counts);
            }

            /* 计算加权错误率 */
            double err = 0.0;
            for (size_t i = 0; i < n; i++) {
                double v = ds->features[feat_idx].data[indices[i]];
                int pred = (v <= threshold) ? left_class : right_class;
                int actual = ds->labels[target_index].labels[indices[i]];
                if (pred != actual) err += weights[i];
            }

            if (err < best_weighted_error) {
                best_weighted_error = err;
                stump.feature_index = feat_idx;
                stump.threshold = threshold;
                stump.left_class = left_class;
                stump.right_class = right_class;
            }

            free(left_labels); free(right_labels);
            free(left_weights); free(right_weights);
        }
    }

    return stump;
}

/* 用决策树桩预测单个样本 */
static int stump_predict(const Stump* stump, const dataset* ds,
                        const size_t* feature_indices, size_t sample_idx) {
    (void)feature_indices;
    double v = ds->features[stump->feature_index].data[sample_idx];
    return (v <= stump->threshold) ? stump->left_class : stump->right_class;
}

static int adaboost_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
                        const dataset* ds, const size_t* feature_indices, size_t n_features,
                        size_t target_index, const size_t* sample_indices, size_t n_samples) {
    (void)config;
    size_t n_estimators = 50;  /* 弱分类器数量 */
    size_t n_classes = ds->labels[target_index].classes;

    AdaBoost_State* ab = (AdaBoost_State*)malloc(sizeof(AdaBoost_State));
    ab->stumps = (Stump*)malloc(sizeof(Stump) * n_estimators);
    ab->alphas = (double*)malloc(sizeof(double) * n_estimators);
    ab->n_estimators = n_estimators;
    ab->n_classes = n_classes;

    /* 初始化权重 */
    double* weights = (double*)malloc(sizeof(double) * n_samples);
    for (size_t i = 0; i < n_samples; i++) weights[i] = 1.0 / n_samples;

    /* 创建临时索引数组 */
    size_t* idx = (size_t*)malloc(sizeof(size_t) * n_samples);
    for (size_t i = 0; i < n_samples; i++) idx[i] = sample_indices[i];

    for (size_t t = 0; t < n_estimators; t++) {
        /* 归一化权重 */
        double weight_sum = 0.0;
        for (size_t i = 0; i < n_samples; i++) weight_sum += weights[i];
        for (size_t i = 0; i < n_samples; i++) weights[i] /= weight_sum;

        /* 训练弱分类器（决策树桩） */
        ab->stumps[t] = build_stump(ds, target_index, idx, n_samples,
                                     feature_indices, n_features, weights);

        /* 计算加权错误率 */
        double error = 0.0;
        for (size_t i = 0; i < n_samples; i++) {
            int pred = stump_predict(&ab->stumps[t], ds, feature_indices, sample_indices[i]);
            int actual = ds->labels[target_index].labels[sample_indices[i]];
            if (pred != actual) error += weights[i];
        }

        /* 防止错误率为0或1 */
        if (error < 1e-10) error = 1e-10;
        if (error > 1.0 - 1e-10) error = 1.0 - 1e-10;

        /* 计算弱分类器权重 */
        ab->alphas[t] = 0.5 * log((1.0 - error) / error);

        /* 更新样本权重 */
        for (size_t i = 0; i < n_samples; i++) {
            int pred = stump_predict(&ab->stumps[t], ds, feature_indices, sample_indices[i]);
            int actual = ds->labels[target_index].labels[sample_indices[i]];
            if (pred == actual) {
                weights[i] *= exp(-ab->alphas[t]);
            } else {
                weights[i] *= exp(ab->alphas[t]);
            }
        }
    }

    free(weights);
    free(idx);
    state->weights = ab;
    state->size = sizeof(AdaBoost_State);
    return 0;
}

static int adaboost_predict(const ML_Weights_t* state, const dataset* ds,
                            const size_t* feature_indices, size_t n_features,
                            const size_t* sample_indices, size_t n_samples,
                            void* output) {
    AdaBoost_State* ab = (AdaBoost_State*)state->weights;
    int* predictions = (int*)output;

    for (size_t s = 0; s < n_samples; s++) {
        size_t idx = sample_indices[s];
        double* class_scores = (double*)calloc(ab->n_classes, sizeof(double));

        /* 累加所有弱分类器的投票 */
        for (size_t t = 0; t < ab->n_estimators; t++) {
            int pred = stump_predict(&ab->stumps[t], ds, feature_indices, idx);
            class_scores[pred] += ab->alphas[t];
        }

        /* 选择得分最高的类别 */
        int best_class = 0;
        for (size_t c = 1; c < ab->n_classes; c++) {
            if (class_scores[c] > class_scores[best_class]) {
                best_class = (int)c;
            }
        }
        predictions[s] = best_class;
        free(class_scores);
    }
    return 0;
}

static void adaboost_free_state(ML_Weights_t* state) {
    if (state->weights) {
        AdaBoost_State* ab = (AdaBoost_State*)state->weights;
        free(ab->stumps);
        free(ab->alphas);
        free(ab);
        state->weights = NULL;
    }
}

static ML_Model_t create_adaboost_model(void) {
    return (ML_Model_t){
        .type = ML_CLASSIFICATION,
        .config = {NULL, 0},
        .state = {NULL, 0},
        .methods = {
            .fit = adaboost_fit,
            .predict = adaboost_predict,
            .predict_proba = NULL,
            .get_coefficients = NULL,
            .serialize = NULL,
            .deserialize = NULL,
            .free_state = adaboost_free_state
        }
    };
}
```

---

## 实现示例：随机森林

### 算法原理

随机森林是多个决策树的集成，每个树使用不同的样本子集（bagging）和特征子集训练。预测时，所有树的预测进行多数表决或平均。

### 与 AdaBoost 的区别

| 特性 | AdaBoost | 随机森林 |
|------|----------|----------|
| 树类型 | 浅树（树桩） | 完全展开的树 |
| 训练方式 | 串行（加权采样） | 并行（独立采样） |
| 投票方式 | 加权投票 | 简单多数表决 |

### 步骤分解

1. **对每棵树**：
   - 有放回抽样（bootstrap）生成训练样本
   - 随机选择部分特征进行分裂
   - 训练决策树
2. **最终预测**：多数表决（分类）或平均（回归）

### 实现代码

```c
/*
 * 随机森林分类器
 *
 * 使用决策树作为基学习器
 */

#define C_MACHINE_LEARNING_IMPLEMENTATION
#include "machine_learning.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

/* 决策树节点（简化版） */
typedef struct RF_Node {
    int is_leaf;
    int class_label;
    size_t feature_index;
    double threshold;
    struct RF_Node* left;
    struct RF_Node* right;
} RF_Node;

/* 单棵树状态 */
typedef struct {
    RF_Node* root;
    size_t n_classes;
} RF_Tree;

/* 随机森林状态 */
typedef struct {
    RF_Tree* trees;        /* 决策树数组 */
    size_t n_trees;        /* 树的数量 */
    size_t n_classes;
    size_t max_features;   /* 每棵树使用的特征数量 */
} RandomForest_State;

/* 熵计算 */
static double rf_entropy(int* labels, size_t n) {
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

/* 信息增益 */
static double rf_info_gain(const dataset* ds, size_t target_index,
                          size_t feature_idx, double threshold,
                          const size_t* indices, size_t n) {
    int* parent_labels = (int*)malloc(sizeof(int) * n);
    for (size_t i = 0; i < n; i++)
        parent_labels[i] = (int)ds->labels[target_index].labels[indices[i]];
    double parent_entropy = rf_entropy(parent_labels, n);
    free(parent_labels);

    size_t n_left = 0, n_right = 0;
    int* left_labels = (int*)malloc(sizeof(int) * n);
    int* right_labels = (int*)malloc(sizeof(int) * n);

    for (size_t i = 0; i < n; i++) {
        double val = ds->features[feature_idx].data[indices[i]];
        if (val <= threshold) {
            left_labels[n_left++] = (int)ds->labels[target_index].labels[indices[i]];
        } else {
            right_labels[n_right++] = (int)ds->labels[target_index].labels[indices[i]];
        }
    }

    double child_entropy = (n_left * rf_entropy(left_labels, n_left) +
                          n_right * rf_entropy(right_labels, n_right)) / n;
    free(left_labels); free(right_labels);
    return parent_entropy - child_entropy;
}

/* 构建决策树 */
static RF_Node* rf_build_tree(const dataset* ds, size_t target_index,
                              const size_t* indices, size_t n,
                              size_t n_features, const size_t* feature_indices,
                              size_t max_depth, size_t min_samples_split,
                              size_t max_features) {
    RF_Node* node = (RF_Node*)malloc(sizeof(RF_Node));
    node->is_leaf = 0; node->left = NULL; node->right = NULL;

    int first_label = (int)ds->labels[target_index].labels[indices[0]];
    int all_same = 1;
    for (size_t i = 1; i < n; i++)
        if ((int)ds->labels[target_index].labels[indices[i]] != first_label)
            { all_same = 0; break; }

    if (all_same || n < min_samples_split || max_depth == 0) {
        node->is_leaf = 1; node->class_label = first_label;
        return node;
    }

    /* 随机选择特征子集 */
    size_t* selected_features = (size_t*)malloc(sizeof(size_t) * max_features);
    for (size_t i = 0; i < max_features; i++) selected_features[i] = feature_indices[i];

    /* Fisher-Yates 随机打乱特征 */
    for (size_t i = max_features - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        size_t tmp = selected_features[i];
        selected_features[i] = selected_features[j];
        selected_features[j] = tmp;
    }

    double best_gain = -1.0;
    size_t best_feature = selected_features[0];
    double best_threshold = 0.0;

    for (size_t fi = 0; fi < max_features; fi++) {
        size_t feat_idx = selected_features[fi];
        double min_val = ds->features[feat_idx].data[indices[0]];
        double max_val = min_val;
        for (size_t i = 1; i < n; i++) {
            double v = ds->features[feat_idx].data[indices[i]];
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }

        for (int t = 1; t <= 5; t++) {
            double threshold = min_val + (max_val - min_val) * t / 6.0;
            double gain = rf_info_gain(ds, target_index, feat_idx, threshold, indices, n);
            if (gain > best_gain) { best_gain = gain; best_feature = feat_idx; best_threshold = threshold; }
        }
    }

    free(selected_features);

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

    node->left = rf_build_tree(ds, target_index, left_idx, n_left, n_features, feature_indices, max_depth - 1, min_samples_split, max_features);
    node->right = rf_build_tree(ds, target_index, right_idx, n_right, n_features, feature_indices, max_depth - 1, min_samples_split, max_features);
    free(left_idx); free(right_idx);
    return node;
}

/* 树桩预测 */
static int rf_predict_single(const RF_Node* node, const dataset* ds, size_t sample_idx) {
    if (node->is_leaf) return node->class_label;
    double val = ds->features[node->feature_index].data[sample_idx];
    return (val <= node->threshold) ? rf_predict_single(node->left, ds, sample_idx)
                                     : rf_predict_single(node->right, ds, sample_idx);
}

static void rf_free_node(RF_Node* node) {
    if (!node) return;
    rf_free_node(node->left); rf_free_node(node->right); free(node);
}

static int randomforest_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
                            const dataset* ds, const size_t* feature_indices, size_t n_features,
                            size_t target_index, const size_t* sample_indices, size_t n_samples) {
    (void)config;
    size_t n_trees = 100;  /* 树的数量 */
    size_t max_features = (size_t)sqrt(n_features);  /* 每次分裂使用的特征数 */
    if (max_features < 2) max_features = 2;

    RandomForest_State* rf = (RandomForest_State*)malloc(sizeof(RandomForest_State));
    rf->trees = (RF_Tree*)malloc(sizeof(RF_Tree) * n_trees);
    rf->n_trees = n_trees;
    rf->n_classes = ds->labels[target_index].classes;
    rf->max_features = max_features;

    /* 有放回抽样（bootstrap）生成每棵树的训练数据 */
    for (size_t t = 0; t < n_trees; t++) {
        size_t* bag_indices = (size_t*)malloc(sizeof(size_t) * n_samples);
        for (size_t i = 0; i < n_samples; i++) {
            bag_indices[i] = sample_indices[rand() % n_samples];  /* 有放回抽样 */
        }

        rf->trees[t].n_classes = rf->n_classes;
        rf->trees[t].root = rf_build_tree(ds, target_index, bag_indices, n_samples,
                                          n_features, feature_indices, 20, 2, max_features);
        free(bag_indices);
    }

    state->weights = rf;
    state->size = sizeof(RandomForest_State);
    return 0;
}

static int randomforest_predict(const ML_Weights_t* state, const dataset* ds,
                               const size_t* feature_indices, size_t n_features,
                               const size_t* sample_indices, size_t n_samples,
                               void* output) {
    (void)feature_indices; (void)n_features;
    RandomForest_State* rf = (RandomForest_State*)state->weights;
    int* predictions = (int*)output;

    for (size_t s = 0; s < n_samples; s++) {
        size_t idx = sample_indices[s];
        size_t* votes = (size_t*)calloc(rf->n_classes, sizeof(size_t));

        /* 每棵树投票 */
        for (size_t t = 0; t < rf->n_trees; t++) {
            int pred = rf_predict_single(rf->trees[t].root, ds, idx);
            votes[pred]++;
        }

        /* 多数表决 */
        size_t best_vote = 0;
        for (size_t c = 1; c < rf->n_classes; c++) {
            if (votes[c] > votes[best_vote]) best_vote = c;
        }
        predictions[s] = (int)best_vote;
        free(votes);
    }
    return 0;
}

static void randomforest_free_state(ML_Weights_t* state) {
    if (state->weights) {
        RandomForest_State* rf = (RandomForest_State*)state->weights;
        for (size_t t = 0; t < rf->n_trees; t++) {
            rf_free_node(rf->trees[t].root);
        }
        free(rf->trees);
        free(rf);
        state->weights = NULL;
    }
}

static ML_Model_t create_randomforest_model(void) {
    srand((unsigned int)time(NULL));  /* 初始化随机种子 */
    return (ML_Model_t){
        .type = ML_CLASSIFICATION,
        .config = {NULL, 0},
        .state = {NULL, 0},
        .methods = {
            .fit = randomforest_fit,
            .predict = randomforest_predict,
            .predict_proba = NULL,
            .get_coefficients = NULL,
            .serialize = NULL,
            .deserialize = NULL,
            .free_state = randomforest_free_state
        }
    };
}
```

---

## 完整示例

```c
#define CSV_IMPLEMENTATION
#include "csv.h"

#define C_MACHINE_LEARNING_IMPLEMENTATION
#include "machine_learning.h"

#include <stdio.h>

int main(void) {
    // 加载数据
    csv_t* csv = csv_load("iris.csv");
    const char* labels[] = {"species"};
    dataset* ds = csv_to_dataset(csv, labels, 1);

    // 配置
    const size_t feat_idx[] = {0, 1, 2, 3};
    size_t n_feat = 4;

    // 创建 GNB 模型
    ML_Model_t gnb = create_gnb_model();

    // 训练
    size_t indices[150];
    for (int i = 0; i < 150; i++) indices[i] = i;
    gnb.methods.fit(&gnb.config, &gnb.state, ds, feat_idx, n_feat, 0, indices, 150);

    // 预测
    int preds[150];
    gnb.methods.predict(&gnb.state, ds, feat_idx, n_feat, indices, 150, preds);

    // 计算准确率
    size_t correct = 0;
    for (int i = 0; i < 150; i++) {
        if (preds[i] == ds->labels[0].labels[i]) correct++;
    }
    printf("GNB Accuracy: %.2f%%\n", 100.0 * correct / 150);

    // 释放
    gnb.methods.free_state(&gnb.state);
    free_dataset(ds);
    free_csv_data(csv);
    free(csv);

    return 0;
}
```

---

## API 参考

### 函数

| 函数 | 说明 |
|------|------|
| `train_test_split` | 分割训练/测试集 |
| `train_model_with_validation` | 带验证的训练 |
| `kfold_cross_validate` | K 折交叉验证 |
| `model_evaluate` | 评估模型性能 |
| `ml_fit_scaling` | 拟合缩放参数 |
| `ml_transform_features` | 应用特征缩放 |
| `ml_free_scaling_params` | 释放缩放参数 |

### 枚举

| 枚举 | 值 | 说明 |
|------|-----|------|
| `ML_REGRESSION` | 0x1 | 回归任务 |
| `ML_CLASSIFICATION` | 0x2 | 分类任务 |
| `SCALING_STANDARD` | 1 | 标准化 |
| `SCALING_MINMAX` | 2 | MinMax 缩放 |
| `SCALING_MAXABS` | 3 | MaxAbs 缩放 |

### 方法检查

```c
typedef enum {
    ML_METHOD_FIT = 0,
    ML_METHOD_PREDICT,
    ML_METHOD_PREDICT_PROBA,
    ML_METHOD_GET_COEFFICIENTS,
    ML_METHOD_SERIALIZE,
    ML_METHOD_DESERIALIZE,
    ML_METHOD_FREE_STATE,
    ML_METHOD_COUNT
} ML_MethodType_t;

bool ml_method_is_implemented(const ML_Model_Impl_t* methods,
                              ML_MethodType_t method);

unsigned int ml_methods_get_implemented(const ML_Model_Impl_t* methods);
```
