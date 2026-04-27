# 随机森林与 AdaBoost 实战教程

本教程将指导你使用 `machine_learning.h` 接口，从零开始实现随机森林和 AdaBoost 算法。

---

## 第一阶段：准备工作

### 1.1 理解接口

在开始实现之前，先理解 `ML_Model_Impl_t` 接口：

```c
struct ML_Model_Impl_t {
    // 训练模型
    int (*fit)(const ML_Model_Config_t* config, ML_Weights_t* state,
               const dataset* ds, const size_t* feature_indices, size_t n_features,
               size_t target_index, const size_t* sample_indices, size_t n_samples);

    // 预测类别
    int (*predict)(const ML_Weights_t* state, const dataset* ds,
                   const size_t* feature_indices, size_t n_features,
                   const size_t* sample_indices, size_t n_samples, void* output);

    // 预测概率（可选）
    int (*predict_proba)(...);

    // 获取系数（可选）
    int (*get_coefficients)(...);

    // 序列化/反序列化（可选）
    int (*serialize)(...);
    int (*deserialize)(...);

    // 释放状态
    void (*free_state)(ML_Weights_t* state);
};
```

**关键点**：
- `state->weights` 用于存储模型参数
- `dataset` 包含 `features`（数值特征）和 `labels`（类别标签）
- `sample_indices` 是样本索引数组，`n_samples` 是样本数量

### 1.2 创建模型骨架

```c
#define C_MACHINE_LEARNING_IMPLEMENTATION
#include "machine_learning.h"
#include <stdlib.h>

/* 模型状态结构 */
typedef struct {
    /* 你需要存储的参数 */
} MyModel_State;

/* 训练函数 */
static int mymodel_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
                       const dataset* ds, const size_t* feature_indices, size_t n_features,
                       size_t target_index, const size_t* sample_indices, size_t n_samples) {
    /* 实现训练逻辑 */
    return 0;
}

/* 预测函数 */
static int mymodel_predict(const ML_Weights_t* state, const dataset* ds,
                          const size_t* feature_indices, size_t n_features,
                          const size_t* sample_indices, size_t n_samples, void* output) {
    /* 实现预测逻辑 */
    return 0;
}

/* 释放函数 */
static void mymodel_free(ML_Weights_t* state) {
    /* 释放你分配的内存 */
}

/* 创建模型的工厂函数 */
static ML_Model_t create_mymodel(void) {
    return (ML_Model_t){
        .type = ML_CLASSIFICATION,  // 或 ML_REGRESSION
        .config = {NULL, 0},
        .state = {NULL, 0},
        .methods = {
            .fit = mymodel_fit,
            .predict = mymodel_predict,
            .predict_proba = NULL,
            .get_coefficients = NULL,
            .serialize = NULL,
            .deserialize = NULL,
            .free_state = mymodel_free
        }
    };
}
```

---

## 第二阶段：决策树（基础组件）

随机森林和 AdaBoost 都依赖决策树作为基学习器。我们先实现一个简化版决策树。

### 2.1 定义树节点结构

```c
typedef struct TreeNode {
    int is_leaf;              // 是否是叶子节点
    int class_label;          // 叶子节点的类别
    size_t feature_index;     // 分裂特征索引
    double threshold;         // 分裂阈值
    struct TreeNode* left;    // 左子树 (<= threshold)
    struct TreeNode* right;   // 右子树 (> threshold)
} TreeNode;

typedef struct {
    TreeNode* root;
    size_t n_classes;
} DecisionTree_State;
```

### 2.2 计算熵

```c
static double entropy(int* labels, size_t n) {
    if (n == 0) return 0.0;

    // 统计每个类别的数量
    size_t* counts = (size_t*)calloc(256, sizeof(size_t));
    for (size_t i = 0; i < n; i++) counts[labels[i]]++;

    // 计算熵
    double ent = 0.0;
    for (size_t i = 0; i < 256 && counts[i] > 0; i++) {
        double p = (double)counts[i] / n;
        if (p > 0) ent -= p * log(p);
    }
    free(counts);
    return ent;
}
```

### 2.3 计算信息增益

```c
static double information_gain(const dataset* ds, size_t target_index,
                              size_t feature_idx, double threshold,
                              const size_t* indices, size_t n) {
    // 父节点熵
    int* parent_labels = (int*)malloc(sizeof(int) * n);
    for (size_t i = 0; i < n; i++)
        parent_labels[i] = (int)ds->labels[target_index].labels[indices[i]];
    double parent_ent = entropy(parent_labels, n);
    free(parent_labels);

    // 分裂样本
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

    // 子节点加权熵
    double child_ent = (n_left * entropy(left_labels, n_left) +
                       n_right * entropy(right_labels, n_right)) / n;
    free(left_labels);
    free(right_labels);

    return parent_ent - child_ent;
}
```

### 2.4 递归构建树

```c
static TreeNode* build_tree(const dataset* ds, size_t target_index,
                          const size_t* indices, size_t n,
                          size_t n_features, const size_t* feature_indices,
                          size_t max_depth, size_t min_samples) {
    TreeNode* node = (TreeNode*)malloc(sizeof(TreeNode));
    node->is_leaf = 0;
    node->left = NULL;
    node->right = NULL;

    // 终止条件：所有样本同类或达到限制
    int first_label = (int)ds->labels[target_index].labels[indices[0]];
    int all_same = 1;
    for (size_t i = 1; i < n; i++)
        if ((int)ds->labels[target_index].labels[indices[i]] != first_label)
            { all_same = 0; break; }

    if (all_same || n < min_samples || max_depth == 0) {
        node->is_leaf = 1;
        node->class_label = first_label;
        return node;
    }

    // 找最佳分裂
    double best_gain = -1.0;
    size_t best_feature = 0;
    double best_threshold = 0.0;

    for (size_t f = 0; f < n_features; f++) {
        size_t feat_idx = feature_indices[f];

        // 找特征值的范围
        double min_val = ds->features[feat_idx].data[indices[0]];
        double max_val = min_val;
        for (size_t i = 1; i < n; i++) {
            double v = ds->features[feat_idx].data[indices[i]];
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }

        // 尝试多个阈值
        for (int t = 1; t <= 5; t++) {
            double threshold = min_val + (max_val - min_val) * t / 6.0;
            double gain = information_gain(ds, target_index, feat_idx, threshold, indices, n);
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

    // 执行分裂
    node->feature_index = best_feature;
    node->threshold = best_threshold;

    size_t* left_idx = (size_t*)malloc(sizeof(size_t) * n);
    size_t* right_idx = (size_t*)malloc(sizeof(size_t) * n);
    size_t n_left = 0, n_right = 0;

    for (size_t i = 0; i < n; i++) {
        double val = ds->features[best_feature].data[indices[i]];
        if (val <= best_threshold)
            left_idx[n_left++] = indices[i];
        else
            right_idx[n_right++] = indices[i];
    }

    node->left = build_tree(ds, target_index, left_idx, n_left,
                           n_features, feature_indices, max_depth - 1, min_samples);
    node->right = build_tree(ds, target_index, right_idx, n_right,
                            n_features, feature_indices, max_depth - 1, min_samples);

    free(left_idx);
    free(right_idx);
    return node;
}
```

### 2.5 预测函数

```c
static int tree_predict_single(const TreeNode* node, const dataset* ds, size_t sample_idx) {
    if (node->is_leaf) return node->class_label;

    double val = ds->features[node->feature_index].data[sample_idx];
    if (val <= node->threshold)
        return tree_predict_single(node->left, ds, sample_idx);
    else
        return tree_predict_single(node->right, ds, sample_idx);
}

static int tree_predict(const ML_Weights_t* state, const dataset* ds,
                       const size_t* feature_indices, size_t n_features,
                       const size_t* sample_indices, size_t n_samples, void* output) {
    (void)feature_indices;
    (void)n_features;
    DecisionTree_State* dt = (DecisionTree_State*)state->weights;
    int* predictions = (int*)output;

    for (size_t s = 0; s < n_samples; s++)
        predictions[s] = tree_predict_single(dt->root, ds, sample_indices[s]);

    return 0;
}
```

---

## 第三阶段：AdaBoost 实现

AdaBoost 的核心思想是：对错误分类的样本提高权重，串行训练多个弱分类器。

### 3.1 AdaBoost 状态结构

```c
typedef struct {
    TreeNode** stumps;    // 弱分类器数组（决策树桩）
    double* alphas;        // 每个弱分类器的权重
    size_t n_estimators;  // 弱分类器数量
    size_t n_classes;
} AdaBoost_State;
```

### 3.2 构建决策树桩

决策树桩是只分裂一次的决策树：

```c
static TreeNode* build_stump(const dataset* ds, size_t target_index,
                            const size_t* indices, size_t n,
                            size_t n_features, const size_t* feature_indices,
                            const double* weights) {
    TreeNode* node = (TreeNode*)malloc(sizeof(TreeNode));
    node->is_leaf = 0;
    node->left = NULL;
    node->right = NULL;

    // 找加权错误率最低的分裂
    double best_weighted_error = 1e300;
    int best_left_class = 0, best_right_class = 0;
    size_t best_feature = 0;
    double best_threshold = 0.0;

    for (size_t f = 0; f < n_features; f++) {
        size_t feat_idx = feature_indices[f];

        double min_val = ds->features[feat_idx].data[indices[0]];
        double max_val = min_val;
        for (size_t i = 1; i < n; i++) {
            double v = ds->features[feat_idx].data[indices[i]];
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }

        for (int t = 1; t <= 10; t++) {
            double threshold = min_val + (max_val - min_val) * t / 11.0;

            // 统计左右子树的类别
            size_t left_counts[256] = {0}, right_counts[256] = {0};
            size_t n_left = 0, n_right = 0;

            for (size_t i = 0; i < n; i++) {
                double v = ds->features[feat_idx].data[indices[i]];
                int label = ds->labels[target_index].labels[indices[i]];
                if (v <= threshold) {
                    left_counts[label]++;
                    n_left++;
                } else {
                    right_counts[label]++;
                    n_right++;
                }
            }

            // 多数表决
            int left_class = 0, right_class = 0;
            for (int i = 1; i < 256; i++) {
                if (left_counts[i] > left_counts[left_class]) left_class = i;
                if (right_counts[i] > right_counts[right_class]) right_class = i;
            }

            // 计算加权错误率
            double weighted_error = 0.0;
            for (size_t i = 0; i < n; i++) {
                double v = ds->features[feat_idx].data[indices[i]];
                int label = ds->labels[target_index].labels[indices[i]];
                int pred = (v <= threshold) ? left_class : right_class;
                if (pred != label) weighted_error += weights[i];
            }

            if (weighted_error < best_weighted_error) {
                best_weighted_error = weighted_error;
                best_feature = feat_idx;
                best_threshold = threshold;
                best_left_class = left_class;
                best_right_class = right_class;
            }
        }
    }

    node->is_leaf = 0;
    node->feature_index = best_feature;
    node->threshold = best_threshold;

    // 创建叶子节点
    TreeNode* left_leaf = (TreeNode*)malloc(sizeof(TreeNode));
    left_leaf->is_leaf = 1;
    left_leaf->class_label = best_left_class;
    left_leaf->left = NULL;
    left_leaf->right = NULL;

    TreeNode* right_leaf = (TreeNode*)malloc(sizeof(TreeNode));
    right_leaf->is_leaf = 1;
    right_leaf->class_label = best_right_class;
    right_leaf->left = NULL;
    right_leaf->right = NULL;

    node->left = left_leaf;
    node->right = right_leaf;

    return node;
}
```

### 3.3 AdaBoost 训练

```c
static int adaboost_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
                        const dataset* ds, const size_t* feature_indices, size_t n_features,
                        size_t target_index, const size_t* sample_indices, size_t n_samples) {
    (void)config;

    size_t n_estimators = 50;  // 弱分类器数量

    AdaBoost_State* adaboost = (AdaBoost_State*)malloc(sizeof(AdaBoost_State));
    adaboost->stumps = (TreeNode**)malloc(sizeof(TreeNode*) * n_estimators);
    adaboost->alphas = (double*)malloc(sizeof(double) * n_estimators);
    adaboost->n_estimators = n_estimators;
    adaboost->n_classes = ds->labels[target_index].classes;

    // 初始化样本权重
    double* weights = (double*)malloc(sizeof(double) * n_samples);
    for (size_t i = 0; i < n_samples; i++) weights[i] = 1.0 / n_samples;

    // 串行训练每个弱分类器
    for (size_t t = 0; t < n_estimators; t++) {
        // 归一化权重
        double weight_sum = 0.0;
        for (size_t i = 0; i < n_samples; i++) weight_sum += weights[i];
        for (size_t i = 0; i < n_samples; i++) weights[i] /= weight_sum;

        // 训练弱分类器
        adaboost->stumps[t] = build_stump(ds, target_index, sample_indices, n_samples,
                                          n_features, feature_indices, weights);

        // 计算加权错误率
        double error = 0.0;
        for (size_t i = 0; i < n_samples; i++) {
            int pred = tree_predict_single(adaboost->stumps[t], ds, sample_indices[i]);
            int actual = ds->labels[target_index].labels[sample_indices[i]];
            if (pred != actual) error += weights[i];
        }

        // 防止错误率为0或1
        if (error < 1e-10) error = 1e-10;
        if (error > 1.0 - 1e-10) error = 1.0 - 1e-10;

        // 计算弱分类器权重
        adaboost->alphas[t] = 0.5 * log((1.0 - error) / error);

        // 更新样本权重
        for (size_t i = 0; i < n_samples; i++) {
            int pred = tree_predict_single(adaboost->stumps[t], ds, sample_indices[i]);
            int actual = ds->labels[target_index].labels[sample_indices[i]];
            if (pred == actual)
                weights[i] *= exp(-adaboost->alphas[t]);
            else
                weights[i] *= exp(adaboost->alphas[t]);
        }
    }

    free(weights);
    state->weights = adaboost;
    state->size = sizeof(AdaBoost_State);
    return 0;
}
```

### 3.4 AdaBoost 预测

```c
static int adaboost_predict(const ML_Weights_t* state, const dataset* ds,
                            const size_t* feature_indices, size_t n_features,
                            const size_t* sample_indices, size_t n_samples,
                            void* output) {
    AdaBoost_State* adaboost = (AdaBoost_State*)state->weights;
    int* predictions = (int*)output;

    for (size_t s = 0; s < n_samples; s++) {
        size_t idx = sample_indices[s];
        double* class_scores = (double*)calloc(adaboost->n_classes, sizeof(double));

        // 累加所有弱分类器的加权投票
        for (size_t t = 0; t < adaboost->n_estimators; t++) {
            int pred = tree_predict_single(adaboost->stumps[t], ds, idx);
            class_scores[pred] += adaboost->alphas[t];
        }

        // 选择得分最高的类别
        int best_class = 0;
        for (size_t c = 1; c < adaboost->n_classes; c++)
            if (class_scores[c] > class_scores[best_class]) best_class = (int)c;

        predictions[s] = best_class;
        free(class_scores);
    }
    return 0;
}
```

### 3.5 释放资源

```c
static void adaboost_free(ML_Weights_t* state) {
    if (state->weights) {
        AdaBoost_State* adaboost = (AdaBoost_State*)state->weights;
        for (size_t t = 0; t < adaboost->n_estimators; t++) {
            // 释放每个树桩
            free(adaboost->stumps[t]->left);
            free(adaboost->stumps[t]->right);
            free(adaboost->stumps[t]);
        }
        free(adaboost->stumps);
        free(adaboost->alphas);
        free(adaboost);
        state->weights = NULL;
    }
}
```

---

## 第四阶段：随机森林实现

随机森林是多个决策树的并行集成，每棵树使用 bootstrap 抽样和随机特征子集。

### 4.1 随机森林状态

```c
typedef struct {
    TreeNode** trees;      // 决策树数组
    size_t n_trees;        // 树的数量
    size_t n_classes;
    size_t max_features;    // 每棵树使用的特征数
} RandomForest_State;
```

### 4.2 Bootstrap 抽样

```c
static size_t* bootstrap_sample(size_t n, unsigned int seed) {
    size_t* indices = (size_t*)malloc(sizeof(size_t) * n);
    srand(seed);
    for (size_t i = 0; i < n; i++)
        indices[i] = (size_t)(rand() % n);  // 有放回抽样
    return indices;
}
```

### 4.3 随机选择特征

```c
static void shuffle_array(size_t* arr, size_t n) {
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        size_t tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}
```

### 4.4 训练随机森林

```c
static int randomforest_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
                            const dataset* ds, const size_t* feature_indices, size_t n_features,
                            size_t target_index, const size_t* sample_indices, size_t n_samples) {
    (void)config;

    size_t n_trees = 100;  // 树的数量
    size_t max_features = (size_t)sqrt(n_features);  // 随机特征子集大小
    if (max_features < 2) max_features = 2;

    RandomForest_State* rf = (RandomForest_State*)malloc(sizeof(RandomForest_State));
    rf->trees = (TreeNode**)malloc(sizeof(TreeNode*) * n_trees);
    rf->n_trees = n_trees;
    rf->n_classes = ds->labels[target_index].classes;
    rf->max_features = max_features;

    // 为每棵树准备随机特征子集
    size_t* selected_features = (size_t*)malloc(sizeof(size_t) * max_features);
    for (size_t i = 0; i < max_features; i++) selected_features[i] = feature_indices[i];

    // 训练每棵树
    for (size_t t = 0; t < n_trees; t++) {
        // Bootstrap 抽样
        size_t* bag = bootstrap_sample(n_samples, (unsigned int)(t * 12345 + t));

        // 随机打乱特征选择
        shuffle_array(selected_features, max_features);

        // 构建树
        rf->trees[t] = build_tree(ds, target_index, bag, n_samples,
                                  max_features, selected_features, 20, 2);

        free(bag);
    }

    free(selected_features);
    state->weights = rf;
    state->size = sizeof(RandomForest_State);
    return 0;
}
```

### 4.5 预测（多数表决）

```c
static int randomforest_predict(const ML_Weights_t* state, const dataset* ds,
                               const size_t* feature_indices, size_t n_features,
                               const size_t* sample_indices, size_t n_samples,
                               void* output) {
    (void)feature_indices;
    (void)n_features;
    RandomForest_State* rf = (RandomForest_State*)state->weights;
    int* predictions = (int*)output;

    for (size_t s = 0; s < n_samples; s++) {
        size_t idx = sample_indices[s];
        size_t* votes = (size_t*)calloc(rf->n_classes, sizeof(size_t));

        // 每棵树投票
        for (size_t t = 0; t < rf->n_trees; t++) {
            int pred = tree_predict_single(rf->trees[t], ds, idx);
            votes[pred]++;
        }

        // 多数表决
        size_t best_vote = 0;
        for (size_t c = 1; c < rf->n_classes; c++)
            if (votes[c] > votes[best_vote]) best_vote = c;

        predictions[s] = (int)best_vote;
        free(votes);
    }
    return 0;
}
```

### 4.6 释放资源

```c
static void randomforest_free(ML_Weights_t* state) {
    if (state->weights) {
        RandomForest_State* rf = (RandomForest_State*)state->weights;
        for (size_t t = 0; t < rf->n_trees; t++) {
            // 递归释放树
            void free_tree(TreeNode* node) {
                if (!node) return;
                free_tree(node->left);
                free_tree(node->right);
                free(node);
            }
            free_tree(rf->trees[t]);
        }
        free(rf->trees);
        free(rf);
        state->weights = NULL;
    }
}
```

---

## 第五阶段：完整使用示例

### 5.1 模型工厂函数

```c
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
            .free_state = adaboost_free
        }
    };
}

static ML_Model_t create_randomforest_model(void) {
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
            .free_state = randomforest_free
        }
    };
}
```

### 5.2 完整示例代码

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

    const size_t feat_idx[] = {0, 1, 2, 3};
    size_t indices[150];
    for (int i = 0; i < 150; i++) indices[i] = i;

    // 测试 AdaBoost
    printf("=== AdaBoost ===\n");
    ML_Model_t adaboost = create_adaboost_model();
    adaboost.methods.fit(&adaboost.config, &adaboost.state, ds, feat_idx, 4, 0, indices, 150);

    int preds[150];
    adaboost.methods.predict(&adaboost.state, ds, feat_idx, 4, indices, 150, preds);

    size_t correct = 0;
    for (int i = 0; i < 150; i++)
        if (preds[i] == ds->labels[0].labels[i]) correct++;
    printf("Accuracy: %.2f%%\n", 100.0 * correct / 150);
    adaboost.methods.free_state(&adaboost.state);

    // 测试随机森林
    printf("\n=== Random Forest ===\n");
    ML_Model_t rf = create_randomforest_model();
    rf.methods.fit(&rf.config, &rf.state, ds, feat_idx, 4, 0, indices, 150);
    rf.methods.predict(&rf.state, ds, feat_idx, 4, indices, 150, preds);

    correct = 0;
    for (int i = 0; i < 150; i++)
        if (preds[i] == ds->labels[0].labels[i]) correct++;
    printf("Accuracy: %.2f%%\n", 100.0 * correct / 150);
    rf.methods.free_state(&rf.state);

    // 清理
    free_dataset(ds);
    free_csv_data(csv);
    free(csv);
    return 0;
}
```

---

## 第六阶段：关键对比

| 特性 | AdaBoost | 随机森林 |
|------|----------|----------|
| **树结构** | 决策树桩（浅树） | 完全展开的决策树 |
| **训练方式** | 串行（加权采样） | 并行（独立采样） |
| **样本权重** | 动态调整 | 均匀（bootstrap） |
| **特征选择** | 全部特征 | 随机特征子集 |
| **预测方式** | 加权投票 | 多数表决 |
| **防止过拟合** | 通过权重调整 | 通过集成和随机性 |

---

## 练习题

1. **修改树桩数量**：尝试不同的 `n_estimators` 值，观察对准确率的影响

2. **添加预测概率**：为 AdaBoost 实现 `predict_proba` 方法

3. **序列化模型**：为随机森林实现 `serialize` 和 `deserialize` 方法

4. **回归版本**：将随机森林改为回归模型（用方差代替熵作为分裂准则）
