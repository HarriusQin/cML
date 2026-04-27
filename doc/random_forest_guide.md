# 随机森林 (Random Forest) 实现指南

## 目录

1. [算法原理](#算法原理)
2. [核心概念](#核心概念)
3. [数据结构设计](#数据结构设计)
4. [关键算法实现](#关键算法实现)
5. [代码解析](#代码解析)
6. [使用示例](#使用示例)
7. [参数调优建议](#参数调优建议)

---

## 算法原理

随机森林是一种**集成学习**方法，通过构建多棵决策树并集成它们的预测结果来提高模型的准确性和稳定性。

### 核心思想

```
随机森林 = 随机 + 森林

"随机"：
  1. 样本随机 - Bootstrap有放回抽样
  2. 特征随机 - 每次分裂只考虑部分特征

"森林"：
  多棵决策树的集成
```

### 预测方式

| 任务类型 | 集成方法 |
|---------|---------|
| **分类** | 多数投票 (Majority Voting) |
| **回归** | 平均值 (Average Prediction) |

---

## 核心概念

### 1. Bootstrap抽样

从原始数据集有放回地抽取n个样本，构建每棵树的训练数据。

```
原始样本: [1, 2, 3, 4, 5]

Bootstrap样本1: [2, 1, 5, 2, 3]  (有重复)
Bootstrap样本2: [4, 3, 1, 1, 5]  (有重复)
Bootstrap样本3: [5, 2, 4, 3, 1]  (有重复)
```

### 2. 特征随机选择

- **分类**：通常选择 `√n_features` 个特征
- **回归**：通常选择 `n_features / 3` 个特征

### 3. 决策树分裂准则

#### 分类 - 基尼不纯度 (Gini Impurity)

```
Gini(D) = 1 - Σ p_i²

其中 p_i 是类别i在数据集D中的比例

Gini(D) 越小，数据集越"纯"
```

#### 回归 - 方差减少 (Variance Reduction)

```
Var(D) = Σ (x_i - μ)² / n

方差减少 = Var(父节点) - [n左/n * Var(左) + n右/n * Var(右)]

方差减少越大，分裂越好
```

### 4. 信息增益 (Information Gain)

```
Gain = Gini(父节点) - (n左/n) * Gini(左节点) - (n右/n) * Gini(右节点)

选择使信息增益最大的特征和阈值进行分裂
```

---

## 数据结构设计

### 决策树节点结构

```c
typedef struct dt_node {
    bool is_leaf;              // 是否为叶节点
    size_t feature_index;      // 分裂特征索引
    double threshold;          // 分裂阈值
    struct dt_node* left;      // 左子节点 (<=threshold)
    struct dt_node* right;     // 右子节点 (>threshold)
    int class_label;           // 类别标签 (分类用)
    double value;              // 节点值 (回归用)
    size_t samples;           // 样本数量
    double impurity;          // 不纯度
} dt_node;
```

### 随机森林数据结构

```c
typedef struct {
    dt_node** trees;           // 决策树数组
    size_t n_estimators;       // 树的数量
    size_t max_depth;          // 树的最大深度
    size_t max_features;       // 每棵树使用的特征数量
    double subsample_ratio;    // 子采样比例
    size_t n_classes;          // 类别数 (分类用)
    bool is_regression;       // 是否为回归模型
} rf_data;
```

### 通用模型句柄

```c
typedef struct model {
    model_type_t type;         // 模型类型
    void* data;                // 内部数据指针
} model;
```

---

## 关键算法实现

### 1. 构建单棵决策树

```c
/**
 * 递归构建决策树
 *
 * @param X              特征矩阵 [n_samples * n_features]
 * @param y              类别标签 [n_samples]
 * @param y_reg          回归值 [n_samples] (回归树用)
 * @param n              样本数量
 * @param n_features     特征数量
 * @param max_features   最大特征数
 * @param max_depth      最大深度
 * @param min_samples_split  分裂所需最小样本数
 * @param min_samples_leaf   叶节点最小样本数
 * @param current_depth  当前深度
 * @param is_regression  是否为回归树
 */
dt_node* build_tree(double* X, int* y, double* y_reg, size_t n,
                    size_t n_features, size_t max_features,
                    size_t max_depth, size_t min_samples_split,
                    size_t min_samples_leaf, size_t current_depth,
                    bool is_regression)
```

**停止条件**：
1. 样本数量小于 `min_samples_split`
2. 达到最大深度 `max_depth`
3. 所有样本属于同一类别（分类）
4. 所有样本值相同（回归）

### 2. 找最佳分裂点

```c
/**
 * 在所有特征中找最佳分裂点
 *
 * @param X            特征矩阵
 * @param y            标签
 * @param n            样本数
 * @param n_features   特征数
 * @param max_features 最大尝试特征数
 * @param best_feat    输出：最佳特征索引
 * @param is_regression 是否为回归
 *
 * @return 最佳分裂阈值
 */
double find_best_split(double* X, int* y, size_t n, size_t n_features,
                      size_t max_features, size_t* best_feat,
                      bool is_regression)
```

**算法步骤**：
1. 遍历随机选择的特征子集
2. 对每个特征，找到最小值和最大值
3. 在特征值范围内等距取10个阈值
4. 对每个阈值计算基尼不纯度/方差减少
5. 选择最优的特征和阈值

### 3. 计算基尼不纯度

```c
/**
 * 计算基尼不纯度
 *
 * Gini(D) = 1 - Σ p_i²
 *
 * @param labels  标签数组
 * @param n       样本数量
 * @return 基尼不纯度值
 */
static double gini(int* labels, size_t n) {
    if (n == 0) return 0.0;

    // 统计每个类别的数量
    size_t counts[256] = {0};
    for (size_t i = 0; i < n; i++)
        counts[labels[i]]++;

    // 计算基尼不纯度
    double gini = 1.0;
    for (size_t c = 0; c < 256; c++) {
        if (counts[c] > 0) {
            double p = (double)counts[c] / n;
            gini -= p * p;
        }
    }
    return gini;
}
```

### 4. Bootstrap采样

```c
/**
 * 执行Bootstrap有放回采样
 *
 * @param X_orig     原始特征矩阵
 * @param y_orig     原始标签
 * @param n_orig     原始样本数
 * @param n_features 特征数
 * @param n_boot     Bootstrap样本数
 * @param X_boot     输出：Bootstrap特征矩阵
 * @param y_boot     输出：Bootstrap标签
 */
void bootstrap_sample(double* X_orig, int* y_orig, size_t n_orig,
                      size_t n_features, size_t n_boot,
                      double* X_boot, int* y_boot) {
    for (size_t i = 0; i < n_boot; i++) {
        size_t idx = rand() % n_orig;  // 有放回随机选择
        for (size_t f = 0; f < n_features; f++)
            X_boot[i * n_features + f] = X_orig[idx * n_features + f];
        y_boot[i] = y_orig[idx];
    }
}
```

### 5. 随机森林预测

```c
/**
 * 随机森林分类预测 - 多数投票
 */
int predict_random_forest_classification(rf_data* rf, const double* x) {
    size_t* votes = calloc(rf->n_classes, sizeof(size_t));

    // 每棵树投票
    for (size_t t = 0; t < rf->n_estimators; t++) {
        int pred = predict_tree(rf->trees[t], x);
        votes[pred]++;
    }

    // 找最多票数
    int best_class = 0;
    size_t max_votes = 0;
    for (size_t c = 0; c < rf->n_classes; c++) {
        if (votes[c] > max_votes) {
            max_votes = votes[c];
            best_class = c;
        }
    }
    free(votes);
    return best_class;
}

/**
 * 随机森林回归预测 - 平均值
 */
double predict_random_forest_regression(rf_data* rf, const double* x) {
    double sum = 0.0;
    size_t valid_trees = 0;

    for (size_t t = 0; t < rf->n_estimators; t++) {
        if (rf->trees[t]) {
            sum += predict_tree_regression(rf->trees[t], x);
            valid_trees++;
        }
    }

    return valid_trees > 0 ? sum / valid_trees : 0.0;
}
```

---

## 代码解析

### 完整训练流程

```c
model* train_random_forest(const dataset* ds, size_t feature_idx,
                           size_t label_idx, model_config config) {
    // 1. 准备数据
    size_t n_samples = ds->rows;
    size_t n_features = ds->num_features;
    double* X = extract_features(ds, feature_idx);
    int* y = extract_labels(ds, label_idx);

    // 2. 初始化森林
    rf_data* rf = malloc(sizeof(rf_data));
    rf->n_estimators = config.n_estimators;
    rf->trees = malloc(sizeof(dt_node*) * rf->n_estimators);

    // 3. 训练每棵树
    for (size_t t = 0; t < rf->n_estimators; t++) {
        // Bootstrap采样
        double* X_boot = bootstrap(X, n_samples, n_features);
        int* y_boot = bootstrap_labels(y, n_samples);

        // 随机选择特征子集
        size_t max_feat = sqrt(n_features);  // 分类用sqrt

        // 构建决策树
        rf->trees[t] = build_tree(X_boot, y_boot, NULL,
                                  n_samples, n_features, max_feat,
                                  config.max_depth, 2, 1, 0, false);

        free(X_boot); free(y_boot);
    }

    // 4. 返回模型
    return create_model(MODEL_RANDOM_FOREST, rf);
}
```

### 内存管理

```
训练随机森林的内存分配：

1. rf_data: 1次
   └── trees: n_estimators个指针

2. 每棵树训练时：
   ├── X_boot: n_samples * n_features * sizeof(double)
   ├── y_boot: n_samples * sizeof(int)
   └── 树节点: 约 2 * n_samples 个节点

3. 总内存 ≈ O(n_estimators * n_samples * n_features)
```

---

## 使用示例

### 分类任务

```c
#include <stdio.h>
#include "dataset.h"
#include "ml.h"

int main(void) {
    // 加载数据
    csv_t* csv = csv_load("iris.csv");
    dataset* ds = csv_to_dataset(csv, (const char*[]){"species"}, 1);

    // 配置模型
    model_config config = default_config;
    config.n_estimators = 100;   // 100棵树
    config.max_depth = 10;       // 最大深度10
    config.verbose = true;

    // 训练
    model* rf = train_random_forest(ds, 0, 0, config);

    // 评估
    double acc = accuracy(rf, ds, 0, 0);
    printf("准确率: %.2f%%\n", acc * 100);

    // 预测
    double features[] = {5.1, 3.5, 1.4, 0.2};
    int pred = predict_random_forest(rf, features, 4);
    printf("预测类别: %d\n", pred);

    // 释放
    model_free(rf);
    free(rf);
    free_dataset(ds);
    free_csv_data(csv);
    free(csv);

    return 0;
}
```

### 回归任务

```c
// 使用sepal_length预测petal_length
double X_reg[150 * 3];  // 3个特征
double y_reg[150];      // 目标

// 构建回归数据集
dataset* ds_reg = create_regression_dataset(X_reg, y_reg, 150, 3);

// 训练
model* rf_reg = train_random_forest(ds_reg, 0, 0, config);

// 评估
double r2 = r2_score(rf_reg, ds_reg, 0, 0);
double mse_val = mse(rf_reg, ds_reg, 0, 0);
printf("R²: %.4f, MSE: %.4f\n", r2, mse_val);
```

---

## 参数调优建议

### 关键参数

| 参数 | 说明 | 分类推荐 | 回归推荐 |
|------|------|---------|---------|
| `n_estimators` | 树的数量 | 100-500 | 100-300 |
| `max_depth` | 最大深度 | 5-20 | 5-15 |
| `max_features` | 特征数 | √n 或 log₂n | n/3 |
| `min_samples_split` | 分裂最小样本 | 2-10 | 5-10 |
| `min_samples_leaf` | 叶节点最小样本 | 1-5 | 2-5 |

### 调参策略

```
1. 从默认参数开始
2. 调整 n_estimators (通常越多越好，但有边际效应)
3. 调整 max_depth (控制模型复杂度)
4. 调整 max_features (控制随机性)
5. 最后微调 min_samples_*
```

### 避免过拟合

```
过拟合信号：
  - 训练集准确率 >> 测试集准确率

解决方案：
  1. 减小 max_depth
  2. 增加 min_samples_leaf
  3. 增加 min_samples_split
  4. 减少 n_estimators (early stopping)
  5. 降低 max_features
```

---

## 随机森林 vs 单棵决策树

| 特性 | 决策树 | 随机森林 |
|------|-------|---------|
| **准确性** | 较低 | 较高 |
| **过拟合风险** | 高 | 低 |
| **稳定性** | 低 | 高 |
| **可解释性** | 高 | 低 |
| **训练时间** | 快 | 慢 |
| **内存占用** | 低 | 高 |
| **预测时间** | 快 | 慢 (需遍历所有树) |

### 为什么随机森林更好？

```
1. 集成的力量 - 多个模型的平均减少方差
2. 随机性 - 减少树之间的相关性
3. Bootstrap - 增加样本多样性
4. 特征子采样 - 增加特征多样性
```
