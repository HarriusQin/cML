# CatBoost

使用有序目标统计减少过拟合的梯度提升。

## 概述

CatBoost（类别提升）使用针对分类特征的新型目标编码和有序提升算法来减少过拟合和预测偏移。

## 主要特性

- **有序目标统计**: 减少目标泄漏的过拟合
- **自动分类处理**: 无需显式编码即可处理分类特征
- **有序提升**: 在不同子集上训练树以防止预测偏移
- **对称树**: 使用遗忘决策树以加快推理速度

## 与 XGBoost 的差异

| 特性 | XGBoost | CatBoost |
|--------|---------|----------|
| 分类处理 | 需要编码 | 原生有序统计 |
| 树结构 | 非对称 | 对称（遗忘） |
| 过拟合预防 | 行/列采样 | 有序提升 |
| 预测偏移 | 可能发生 | 通过排序缓解 |
| 缺失值 | 需要处理 | 原生处理 |

## 数据结构

```c
typedef struct CatBoost_State {
    CatBoost_Tree** trees;
    size_t n_trees;
    double learning_rate;
    double l2_leaf_reg;
    size_t max_depth;
} CatBoost_State;
```

## 函数

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
```

## 示例

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

// 预测概率
double probs[100 * n_classes];
CatBoost_predict_proba(&model.state, test_ds, NULL, n_features,
                      test_idx, 100, n_classes, probs);
```

## 说明

- CatBoost 使用有序提升来防止预测偏移
- 对称（遗忘）树推理更快
- 目标统计在训练期间即时计算
- 随机排列每个迭代都不同
