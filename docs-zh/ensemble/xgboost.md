# XGBoost

带正则化的梯度提升决策树。

## 概述

XGBoost（极限梯度提升）是一种优化的梯度提升实现，内置正则化（L1/L2）以防止过拟合。它序列地训练决策树来修正前面树的错误。

## 算法

1. 对于每次迭代 $t$:
   - 计算损失函数的梯度和二阶导数
   - 通过最大化增益找到最佳分裂:
   $$\text{gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$
   - 使用 $w = -G / (H + \lambda)$ 创建新叶权重
   - 更新预测

## 数据结构

```c
typedef struct XGBoost_State {
    XGBoost_Tree** trees;
    double tree_weights;
    size_t n_trees;
    double learning_rate;
    double reg_lambda;        // L2 正则化
    double reg_alpha;         // L1 正则化
    size_t max_depth;
    double subsample;        // 行采样
    double colsample_bytree;  // 列采样
} XGBoost_State;
```

## 函数

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
                          size_t n_classes, void* probabilities);
```

## 示例

```c
#define XGBOOST_IMPLEMENTATION
#include "xgboost.h"

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

// 训练
model.methods.fit(&model.config, &model.state,
                  train_ds, NULL, n_features, label_idx,
                  train_idx, train_size);

// 预测
int predictions[100];
model.methods.predict(&model.state, test_ds, NULL, n_features,
                     test_idx, 100, predictions);
```

## 说明

- XGBoost 使用基于预排序的分裂查找（精确算法）
- 梯度裁剪有助于 logistic 回归的稳定性
- subsample 和 colsample_bytree 添加随机性以进行正则化
- 叶权重公式: $w_j = -G_j / (H_j + \lambda)$
