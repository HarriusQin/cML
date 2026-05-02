# AdaBoost

使用重加权弱分类器的序列增强。

## 概述

AdaBoost（自适应增强）通过序列训练决策树桩来构建强分类器。每个后续分类器专注于被前面分类器错误分类的样本。

## 算法

1. 初始化样本权重 $D_1(i) = 1/n$
2. 对于 $t = 1, \ldots, T$:
   - 训练弱分类器 $h_t$ 最小化加权误差
   - 计算弱分类器权重: $\alpha_t = \frac{1}{2} \ln\frac{1-\epsilon_t}{\epsilon_t}$
   - 更新样本权重: $D_{t+1}(i) = D_t(i) \exp(-\alpha_t y_i h_t(\mathbf{x}_i))$
   - 归一化权重
3. 最终分类器: $H(\mathbf{x}) = \text{sign}\left(\sum_t \alpha_t h_t(\mathbf{x})\right)$

## 数据结构

```c
typedef struct {
    DecisionTree** stumps;     // 弱分类器（决策树桩）
    double* alphas;             // 每个树桩的权重
    int n_stumps;
} AdaBoost_State;
```

## 函数

```c
ML_Model_t create_adaboost_model(void);

int AdaBoost_fit(const ML_Model_Config_t* config, AdaBoost_State* state,
                 const dataset* ds, size_t* feat_idx, size_t n_features,
                 size_t label_idx, size_t* indices, size_t n_samples);

int AdaBoost_predict(const AdaBoost_State* state, const dataset* ds,
                     size_t* feat_idx, size_t n_features,
                     size_t* indices, size_t n_samples, int* predictions);
```

## 示例

```c
#define ADABOOST_IMPLEMENTATION
#include "adaboost.h"

ML_Model_t model = create_adaboost_model();

AdaBoost_Config_t config = {
    .n_estimators = 50,
    .learning_rate = 1.0
};
model.config.params = &config;
model.config.size = sizeof(config);

model.methods.fit(&model.config, &model.state,
                 train_ds, feat_idx, n_features, label_idx,
                 train_idx, train_size);

int predictions[100];
model.methods.predict(&model.state, test_ds, feat_idx, n_features,
                     test_idx, 100, predictions);
```

## 与随机森林的比较

| 方面 | AdaBoost | 随机森林 |
|--------|---------|----------|
| 训练 | 序列 | 并行 |
| 弱分类器 | 树桩（深度 1） | 完全树 |
| 权重调整 | 基于误差 | 相等 bootstrap |
| 对噪声的敏感性 | 高 | 低 |

## 说明

- 对噪声和异常值敏感
- 学习率控制每个弱分类器的贡献
- 通常使用深度为 1 的决策树（决策树桩）
- 可以与任何弱分类器一起使用
