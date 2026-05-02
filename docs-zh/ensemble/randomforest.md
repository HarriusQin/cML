# 随机森林

使用随机特征选择的决策树 bootstrap 聚合（bagging）。

## 算法

1. 对于 $b = 1, \ldots, B$ 棵树：
   - 抽取大小为 $N$ 的 bootstrap 样本（有放回）
   - 在 bootstrap 样本上训练决策树
   - 在每个节点，随机选择 $m$ 个特征
   - 在选中的 $m$ 个特征中找最佳分裂
2. 聚合预测：
   - **分类**: 多数投票
   - **回归**: 平均预测

## 数据结构

```c
#define RANDOMFOREST_MAX_ESTIMATORS 200

typedef struct {
    int n_estimators;
    DecisionTree** trees;         // Bootstrap 树
    double* feature_importance;   // [n_features] 重要性分数
} RandomForest_Weights_t;
```

## 配置

```c
typedef struct {
    int n_estimators;             // 树的数量（默认: 100）
    int max_depth;                // 最大树深度（默认: 无限制）
    size_t min_samples_split;     // 分裂所需最小样本数（默认: 2）
    size_t n_features_subset;      // 每次分裂的特征数（默认: sqrt(n)）
} RandomForest_Config_t;
```

## 函数

```c
ML_Model_t create_random_forest_model(void);

int RandomForest_fit(const ML_Model_Config_t* config, RandomForest_Weights_t* state,
                      const dataset* ds, size_t* feat_idx, size_t n_features,
                      size_t label_idx, size_t* indices, size_t n_samples);

int RandomForest_predict(const RandomForest_Weights_t* state, const dataset* ds,
                          size_t* feat_idx, size_t n_features,
                          size_t* indices, size_t n_samples, int* predictions);
```

## 特征重要性

训练后可计算特征重要性分数：

```c
double* importance = forest->feature_importance;
for (size_t i = 0; i < n_features; i++) {
    printf("特征 %zu: %.4f\n", i, importance[i]);
}
```

## 示例

```c
#define RANDOMFOREST_IMPLEMENTATION
#include "randomforest.h"

ML_Model_t model = create_random_forest_model();

RandomForest_Config_t config = {
    .n_estimators = 200,
    .max_depth = 20,
    .min_samples_split = 5,
    .n_features_subset = 0  // 0 = 使用 sqrt(n)
};
model.config.params = &config;
model.config.size = sizeof(config);

model.methods.fit(&model.config, &model.state,
                  train_ds, NULL, n_features, label_idx,
                  train_idx, train_size);

int predictions[100];
model.methods.predict(&model.state, test_ds, NULL, n_features,
                     test_idx, 100, predictions);
```

## 相对于 AdaBoost 的优势

- 对噪声更鲁棒（平均减少方差）
- 可并行训练（树是独立的）
- 特征重要性估计
- 更不易过拟合

## 说明

- `n_features_subset = 0` 默认为 $\sqrt{n\_features}$
- 更高的 `n_estimators` 通常更好但更慢
- 对于不平衡数据，考虑类别加权
