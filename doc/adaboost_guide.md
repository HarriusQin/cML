# AdaBoost 实现指南

## 目录

1. [算法原理](#算法原理)
2. [核心概念](#核心概念)
3. [数据结构设计](#数据结构设计)
4. [关键算法实现](#关键算法实现)
5. [代码解析](#代码解析)
6. [使用示例](#使用示例)
7. [参数调优](#参数调优)
8. [与随机森林对比](#与随机森林对比)

---

## 算法原理

AdaBoost (Adaptive Boosting) 是一种**序列集成**方法，通过训练多个弱分类器并加权组合来构建强分类器。

### 核心思想

```
AdaBoost = Adaptive + Boosting

"Boosting" - 序列提升：
  1. 训练第一个弱分类器
  2. 根据错误调整样本权重
  3. 训练第二个弱分类器（重点关注错误样本）
  4. 重复以上步骤

"Adaptive" - 自适应调整：
  每个弱分类器的权重根据其准确率自动调整
```

### 算法流程

```
输入: 训练数据 (x₁,y₁), (x₂,y₂), ..., (xₙ,yₙ)
      弱分类器数量 M

1. 初始化样本权重:
   D₁(i) = 1/n  (均匀分布)

2. For m = 1 to M:
   a. 使用权重 Dₘ 训练弱分类器 hₘ
   b. 计算弱分类器的加权错误率:
      εₘ = Σ Dₘ(i) * [hₘ(xᵢ) ≠ yᵢ]

   c. 计算弱分类器权重:
      αₘ = (1/2) * ln((1-εₘ)/εₘ)

   d. 更新样本权重:
      Dₘ₊₁(i) = Dₘ(i) * exp(-αₘ * yᵢ * hₘ(xᵢ))

   e. 归一化权重:
      Dₘ₊₁(i) = Dₘ₊₁(i) / Σ Dₘ₊₁(j)

3. 最终分类器:
   H(x) = sign(Σ αₘ * hₘ(x))
```

### 数学解释

#### 权重更新公式

```
正确分类: yᵢ * hₘ(xᵢ) = +1
  Dₘ₊₁(i) = Dₘ(i) * exp(-αₘ)

错误分类: yᵢ * hₘ(xᵢ) = -1
  Dₘ₊₁(i) = Dₘ(i) * exp(+αₘ)

由于 αₘ > 0，错误样本的权重被放大，正确样本的权重被缩小
```

#### 权重系数

```
αₘ = (1/2) * ln((1-εₘ)/εₘ)

当 εₘ = 0.5 (随机猜测): αₘ = 0
当 εₘ → 0 (完美分类):   αₘ → +∞
当 εₘ → 1 (完全错误):   αₘ → -∞
```

---

## 核心概念

### 1. 弱分类器 - 决策树桩

AdaBoost通常使用**决策树桩** (Decision Stump) 作为弱分类器：

```c
typedef struct {
    size_t feature_index;   // 分裂特征
    double threshold;       // 分裂阈值
    int left_class;         // 左子节点类别
    int right_class;        // 右子节点类别
    double alpha;           // 权重系数
} adaboost_stump;
```

决策树桩是最简单的决策树：
- 只有一层分裂
- 只有一个特征被使用
- 类似: `if x[feature] <= threshold then class_A else class_B`

### 2. 加权错误率

```
εₘ = Σ wᵢ * I(yᵢ ≠ hₘ(xᵢ))

其中:
  wᵢ = 样本权重
  I(condition) = 条件为真返回1，否则返回0
```

### 3. 指数损失函数

AdaBoost优化的是指数损失：

```
L(H) = Σ exp(-yᵢ * H(xᵢ))

其中 H(x) = Σ αₘ * hₘ(x) 是最终分类器
```

### 4. 实时权重调整

```
训练过程中，被错误分类的样本权重会指数增长，
而正确分类的样本权重会指数衰减。
这使得后续分类器自动聚焦于"难分类"的样本。
```

---

## 数据结构设计

### AdaBoost数据结构

```c
/**
 * AdaBoost弱分类器 - 决策树桩
 */
typedef struct {
    size_t feature_index;   // 分裂特征索引
    double threshold;       // 分裂阈值
    int left_class;         // 左侧类别 (<=threshold时)
    int right_class;        // 右侧类别 (>threshold时)
    double alpha;           // 该弱分类器的权重系数
} adaboost_stump;

/**
 * AdaBoost模型数据
 */
typedef struct {
    adaboost_stump* stumps;     // 弱分类器数组
    double* weights;            // 当前样本权重
    size_t n_stumps;           // 弱分类器数量
    size_t n_classes;          // 类别数
} adaboost_data;
```

### 模型句柄

```c
typedef struct model {
    model_type_t type;         // MODEL_ADABOOST
    void* data;                // adaboost_data*
} model;
```

---

## 关键算法实现

### 1. 训练决策树桩

```c
/**
 * 训练单个决策树桩（弱分类器）
 *
 * 遍历所有特征和阈值，找使加权错误率最小的分裂点
 *
 * @param X         特征矩阵 [n * n_features]
 * @param y         标签数组 [n]
 * @param weights   样本权重 [n]
 * @param n         样本数量
 * @param n_features 特征数量
 * @param n_classes  类别数量
 *
 * @return 训练好的决策树桩
 */
adaboost_stump* train_stump(double* X, int* y, double* weights,
                           size_t n, size_t n_features, size_t n_classes) {

    adaboost_stump* stump = malloc(sizeof(adaboost_stump));

    double best_error = 1e30;

    // 遍历每个特征
    for (size_t f = 0; f < n_features; f++) {

        // 找特征的最小值和最大值
        double min_val = X[f], max_val = X[f];
        for (size_t i = 1; i < n; i++) {
            double v = X[i * n_features + f];
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }

        // 跳过常值特征
        if (fabs(max_val - min_val) < 1e-10) continue;

        // 尝试多个阈值
        for (int t = 1; t <= 10; t++) {
            double thresh = min_val + (max_val - min_val) * t / 10.0;

            // 尝试每个类别组合
            for (size_t c = 0; c < n_classes; c++) {
                int left_class = (int)c;
                int right_class = (int)((c + 1) % n_classes);

                // 计算加权错误率
                double error = 0.0;
                for (size_t i = 0; i < n; i++) {
                    int pred = (X[i * n_features + f] <= thresh) ?
                               left_class : right_class;
                    if (pred != y[i])
                        error += weights[i];  // 加权！
                }

                // 更新最佳分裂
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

    // 计算alpha
    // 防止除零
    if (best_error < 1e-10) best_error = 1e-10;
    if (best_error > 1.0 - 1e-10) best_error = 1.0 - 1e-10;

    stump->alpha = 0.5 * log((1.0 - best_error) / best_error);

    return stump;
}
```

### 2. 权重更新

```c
/**
 * 更新AdaBoost样本权重
 *
 * Dₘ₊₁(i) = Dₘ(i) * exp(-αₘ * yᵢ * hₘ(xᵢ))
 *
 * @param weights    输入/输出：样本权重
 * @param X         特征矩阵
 * @param y         标签
 * @param stump     当前弱分类器
 * @param n         样本数量
 * @param n_features 特征数
 */
void update_weights(double* weights, double* X, int* y,
                    adaboost_stump* stump, size_t n, size_t n_features) {

    double alpha = stump->alpha;
    double weight_sum = 0.0;

    // 更新每个样本的权重
    for (size_t i = 0; i < n; i++) {
        // 预测
        int pred = (X[i * n_features + stump->feature_index] <= stump->threshold) ?
                   stump->left_class : stump->right_class;

        // y_i * h(x_i): +1表示正确，-1表示错误
        double y_i_times_h = (pred == y[i]) ? 1.0 : -1.0;

        // 指数更新
        weights[i] *= exp(-alpha * y_i_times_h);
        weight_sum += weights[i];
    }

    // 归一化
    if (weight_sum > 0) {
        for (size_t i = 0; i < n; i++)
            weights[i] /= weight_sum;
    }
}
```

### 3. AdaBoost训练循环

```c
/**
 * 训练AdaBoost模型
 */
model* train_adaboost(const dataset* ds, size_t feature_idx,
                      size_t label_idx, model_config config) {

    size_t n_samples = ds->rows;
    size_t n_features = ds->num_features;
    size_t n_classes = ds->labels[label_idx].classes;
    size_t n_stumps = config.n_estimators;

    // 分配内存
    model* m = malloc(sizeof(model));
    adaboost_data* data = malloc(sizeof(adaboost_data));

    data->stumps = malloc(sizeof(adaboost_stump) * n_stumps);
    data->weights = malloc(sizeof(double) * n_samples);
    data->n_stumps = n_stumps;
    data->n_classes = n_classes;

    // 准备数据
    double* X = extract_features(ds, feature_idx);
    int* y = extract_labels(ds, label_idx);

    // 初始化权重为均匀分布
    for (size_t i = 0; i < n_samples; i++)
        data->weights[i] = 1.0 / n_samples;

    // AdaBoost主循环
    for (size_t t = 0; t < n_stumps; t++) {

        // a. 训练弱分类器
        adaboost_stump* stump = train_stump(X, y, data->weights,
                                            n_samples, n_features, n_classes);
        data->stumps[t] = *stump;
        free(stump);

        // b. 计算加权错误率
        double error = 0.0;
        for (size_t i = 0; i < n_samples; i++) {
            int pred = predict_stump(&data->stumps[t],
                                     &X[i * n_features]);
            if (pred != y[i])
                error += data->weights[i];
        }

        // 防止数值问题
        if (error < 1e-10) error = 1e-10;
        if (error > 1.0 - 1e-10) error = 1.0 - 1e-10;

        // c. 更新alpha
        data->stumps[t].alpha = 0.5 * log((1.0 - error) / error);

        // d. 更新样本权重
        update_weights(data->weights, X, y, &data->stumps[t],
                       n_samples, n_features);
    }

    free(X); free(y);

    m->type = MODEL_ADABOOST;
    m->data = data;

    return m;
}
```

### 4. AdaBoost预测

```c
/**
 * AdaBoost预测 - 加权多数投票
 */
int predict_adaboost(const model* m, const double* x, size_t n) {
    (void)n;  // 未使用

    adaboost_data* data = (adaboost_data*)m->data;

    // 累加每个类别的加权分数
    double* class_scores = calloc(data->n_classes, sizeof(double));

    for (size_t t = 0; t < data->n_stumps; t++) {
        // 跳过无效分类器
        if (data->stumps[t].alpha <= 0) continue;

        // 预测
        int pred = predict_stump(&data->stumps[t], x);

        // 累加权重
        if (pred >= 0 && (size_t)pred < data->n_classes) {
            class_scores[pred] += data->stumps[t].alpha;
        }
    }

    // 找最高分
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

/**
 * 决策树桩预测
 */
int predict_stump(const adaboost_stump* stump, const double* x) {
    if (x[stump->feature_index] <= stump->threshold)
        return stump->left_class;
    else
        return stump->right_class;
}
```

---

## 代码解析

### 完整训练流程图

```
train_adaboost()
    │
    ├── 1. 初始化
    │   ├── 提取特征矩阵 X
    │   ├── 提取标签 y
    │   └── weights[] = 1/n (均匀分布)
    │
    ├── 2. For t = 1 to M:
    │   │
    │   ├── train_stump(X, y, weights)
    │   │   ├── 遍历所有特征
    │   │   ├── 对每个特征尝试多个阈值
    │   │   ├── 对每个阈值计算加权错误率
    │   │   └── 返回最佳分裂点
    │   │
    │   ├── 计算 alpha_t = 0.5 * log((1-ε)/ε)
    │   │
    │   └── update_weights(X, y, weights, stump)
    │       ├── 对每个样本:
    │       │   ├── 预测正确: weight *= exp(-alpha)
    │       │   └── 预测错误: weight *= exp(+alpha)
    │       └── 归一化
    │
    └── 3. 返回模型
```

### 内存布局

```
adaboost_data:
    ├── stumps: adaboost_stump[M]
    │   └── 每个stump包含:
    │       ├── feature_index
    │       ├── threshold
    │       ├── left_class
    │       ├── right_class
    │       └── alpha
    │
    └── weights: double[n_samples]
```

---

## 使用示例

```c
#include <stdio.h>
#include "dataset.h"
#include "ml.h"

int main(void) {
    // 加载数据
    csv_t* csv = csv_load("iris.csv");
    dataset* ds = csv_to_dataset(csv, (const char*[]){"species"}, 1);

    // 配置
    model_config config = default_config;
    config.n_estimators = 50;    // 50个弱分类器
    config.verbose = true;

    // 训练
    model* ada = train_adaboost(ds, 0, 0, config);

    // 评估
    double acc = accuracy(ada, ds, 0, 0);
    printf("准确率: %.2f%%\n", acc * 100);

    // 预测
    double features[] = {5.1, 3.5, 1.4, 0.2};
    int pred = predict_adaboost(ada, features, 4);
    printf("预测类别: %d\n", pred);

    // 释放
    model_free(ada);
    free(ada);
    free_dataset(ds);
    free_csv_data(csv);
    free(csv);

    return 0;
}
```

---

## 参数调优

### 关键参数

| 参数 | 说明 | 推荐值 | 注意事项 |
|------|------|--------|---------|
| `n_estimators` | 弱分类器数量 | 50-200 | 太多可能过拟合 |
| `learning_rate` | 学习率(未使用) | 1.0 | AdaBoost不用此参数 |

### 调参建议

```
1. n_estimators (弱分类器数量)
   - 从50开始
   - 如果训练误差小但测试误差大，减少数量
   - 边际效应：超过一定数量后提升不明显

2. 弱分类器复杂度
   - 本实现使用决策树桩（最简单的决策树）
   - 可尝试稍微复杂的树（如深度=2）

3. 监控
   - 每轮训练后检查错误率
   - 错误率应该在下降后趋于稳定
```

---

## 与随机森林对比

| 特性 | AdaBoost | 随机森林 |
|------|----------|---------|
| **集成方式** | 序列 | 并行 |
| **弱分类器** | 决策树桩(深度=1) | 决策树(不限深度) |
| **权重调整** | 样本权重迭代调整 | Bootstrap采样 |
| **训练时间** | 较慢(序列) | 较快(并行) |
| **对噪声** | 敏感 | 较鲁棒 |
| **可解释性** | 中等 | 较低 |

### 主要区别

```
AdaBoost:
  - 后一个分类器依赖前一个的结果
  - 样本权重自动调整
  - 弱分类器必须比随机好一点点(ε < 0.5)

Random Forest:
  - 树之间相互独立
  - 随机特征子集
  - 可以是任意深度的完整树
```

### 何时使用哪个？

```
选择 AdaBoost 当:
  - 数据相对干净（噪声少）
  - 需要较少的弱分类器
  - 需要自适应调整样本权重

选择 Random Forest 当:
  - 数据有噪声
  - 需要更稳定的模型
  - 需要同时做分类和回归
```
