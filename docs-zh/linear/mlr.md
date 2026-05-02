# 多元线性回归 (MLR)

使用正规方程学习线性关系。

## 概述

多元线性回归（MLR）通过找到特征和目标之间的线性关系来预测连续值：

$$\hat{y} = \mathbf{X}\boldsymbol{\theta}$$

其中 $\boldsymbol{\theta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$

## 算法

1. 构建设计矩阵 $\mathbf{X}$（添加偏置列）
2. 计算 $\mathbf{X}^T\mathbf{X}$
3. 求逆或使用 SVD 获得解
4. 对于病态矩阵，使用 SVD 增强数值稳定性

## 数据结构

```c
typedef struct {
    Matrix* weights;    // [n_features, 1]
    double* bias;       // 偏置项
} MLR_State;
```

## 函数

```c
int mlr_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
            const dataset* ds, const size_t* feature_indices,
            size_t n_features, size_t target_index,
            const size_t* sample_indices, size_t n_samples);

int mlr_predict(const ML_Weights_t* state, const dataset* ds,
                const size_t* feature_indices, size_t n_features,
                const size_t* sample_indices, size_t n_samples,
                void* output);
```

## 示例

```c
#define MLR_IMPLEMENTATION
#include "mlr.h"

// 训练
ML_Weights_t state = {0};
mlr_fit(NULL, &state, ds, feat_idx, n_features, target_idx,
        train_idx, train_size);

// 预测
double predictions[100];
mlr_predict(&state, test_ds, feat_idx, n_features,
            test_idx, 100, predictions);

// 释放
free(state.weights);
free(state.bias);
```

## 说明

- 使用正规方程，不需要迭代
- 对于病态矩阵自动使用 SVD
- 特征缩放可以提高数值稳定性
- 偏置项作为第一个权重存储
