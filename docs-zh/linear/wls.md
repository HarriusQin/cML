# 加权最小二乘 (WLS)

用于异方差数据的带样本权重的回归。

## 概述

加权最小二乘（WLS）通过为不同样本分配不同的权重来扩展普通最小二乘。当观测值具有不同方差或以不同精度测量时，这是必需的。

## 问题形式

普通 LS 最小化: $\sum_{i=1}^n (y_i - \mathbf{x}_i^T\boldsymbol{\theta})^2$

WLS 最小化: $\sum_{i=1}^n w_i (y_i - \mathbf{x}_i^T\boldsymbol{\theta})^2$

其中 $w_i$ 是样本 $i$ 的权重，通常与 $1/\sigma_i^2$ 成正比。

## 解法

加权正规方程：

$$\boldsymbol{\theta} = (\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^T\mathbf{W}\mathbf{y}$$

其中 $\mathbf{W} = \text{diag}(w_1, w_2, \ldots, w_n)$。

## 数据结构

```c
typedef struct WLS_State {
    Matrix* weights;   // 系数矩阵 [1, n_features]
    double* bias;     // 截距项
} WLS_State;
```

## 函数

```c
// 拟合加权线性回归
static int wls_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
                   const dataset* ds, const size_t* feature_indices,
                   size_t n_features, size_t target_index,
                   const size_t* sample_indices, size_t n_samples,
                   const double* weights);  // 样本权重

// 使用拟合模型预测
static int wls_predict(const ML_Weights_t* state, const dataset* ds,
                        const size_t* feature_indices, size_t n_features,
                        const size_t* sample_indices, size_t n_samples,
                        void* output);
```

## 示例

```c
#define WLS_IMPLEMENTATION
#include "wls.h"

// 基于测量不确定性的样本权重
double weights[] = {1.0, 1.0, 0.5, 0.5, 0.25, 0.25};  // 对噪声大的样本赋予较低权重

ML_Weights_t state = {0};
wls_fit(NULL, &state, ds, feat_idx, n_features, target_idx,
        train_idx, n_samples, weights);

// 预测
double predictions[100];
wls_predict(&state, test_ds, feat_idx, n_features,
           test_idx, 100, predictions);
```

## 说明

- 权重应与方差成反比以获得有效估计
- WLS 是广义最小二乘（GLS）的特例
- 如果权重未知，使用可行 GLS（FGLS）与估计权重
- 当 $\mathbf{X}^T\mathbf{W}\mathbf{X}$ 奇异时使用基于 SVD 的伪逆
