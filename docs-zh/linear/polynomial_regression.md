# 多项式回归

扩展线性回归以捕获非线性关系。

## 概述

多项式回归通过将每个特征的幂作为新特征添加到线性模型中，使模型能够拟合曲线，同时仍使用线性回归的正规方程。

## 算法

对于 degree $d$ 的多项式：

$$y_i = \theta_0 + \theta_1 x_i + \theta_2 x_i^2 + \cdots + \theta_d x_i^d + \epsilon_i$$

设计矩阵扩展为包含多项式项：

$$\mathbf{X} = \begin{bmatrix} 1 & x_1 & x_1^2 & \cdots & x_1^d \\ 1 & x_2 & x_2^2 & \cdots & x_2^d \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_n & x_n^2 & \cdots & x_n^d \end{bmatrix}$$

然后求解正规方程: $\boldsymbol{\theta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$

## 数据结构

```c
typedef struct PolyReg_State {
    Matrix* weights;    // 多项式系数 [degree]
    double* bias;       // 截距项
    size_t degree;      // 多项式阶数
} PolyReg_State;
```

## 函数

```c
// 生成到指定阶数的多项式特征
static int polynomial_features(
    const double* x,       // 输入 [n_samples]
    size_t n_samples,
    size_t degree,
    Matrix* result          // 输出 [n_samples, degree]
);

// 拟合多项式回归
static int poly_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
                    const double* x, const double* y,
                    size_t n_samples, size_t degree);

// 使用拟合模型预测
static int poly_predict(const ML_Weights_t* state,
                         const double* x, size_t n_samples,
                         double* output);
```

## 示例

```c
#define POLYNOMIAL_REGRESSION_IMPLEMENTATION
#include "polynomial_regression.h"

// 生成数据: y = 3 + 2x + x^2 + noise
double x[] = {0, 1, 2, 3, 4, 5};
double y[] = {3.1, 6.2, 13.0, 22.1, 35.0, 50.2};

// 拟合 degree-2 多项式
PolyReg_State state = {0};
poly_fit(NULL, &state, x, y, 6, 2);

// 预测
double x_test[] = {1.5, 2.5, 3.5};
double y_pred[3];
poly_predict(&state, x_test, 3, y_pred);

// 释放
poly_free(&state);
```

## 偏差-方差权衡

| 阶数 | 偏差 | 方差 | 过拟合风险 |
|--------|------|----------|-----------------|
| 1（线性） | 高 | 低 | 欠拟合 |
| 2-3 | 中 | 中 | 良好平衡 |
| 高（>10） | 低 | 高 | 过拟合 |

## 说明

- 为保证数值稳定性，应始终对多项式特征进行标准化
- 高阶多项式会显著增加过拟合风险
- 对于高阶多项式，使用正则化（岭回归）
- 病态矩阵使用 SVD 求解
