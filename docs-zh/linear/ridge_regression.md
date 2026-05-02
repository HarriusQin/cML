# 岭回归

带 L2 正则化的线性回归。求解：

$$\mathbf{w} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$

## 算法

1. 给设计矩阵 $\mathbf{X}$ 添加偏置列
2. 计算正则化正规方程：$\mathbf{w} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$
3. 使用 SVD 保证数值稳定性

## 配置

```c
typedef struct {
    double alpha;             // 正则化强度（默认: 1.0）
    int verbose;
} Ridge_Config_t;
```

## 数据结构

```c
typedef struct {
    double* weights;          // [n_features + 1]
    double alpha;             // 正则化参数
    double rss;
    double r_squared;
} Ridge_Weights_t;
```

## 函数

```c
ML_Model_t create_ridge_regression_model(double alpha);

int Ridge_fit(const ML_Model_Config_t* config, Ridge_Weights_t* state,
              const dataset* ds, size_t* feat_idx, size_t n_features,
              size_t label_idx, size_t* indices, size_t n_samples);

int Ridge_predict(const Ridge_Weights_t* state, const dataset* ds,
                  size_t* feat_idx, size_t n_features,
                  size_t* indices, size_t n_samples, double* predictions);
```

## 示例

```c
#define RIDGE_REGRESSION_IMPLEMENTATION
#include "ridge_regression.h"

// 创建 alpha=0.5 的模型
ML_Model_t model = create_ridge_regression_model(0.5);

Ridge_Config_t config = {.alpha = 0.5, .verbose = 1};
model.config.params = &config;
model.config.size = sizeof(config);

model.methods.fit(&model.config, &model.state, ds, ...);

// 获取系数
double coef[5];
model.methods.get_coefficients(&model.state, coef);
```

## 使用场景

- **多重共线性**: 当特征相关时，普通 MLR 会变得不稳定
- **过拟合**: L2 正则化惩罚大的权重
- **高维数据**: 当特征数接近或超过样本数时

## 说明

- α=0: 等同于普通最小二乘
- α→∞: 所有权重趋近于零
- 特征缩放可获得最佳效果
