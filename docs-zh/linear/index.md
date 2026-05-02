# 线性模型

学习特征和目标之间线性关系的模型。

## 模型

| 模型 | 文件 | 类型 | 用途 |
|------|------|------|----------|
| 多元线性回归 | [mlr.md](mlr.md) | 回归 | 连续目标 |
| 岭回归 | [ridge_regression.md](ridge_regression.md) | 回归 | L2 正则化 |
| 多项式回归 | [polynomial_regression.md](polynomial_regression.md) | 回归 | 非线性关系 |
| Softmax 回归 | [softmax_regression.md](softmax_regression.md) | 分类 | 多分类 |
| 加权最小二乘 | [wls.md](wls.md) | 回归 | 异方差数据 |

## 共同性质

- 所有模型都求解正规方程的变体: $\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$
- 特征缩放有助于梯度下降变体
- 岭回归添加 L2 正则化: $\mathbf{w} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$

## 特征缩放

大多数线性模型受益于特征缩放：

```c
ML_ScalingParams_t* scaler = ml_fit_scaling(ds, feat_idx, n_features,
                                            train_idx, train_size,
                                            SCALING_STANDARD);
dataset* scaled_train = ml_transform_features(scaler, ds, feat_idx, n_features,
                                              train_idx, train_size);
```
