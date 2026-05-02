# 集成学习

组合多个弱学习器来创建强分类器。

## 模型

| 模型 | 文件 | 关键算法 |
|-------|------|--------------|
| AdaBoost | [adaboost.md](adaboost.md) | 使用重加权的序列提升 |
| 随机森林 | [randomforest.md](randomforest.md) | 使用树的并行 bagging |
| XGBoost | [xgboost.md](xgboost.md) | 梯度提升 |
| CatBoost | [catboost.md](catboost.md) | 有序提升 |

## 共同概念

### 弱学习器

性能略好于随机猜测的模型。在本库中，通常使用决策树桩（深度为 1 的树）。

### Bagging 与 Boosting

- **Bagging**（随机森林）：在 bootstrap 样本上并行训练学习器
- **Boosting**（AdaBoost）：序列训练学习器，重新加权错误分类的样本

### 特征随机性

随机森林在每次分裂时随机选择特征子集，使树去相关。
