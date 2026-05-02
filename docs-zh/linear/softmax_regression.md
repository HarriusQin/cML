# Softmax 回归

使用一对多 softmax 的多分类分类器。对每个类别 $k$：

$$P(y=k|\mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}}}$$

## 算法

1. 对于 K 个类别，训练 K 个二分类器（一对多）
2. 每个分类器 $k$ 求解: $P(y=k) = \sigma(\mathbf{w}_k^T \mathbf{x})$
3. 预测时：选择概率最高的类别

## 两种训练方法

### 正规方程（闭式解）

```c
ML_Model_t create_softmax_model(void);
```
使用 SVD 保证稳定性。速度快但对于大的 K 内存消耗大。

### 梯度下降

```c
ML_Model_t create_softmax_model_gd(void);
```
迭代优化。对于大的 K 更节省内存。

## 配置（梯度下降）

```c
typedef struct {
    double learning_rate;     // 默认: 0.1
    size_t max_iter;          // 默认: 500
    double tolerance;         // 默认: 1e-5
    int verbose;              // 打印进度
} SoftmaxReg_Config;
```

## 数据结构

```c
typedef struct {
    double** weights;         // [K][n_features] 类别权重
    double* bias;             // [K] 类别偏置
    size_t n_classes;
    double learning_rate;
} SoftmaxReg_Weights_t;
```

## 函数

```c
ML_Model_t create_softmax_model(void);
ML_Model_t create_softmax_model_gd(void);

// 概率预测
int softmax_predict_proba(const SoftmaxReg_Weights_t* state,
                          const dataset* ds, size_t* feat_idx,
                          size_t n_features, size_t* indices,
                          size_t n_samples, float* probabilities);
```

## 示例

```c
#define SOFTMAX_REGRESSION_IMPLEMENTATION
#include "softmax_regression.h"

// 创建模型（梯度下降变体）
ML_Model_t model = create_softmax_model_gd();

// 配置
SoftmaxReg_Config config = {
    .learning_rate = 0.1,
    .max_iter = 500,
    .tolerance = 1e-5,
    .verbose = 1
};
model.config.params = &config;
model.config.size = sizeof(config);

// 训练
model.methods.fit(&model.config, &model.state,
                   train_ds, feat_idx, n_features, label_idx,
                   train_idx, train_size);

// 预测类别标签
int predictions[100];
model.methods.predict(&model.state, test_ds, feat_idx, n_features,
                      test_idx, 100, predictions);

// 获取概率
float probs[100 * 3];  // 100 样本, 3 类别
softmax_predict_proba(&model.state, test_ds, feat_idx, n_features,
                      test_idx, 100, probs);
```

## 说明

- 自动添加偏置项
- 一对多方法允许多分类而不需修改算法
- 梯度下降变体对于多类别更节省内存
- 特征缩放可改善梯度下降的收敛性
