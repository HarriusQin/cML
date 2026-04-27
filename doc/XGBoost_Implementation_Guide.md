# XGBoost 原理与 C 语言实现指南

基于论文 "XGBoost: A Scalable Tree Boosting System" (Chen & Guestrin, 2016) 及 cML 框架。

---

## 1. XGBoost vs AdaBoost 核心区别

| 特性 | AdaBoost | XGBoost |
|------|----------|---------|
| **基础学习器** | 决策树桩（单层，单一特征） | 完整决策树（递归多叉树） |
| **优化方式** | 样本权重重分配 | 梯度下降（一阶/二阶梯度） |
| **目标函数** | 指数损失 min Σexp(-yi·f(xi)) | 正则化目标 L(ŷ) + ΣΩ |
| **预测合并** | 加权投票 | 叠加（Additive Training） |
| **正则化** | 单一 alpha 权重 | L1/L2 正则化 + 树结构惩罚 |

---

## 2. 正则化目标函数

### 2.1 目标函数

$$
\text{Obj} = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)
$$

其中：
- $l(y_i, \hat{y}_i)$：可微凸损失函数（如 logistic loss）
- $\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$：树结构正则化
  - $T$：叶节点数
  - $w_j$：第 $j$ 个叶子的权重
  - $\gamma$：最小分裂损失（min_split_loss）
  - $\lambda$：L2 正则化系数

### 2.2 叠加训练（Additive Training）

第 $t$ 轮预测：

$$
\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta \cdot f_t(x_i)
$$

- $\hat{y}_i^{(0)}$：初始预测（通常为 label 均值或 0）
- $\eta$：学习率（shrinkage），默认 0.3
- $f_t(x_i)$：第 $t$ 棵树对样本 $i$ 的预测

---

## 3. 梯度提升框架

### 3.1 损失函数展开

对第 $t$ 轮泰勒展开（二阶近似）：

$$
l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) \approx l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t(x_i)^2
$$

其中：
- $g_i = \frac{\partial l(y_i, \hat{y})}{\partial \hat{y}}\big|_{\hat{y}=\hat{y}_i^{(t-1)}}$ 一阶导数（梯度）
- $h_i = \frac{\partial^2 l(y_i, \hat{y})}{\partial \hat{y}^2}\big|_{\hat{y}=\hat{y}_i^{(t-1)}}$ 二阶导数（Hessian）

### 3.2 通用梯度提升算法

```
输入: 数据集 D = {(xi, yi)}, 损失函数 l, 轮数 T
输出: K 棵树的集合 {f_k}

1. 初始化预测 ŷ_i^{(0)} = 0（或 label 均值）
2. for t = 1 to T do:
     a. 计算梯度: g_i = ∂l(yi, ŷ_i^{(t-1)})/∂ŷ, h_i = ∂²l/∂ŷ²
     b. 构造树 f_t（贪心算法，见 Section 4）
     c. 更新预测: ŷ_i^{(t)} = ŷ_i^{(t-1)} + η · f_t(xi)
3. 输出最终模型: F(x) = Σ_{k=1}^K η · f_k(x)
```

---

## 4. 树结构学习（贪心算法）

### 4.1 树结构评分

给定一个叶子 $j$ 上的一组样本 $\{I_j\}$：

$$
\text{Obj}_j = -\frac{1}{2} \frac{G_j^2}{H_j + \lambda} + \gamma
$$

其中：
- $G_j = \sum_{i \in I_j} g_i$
- $H_j = \sum_{i \in I_j} h_i$

给定一个分裂点，分裂后的增益：

$$
\text{Gain} = \frac{1}{2} \left[
    \frac{G_L^2}{H_L + \lambda} +
    \frac{G_R^2}{H_R + \lambda} -
    \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}
\right] - \gamma
$$

### 4.2 精确贪心算法

```
输入: 当前节点样本集合 I, 梯度 g[1..n], Hessian h[1..n]
输出: 最优分裂

best_gain = -∞
for each feature f do:
    # 按特征值排序
    samples = sort(I, by=feature[f].value)

    # 计算所有可能的分裂点
    for split_idx = 1 to |samples| - 1 do:
        if samples[split_idx].value == samples[split_idx+1].value:
            continue

        G_L = Σ_{i in left} g_i,  H_L = Σ_{i in left} h_i
        G_R = Σ_{i in right} g_i, H_R = Σ_{i in right} h_i

        gain = 0.5 * (G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ)) - γ

        if gain > best_gain:
            best_gain = gain
            best_split = (feature=f, threshold=midpoint)
```

### 4.3 叶子权重计算

当树构建完成，每个叶子 $j$ 的最优权重：

$$
w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}
$$

---

## 5. 常用损失函数

### 5.1 分类（Logistic Loss）

$$
l(y, \hat{y}) = -[y \cdot \log(\sigma(\hat{y})) + (1-y) \cdot \log(1-\sigma(\hat{y}))]
$$

其中 $\sigma(\hat{y}) = 1/(1+e^{-\hat{y}})$。

梯度：
- $g = \sigma(\hat{y}) - y$
- $h = \sigma(\hat{y}) \cdot (1 - \sigma(\hat{y}))$

### 5.2 回归（MSE）

$$
l(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2
$$

梯度：
- $g = \hat{y} - y$
- $h = 1$

---

## 6. C 语言实现结构

### 6.1 数据结构

```c
// 树节点
typedef struct XGBoost_TreeNode {
    size_t feature_index;    // 分割特征
    double threshold;        // 分割阈值
    int is_leaf;             // 是否叶节点
    double weight;           // 叶节点权重 w*
    struct XGBoost_TreeNode* left;
    struct XGBoost_TreeNode* right;
} XGBoost_TreeNode;

// 树
typedef struct XGBoost_Tree {
    XGBoost_TreeNode* root;
    size_t depth;
    size_t n_leaves;
} XGBoost_Tree;

// 模型状态
typedef struct {
    XGBoost_Tree** trees;
    double* tree_weights;    // 学习率已乘入
    size_t n_trees;

    // 超参数
    double learning_rate;    // eta, 默认 0.3
    double reg_lambda;       // λ, 默认 1.0
    double reg_alpha;        // α, 默认 0.0
    double min_split_gain;   // γ, 默认 0.0
    double min_child_weight;  // 默认 1.0
    size_t max_depth;        // 默认 6
    double subsample;        // 行采样, 默认 1.0
    double colsample_bytree;  // 列采样, 默认 1.0
} XGBoost_State;
```

### 6.2 核心函数接口

```c
// 参照 machine_learning.h 的 ML_Model_Impl_t 接口
int XGBoost_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
                const dataset* ds, const size_t* feature_indices,
                size_t n_features, size_t target_index,
                const size_t* sample_indices, size_t n_samples);

int XGBoost_predict(const ML_Weights_t* state, const dataset* ds,
                   const size_t* feature_indices, size_t n_features,
                   const size_t* sample_indices, size_t n_samples,
                   void* output);

int XGBoost_predict_proba(const ML_Weights_t* state, const dataset* ds,
                         const size_t* feature_indices, size_t n_features,
                         const size_t* sample_indices, size_t n_samples,
                         size_t n_classes, void* output);

void XGBoost_free(ML_Weights_t* state);
```

### 6.3 实现要点

#### 梯度计算

```c
static void compute_gradients(double* g, double* h,
                              const double* predictions,
                              const int* labels,
                              size_t n_samples,
                              double(*loss_grad)(double pred, int label)) {
    for (size_t i = 0; i < n_samples; i++) {
        g[i] = loss_grad(predictions[i], labels[i]);
        h[i] = loss_hess(predictions[i], labels[i]);  // 二阶梯度
    }
}

// Logistic loss 梯度
static double logistic_grad(double pred, int label) {
    double p = 1.0 / (1.0 + exp(-pred));
    return p - label;
}

static double logistic_hess(double pred, int label) {
    double p = 1.0 / (1.0 + exp(-pred));
    return p * (1.0 - p);
}
```

#### 树节点分裂

```c
static double compute_split_gain(double G_L, double H_L,
                                 double G_R, double H_R,
                                 double lambda, double gamma) {
    double gain = 0.5 * (
        G_L * G_L / (H_L + lambda) +
        G_R * G_R / (H_R + lambda) -
        (G_L + G_R) * (G_L + G_R) / (H_L + H_R + lambda)
    ) - gamma;
    return gain;
}
```

#### 递归建树

```c
static XGBoost_TreeNode* build_tree(double* g, double* h,
                                    const dataset* ds,
                                    const size_t* indices,
                                    size_t n_samples,
                                    const XGBoost_State* params,
                                    size_t depth) {
    // 终止条件
    if (depth >= params->max_depth || n_samples < params->min_child_weight)
        return create_leaf_node(g, h, indices, n_samples, params);

    // 找最优分裂
    Split best_split = {0};
    double best_gain = 0;

    for (size_t f = 0; f < n_features; f++) {
        // 排序 + 遍历所有可能分裂点
        // ... 计算 G_L, H_L, G_R, H_R ...
        double gain = compute_split_gain(...);
        if (gain > best_gain) {
            best_gain = gain;
            best_split = ...;
        }
    }

    if (best_gain <= params->min_split_gain)
        return create_leaf_node(...);

    // 递归构建左右子树
    XGBoost_TreeNode* node = create_internal_node(best_split);
    node->left = build_tree(g, h, ds, left_indices, left_size, params, depth+1);
    node->right = build_tree(g, h, ds, right_indices, right_size, params, depth+1);
    return node;
}
```

#### 预测合并

```c
static double tree_predict(const XGBoost_TreeNode* node, const dataset* ds, size_t idx) {
    if (node->is_leaf)
        return node->weight;

    double val = ds->features[node->feature_index].data[idx];
    if (val <= node->threshold)
        return tree_predict(node->left, ds, idx);
    else
        return tree_predict(node->right, ds, idx);
}

static int XGBoost_predict(...) {
    XGBoost_State* xs = (XGBoost_State*)state->weights;

    for (size_t s = 0; s < n_samples; s++) {
        size_t idx = sample_indices[s];
        double output = 0.0;

        for (size_t t = 0; t < xs->n_trees; t++) {
            output += xs->tree_weights[t] * tree_predict(xs->trees[t]->root, ds, idx);
        }

        // 转换为类别或概率
        ((double*)output)[s] = 1.0 / (1.0 + exp(-output));  // sigmoid for binary
    }
    return 0;
}
```

---

## 7. 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `learning_rate` (eta) | 0.3 | 每棵树乘以的收缩因子 |
| `max_depth` | 6 | 树的最大深度 |
| `min_child_weight` | 1 | 叶节点最小样本权重和 |
| `subsample` | 1.0 | 行采样比例（每轮随机选部分样本） |
| `colsample_bytree` | 1.0 | 列采样比例（每轮随机选部分特征） |
| `reg_lambda` | 1.0 | L2 正则化系数 |
| `reg_alpha` | 0.0 | L1 正则化系数 |
| `min_split_gain` | 0.0 | 分裂所需最小增益 |

---

## 8. 与 adaboost.h 的模式对比

| AdaBoost (adaboost.h) | XGBoost |
|----------------------|---------|
| `AdaBoost_State` + `AdaBoost_WeakLearner` | `XGBoost_State` + `XGBoost_Tree` |
| `stump_fit()` — 单特征/阈值搜索 | `build_tree()` — 多特征贪心搜索 |
| `stump_predict()` — 阈值判断 | `tree_predict()` — 递归遍历 |
| `alpha[t] = 0.5 * log((1-e)/e)` | `weight = -Σg/(Σh + λ)` |
| `weights[i] *= exp(-alpha * y * pred)` | 预计算 g/h，梯度下降更新 |

---

## 9. 进一步优化方向

1. **近似算法**：大数据集下按特征值分桶，减少排序开销
2. **缓存优化**：预排序样本索引，提升随机访问效率
3. **剪枝**：后剪枝（post-pruning）替代深度限制
4. **早停**：监控验证集 loss，连续 N 轮无改善则停止
5. **稀疏感知**：处理缺失值特征（XGBoost 默认将缺失值分配到增益最大方向）