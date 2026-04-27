# CatBoost 原理与 C 语言实现指南

基于论文 "CatBoost: unbiased boosting with categorical features" (Dorogush, Ershov, Gulin, 2018) 及 cML 框架。

---

## 1. CatBoost 核心创新

| 特性 | 描述 | 解决的问题 |
|------|------|-----------|
| **Ordered Boosting** | 使用随机排列避免预测偏移 | 目标泄露（target leakage） |
| **对称树（Oblivious Trees）** | 同一深度的叶节点数量相同 | 训练/推理效率，内存优化 |
| **目标编码（Target Statistics）** | 对类别特征计算条件均值 | 高基数类别特征 |
| **内置随机排列** | 每棵树使用不同的 permutation | 防止信息泄露 |

---

## 2. 预测偏移问题（Prediction Shift）

### 2.1 问题根源

标准梯度提升存在**目标泄露**：

```
第 t 轮训练时：
  - 样本 i 的梯度 g_i 依赖于模型在第 t-1 轮对样本 i 的预测
  - 而这个预测本身是用包含了样本 i 目标值的数据计算的
  → 模型过度拟合训练数据的梯度
```

### 2.2 CatBoost 的解决方案：Ordered Boosting

对于样本 $i$，仅使用**排在她之前**的样本计算梯度：

```
对样本 i 的梯度：
  只使用 permutation 中排在 i 之前的样本来计算
  → 避免用"未来"的信息
```

数学表达：
$$
f_t(x_i) \text{ 仅依赖于 } \{y_j \mid \sigma(j) < \sigma(i)\}
$$

---

## 3. 目标统计（Target Statistics）

### 3.1 定义

对于类别特征 $k$ 和样本 $i$：

$$
\hat{y}_i^{(k)} = \frac{\sum_{j: x_j^{(k)} = x_i^{(k)}} \mathbf{1}_{\sigma(j) < \sigma(i)} y_j + a \cdot prior}{\sum_{j: x_j^{(k)} = x_i^{(k)}} \mathbf{1}_{\sigma(j) < \sigma(i)} + a}
$$

其中：
- $prior$：先验值（通常是全局目标均值）
- $a$：平滑参数（默认 1）
- $\sigma$：随机排列

### 3.2 计算过程

```
对于每个类别值 c:
    收集所有 x_j = c 的样本索引
    只使用 permutation 中靠前的样本计算均值
```

### 3.3 数值特征处理

CatBoost 将数值特征离散化为桶，视为有序类别特征。

---

## 4. 对称树（Oblivious Trees）

### 4.1 树结构特点

```
深度 0:                    [condition]
                        /                \
深度 1:             [cond]              [cond]
              /         \          /         \
深度 2:       [cond] [cond]       [cond] [cond]
           /  \      /  \       /   \    /   \
深度 3:   L    L    L    L      L     L   L    L
```

同一深度的所有内部节点使用**相同条件**，所有叶节点深度相同。

### 4.2 叶子权重

$$
w_j = \frac{\sum_{i \in \text{leaf}_j} y_i}{\text{count}_j}
$$

（也可以使用更复杂的公式，包含先验平滑）

### 4.3 优势

| 优势 | 说明 |
|------|------|
| **内存效率** | 叶节点数量固定为 $2^d$，可压缩存储 |
| **推理速度** | 只需执行 $d$ 次比较（$d$ = 深度） |
| **正则化** | 对称结构天然防止过拟合 |
| **可解释性** | 特征重要性更清晰 |

---

## 5. CatBoost 训练算法

### 5.1 完整算法

```
输入: 数据 D = {(xi, yi)}, 迭代次数 T, 排列数 M
输出: M 棵对称树的集合

for m = 1 to M do:
    # Step 1: 生成随机排列
    σ = random_permutation(D)

    # Step 2: 计算有序目标统计（用于类别特征）
    for each categorical feature k:
        for each category c in feature k:
            TS[k][c] = compute_target_statistics(D, σ, k, c)

    # Step 3: 构建对称树
    tree = build_oblivious_tree(TS, gradients)

    # Step 4: 更新预测
    update_predictions(tree)

    # Step 5: 计算梯度用于下一轮
    compute_gradients(new_predictions, y)
```

### 5.2 梯度计算

CatBoost 支持多种损失函数，二分类使用 Logistic Loss：

$$
g_i = \sigma(\hat{y}_i) - y_i
$$

其中 $\sigma(x) = 1/(1+e^{-x})$

---

## 6. C 语言实现结构

### 6.1 数据结构

```c
// 类别特征信息
typedef struct {
    size_t n_categories;        // 该特征的唯一值数量
    size_t* category_indices;   // 每个类别对应的样本索引数组
    double* category_stats;      // 每个类别的目标统计值
} CategoricalFeature;

// 对称树节点
typedef struct CatBoost_TreeNode {
    size_t feature_index;       // 分割特征
    double threshold;            // 分割阈值（或桶索引）
    int is_leaf;                // 是否叶节点
    double weight;              // 叶节点权重（预测值）
    // 对称树不需要左右指针，深度相同
} CatBoost_TreeNode;

// 对称树（所有叶节点深度相同）
typedef struct {
    CatBoost_TreeNode* nodes;   // 节点数组（按层存储）
    size_t depth;               // 树深度
    size_t n_leaves;            // 叶节点数量 = 2^depth
} CatBoost_Tree;

// 模型状态
typedef struct {
    CatBoost_Tree** trees;
    double* tree_weights;       // 乘以学习率
    size_t n_trees;
    size_t n_features;          // 原始特征数
    size_t n_categorical;      // 类别特征数

    // 类别特征映射
    CategoricalFeature* cat_features;
    size_t n_cat_features;

    // 超参数
    double learning_rate;       // 默认 0.03
    double l2_leaf_reg;        // L2 正则化，默认 3.0
    double prior;              // 先验值，默认 0.5
    double smoothing;           // 平滑参数 a，默认 1.0
    size_t depth;              // 树深度，默认 6
    size_t n permutations;      // 排列数，默认 1
} CatBoost_State;
```

### 6.2 核心函数接口

```c
// 参照 adaboost.h 的 ML_Model 接口
int CatBoost_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
                const dataset* ds, const size_t* feature_indices,
                size_t n_features, size_t target_index,
                const size_t* sample_indices, size_t n_samples);

int CatBoost_predict(const ML_Weights_t* state, const dataset* ds,
                     const size_t* feature_indices, size_t n_features,
                     const size_t* sample_indices, size_t n_samples,
                     void* output);

int CatBoost_predict_proba(const ML_Weights_t* state, const dataset* ds,
                          const size_t* feature_indices, size_t n_features,
                          const size_t* sample_indices, size_t n_samples,
                          size_t n_classes, void* output);

void CatBoost_free(ML_Weights_t* state);

static ML_Model_t create_catboost_model(void);
```

---

## 7. 实现要点

### 7.1 有序目标统计计算

```c
static double compute_target_statistics(const dataset* ds,
                                        const size_t* perm_order,
                                        size_t n_samples,
                                        double prior,
                                        double smoothing,
                                        size_t cat_idx,
                                        size_t category_value) {
    double sum_y = 0.0;
    size_t count = 0;

    // 遍历所有样本，找同类别且排在当前样本之前的
    for (size_t pos = 0; pos < n_samples; pos++) {
        size_t sample_idx = perm_order[pos];

        // 检查是否为同一类别
        if (ds->features[cat_idx].data[sample_idx] == category_value) {
            // 只使用排在当前位置之前的样本
            // ... 这需要额外的数据结构或多次遍历
        }
    }

    return (sum_y + smoothing * prior) / (count + smoothing);
}
```

**优化实现**：预先按类别分组，维护每个类别的样本有序列表。

### 7.2 对称树构建

```c
// 简化的对称树：每层使用同一个分割条件
static CatBoost_Tree* build_oblivious_tree(double* gradients,
                                           const dataset* ds,
                                           const size_t* indices,
                                           size_t n_samples,
                                           const CatBoost_State* params) {
    size_t n_leaves = 1 << params->depth;  // 2^depth

    CatBoost_Tree* tree = (CatBoost_Tree*)malloc(sizeof(CatBoost_Tree));
    tree->depth = params->depth;
    tree->n_leaves = n_leaves;
    tree->nodes = (CatBoost_TreeNode*)malloc(sizeof(CatBoost_TreeNode) * (2 * n_leaves - 1));

    // 递归划分：每层使用最优特征和阈值
    size_t node_idx = 0;
    size_t* current_indices = (size_t*)malloc(sizeof(size_t) * n_samples);
    size_t* left_indices = (size_t*)malloc(sizeof(size_t) * n_samples);
    size_t* right_indices = (size_t*)malloc(sizeof(size_t) * n_samples);

    memcpy(current_indices, indices, sizeof(size_t) * n_samples);
    size_t current_size = n_samples;

    for (size_t depth = 0; depth < params->depth; depth++) {
        size_t n_internal_nodes = 1 << depth;
        double best_gain = -1e300;
        size_t best_feature = 0;
        double best_threshold = 0;

        // 遍历所有特征找最优分割
        for (size_t f = 0; f < ds->num_features; f++) {
            // 对当前节点集合找最优阈值
            double gain = find_best_split(ds, current_indices, current_size,
                                          f, gradients, params, &best_threshold);

            if (gain > best_gain) {
                best_gain = gain;
                best_feature = f;
            }
        }

        // 设置当前深度的所有节点（对称树：同一深度相同条件）
        for (size_t n = 0; n < n_internal_nodes; n++) {
            tree->nodes[node_idx + n].feature_index = best_feature;
            tree->nodes[node_idx + n].threshold = best_threshold;
            tree->nodes[node_idx + n].is_leaf = 0;
        }

        // 分割样本
        split_samples(ds, current_indices, current_size,
                      best_feature, best_threshold,
                      left_indices, right_indices, &left_size, &right_size);

        // 准备下一轮（所有节点用相同的分割）
        memcpy(current_indices, left_indices, sizeof(size_t) * left_size);
        memcpy(current_indices + left_size, right_indices, sizeof(size_t) * right_size);
        current_size = left_size + right_size;

        node_idx += n_internal_nodes;
    }

    // 设置叶节点权重
    size_t leaf_start = (1 << params->depth) - 1;
    for (size_t i = 0; i < n_leaves; i++) {
        // 计算该叶子的平均梯度作为权重
        tree->nodes[leaf_start + i].is_leaf = 1;
        tree->nodes[leaf_start + i].weight = compute_leaf_weight(
            gradients, current_indices, current_size, params);
    }

    free(current_indices);
    free(left_indices);
    free(right_indices);

    return tree;
}
```

### 7.3 对称树预测

```c
static double catboost_tree_predict(const CatBoost_Tree* tree,
                                    const dataset* ds,
                                    size_t sample_idx) {
    size_t node = 0;

    while (!tree->nodes[node].is_leaf) {
        double val = ds->features[tree->nodes[node].feature_index].data[sample_idx];
        if (val <= tree->nodes[node].threshold) {
            node = node * 2 + 1;  // 左子节点
        } else {
            node = node * 2 + 2;  // 右子节点
        }
    }

    return tree->nodes[node].weight;
}
```

---

## 8. CatBoost vs XGBoost vs AdaBoost

| 特性 | AdaBoost | XGBoost | CatBoost |
|------|----------|---------|----------|
| **树结构** | 单层树桩 | 任意二叉树 | 对称树（Oblivious） |
| **分裂方式** | 单特征贪心 | 多特征贪心 | 多特征贪心（对称约束） |
| **目标泄露** | 存在 | 存在 | 通过 Ordered 机制避免 |
| **类别特征** | 需编码 | 需编码 | 内置目标统计 |
| **正则化** | 无明确机制 | L1/L2 | 对称树 + L2 |
| **学习率** | 无 | 可选 | 可选 |
| **梯度类型** | 指数损失 | 一阶/二阶 | 一阶/二阶 |

---

## 9. 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `learning_rate` | 0.03 | 学习率 |
| `depth` | 6 | 对称树深度 |
| `l2_leaf_reg` | 3.0 | L2 正则化 |
| `smoothing` | 1.0 | 目标统计平滑参数 |
| `prior` | 0.5 | 先验概率 |
| `iterations` | 1000 | 迭代次数 |
| `random_strength` | 1.0 | 随机强度（用于排序） |
| `bootstrap_type` | "Bayesian" | 采样方式 |

---

## 10. 与 adaboost.h 的模式对比

| AdaBoost (adaboost.h) | CatBoost |
|----------------------|----------|
| `AdaBoost_State` + `AdaBoost_WeakLearner` | `CatBoost_State` + `CatBoost_Tree` |
| `stump_fit()` — 单层决策树 | `build_oblivious_tree()` — 对称多叉树 |
| 样本权重重分配 | 有序梯度计算 |
| 无类别特征处理 | 内置目标统计 |
| `alpha[t] = 0.5 * log((1-e)/e)` | `weight = avg(gradients_in_leaf)` |

---

## 11. 实现优先级建议

1. **Phase 1**: 先实现基础对称树结构和 Ordered Boosting
2. **Phase 2**: 添加类别特征目标统计
3. **Phase 3**: 实现多排列（Multi-pass ordering）
4. **Phase 4**: 添加高级正则化和剪枝