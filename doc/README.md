# 机器学习模型库 (Machine Learning Library)

## 概述

本库提供三种常用的机器学习模型，使用纯C语言实现，具有轻量级、无外部依赖的特点。

### 支持的模型

| 模型 | 类型 | 用途 |
|------|------|------|
| **高斯朴素贝叶斯 (GNB)** | 分类 | 文本分类、疾病诊断、垃圾邮件检测 |
| **随机森林 (RF)** | 分类/回归 | 金融预测、医学诊断、特征选择 |
| **K近邻 (KNN)** | 分类/回归 | 推荐系统、图像识别、异常检测 |

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `ml.h` | 机器学习模型头文件，包含所有函数声明和实现 |
| `test_ml.c` | 测试程序，演示所有模型的使用方法 |
| `iris.csv` | Iris数据集（150个样本，4个特征，3个类别） |
| `dataset.h` | 数据集处理库（依赖） |
| `csv.h` | CSV解析库（依赖） |

---

## 编译与运行

### 编译

```bash
# 直接编译（需要链接数学库）
gcc -o test_ml test_ml.c -lm

# 或者使用Make（如果Makefile存在）
make test_ml
```

### 运行

```bash
./test_ml
```

---

## 模型详解

### 1. 高斯朴素贝叶斯 (Gaussian Naive Bayes)

#### 算法原理

高斯朴素贝叶斯是基于贝叶斯定理的分类算法，假设所有特征条件独立且服从高斯（正态）分布。

**贝叶斯定理：**
```
P(y=c|x) = P(x|y=c) * P(y=c) / P(x)
```

**高斯概率密度函数：**
```
P(x|μ,σ²) = (1/√(2πσ²)) * exp(-(x-μ)²/2σ²)
```

#### 优点
- 训练和预测速度快
- 对小规模数据集效果好
- 对特征之间的独立性假设在某些场景下反而能防止过拟合

#### 缺点
- 假设特征条件独立，往往不成立
- 对于连续特征，假设高斯分布可能不准确

#### 使用示例

```c
#include "ml.h"

// 加载数据
csv_t* csv = csv_load("iris.csv");
dataset* ds = csv_to_dataset(csv, (const char*[]){"species"}, 1);

// 训练模型
model* gnb = train_gnb(ds, 0, 0, default_config);

// 预测
double features[] = {5.1, 3.5, 1.4, 0.2};
int prediction = predict_gnb(gnb, features, 4);

// 评估
double acc = accuracy(gnb, ds, 0, 0);
printf("准确率: %.2f%%\n", acc * 100);

// 释放资源
free_gnb(gnb);
free(gnb);
free_dataset(ds);
free(ds);
free_csv_data(csv);
free(csv);
```

---

### 2. 随机森林 (Random Forest)

#### 算法原理

随机森林是一种集成学习方法，通过构建多棵决策树并进行集成预测：

1. **Bootstrap采样**：从n个样本中有放回地抽取n个样本
2. **随机特征选择**：每棵树只使用部分特征进行分裂
3. **集成预测**：
   - 分类：投票法（多数票获胜）
   - 回归：平均法（所有树预测值的平均）

#### 决策树分裂标准

**分类**：使用基尼不纯度 (Gini Impurity)
```
Gini = 1 - Σ p_i²
```

**信息增益**：
```
Gain = Gini(parent) - (n_left/n)*Gini(left) - (n_right/n)*Gini(right)
```

#### 优点
- 可以处理高维数据
- 不容易过拟合
- 可以处理缺失值
- 提供特征重要性评估
- 可以同时处理分类和回归问题

#### 缺点
- 对于有大量噪声的数据可能过拟合
- 训练时间较长（尤其是树的数量多时）
- 模型可解释性不如单棵决策树

#### 关键参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `n_estimators` | 树的数量 | 50-500 |
| `max_depth` | 树的最大深度 | 5-20（不限制则填0） |
| `learning_rate` | 子采样比例 | 0.5-1.0 |

#### 使用示例

```c
// 分类
model_config config = default_config;
config.n_estimators = 100;
config.max_depth = 10;

model* rf = train_rf(ds, 0, 0, config);
int pred = (int)predict_rf(rf, features, 4);
double acc = accuracy(rf, ds, 0, 0);

// 回归
model* rf_reg = train_rf(ds_reg, 0, 0, config);
double pred_value = predict_rf(rf_reg, features, 4);
double r2 = r2_score(rf_reg, ds_reg, 0, 0);
double mse_val = mse(rf_reg, ds_reg, 0, 0);
```

---

### 3. K近邻 (K-Nearest Neighbors)

#### 算法原理

KNN是一种惰性学习算法，没有显式的训练过程，预测时直接计算距离。

**预测过程：**
1. 计算待预测样本与所有训练样本的距离（默认使用欧氏距离）
2. 找出距离最近的K个样本
3. 集成预测结果：
   - 分类：多数投票
   - 回归：平均值

**欧氏距离：**
```
d(x, y) = √(Σ(x_i - y_i)²)
```

#### 优点
- 简单直观，易于理解
- 对数据分布没有假设
- 可以同时处理分类和回归
- 适合多分类问题

#### 缺点
- 预测时间复杂度高 O(nd)，不适合大规模数据
- 对K值敏感（K太小容易过拟合，K太大容易欠拟合）
- 对特征尺度敏感（需要标准化）
- 对噪声数据敏感

#### K值选择

- **K太小**：容易受到噪声影响，可能过拟合
- **K太大**：容易欠拟合，决策边界平滑
- **经验法则**：K ≈ √n（n为样本数）

#### 使用示例

```c
// 配置KNN参数
model_config config = default_config;
config.k_value = 5;

// 分类
model* knn = train_knn(ds, 0, 0, config);
int pred = (int)predict_knn(knn, features, 4);
double acc = accuracy(knn, ds, 0, 0);

// 回归
model* knn_reg = train_knn(ds_reg, 0, 0, config);
double pred_value = predict_knn(knn_reg, features, 4);
double r2 = r2_score(knn_reg, ds_reg, 0, 0);
```

---

## 评估指标

### 分类指标

| 指标 | 说明 | 公式 |
|------|------|------|
| **准确率 (Accuracy)** | 正确预测的比例 | Accuracy = TP + TN / (TP + TN + FP + FN) |

### 回归指标

| 指标 | 说明 | 公式 |
|------|------|------|
| **R²决定系数** | 模型解释的方差比例 | R² = 1 - Σ(y-ŷ)²/Σ(y-ȳ)² |
| **MSE均方误差** | 预测误差的平方平均 | MSE = (1/n)Σ(y-ŷ)² |

---

## 数据集格式

本库使用 `dataset.h` 中定义的列式存储结构：

```
数据集结构：
features[0]: [f00, f01, f02, ..., f0n]  <- 第一个特征的所有样本值
features[1]: [f10, f11, f12, ..., f1n]  <- 第二个特征的所有样本值
...
features[m]: [fm0, fm1, fm2, ..., fmn]  <- 第m个特征的所有样本值
labels[0]:   [l00, l01, l02, ..., l0n]  <- 第一个标签的所有样本值

访问方式：
- 样本i的特征j: ds->features[j].data[i]
- 样本i的标签: ds->labels[0].labels[i]
```

---

## 内存管理

使用本库时需要注意内存的分配和释放：

```c
// 1. 创建模型后使用
model* gnb = train_gnb(ds, 0, 0, default_config);

// 2. 使用模型进行预测
int pred = predict_gnb(gnb, features, 4);

// 3. 释放模型（先释放内部数据，再释放模型结构）
free_gnb(gnb);  // 释放内部数据
free(gnb);      // 释放模型结构本身

// 4. 释放数据集
free_dataset(ds);
free(ds);

// 5. 释放CSV数据
free_csv_data(csv);
free(csv);
```

或者使用通用的 `model_free()` 函数自动释放：

```c
model_free(gnb);  // 自动识别模型类型并释放
free(gnb);
```

---

## 注意事项

1. **特征标准化**：KNN对特征尺度敏感，建议在使用前进行标准化
2. **K值选择**：可以通过交叉验证选择最优的K值
3. **随机森林的随机性**：由于使用了随机采样，每次训练结果可能略有不同
4. **数据维度**：高维数据会导致距离计算变得不那么有意义（维度诅咒）

---

## 扩展建议

如需扩展本库，可以考虑添加以下功能：

1. **特征标准化函数**
2. **交叉验证框架**
3. **更多评估指标**（Precision, Recall, F1-Score等）
4. **模型持久化**（保存/加载模型）
5. **并行化训练**（多线程/多进程）
6. **更多模型**（SVM, 决策树回归, 梯度提升等）
