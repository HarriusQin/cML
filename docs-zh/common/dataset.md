# 数据集

面向列的数据结构，用于机器学习。

## 概述

`dataset` 结构按列组织数据而非按行，这对 ML 操作更高效，因为通常需要访问特定特征的所有值。

## 数据结构

### feature_column

```c
typedef struct {
    size_t num_samples;
    double* data;           // 原始数据值
    double min_val;
    double max_val;
    double mean;
    double std_dev;
} feature_column;
```

### label_column

```c
typedef struct {
    size_t num_samples;
    size_t* labels;         // 整数类索引
    size_t classes;         // 唯一类别数
    char** value_map;       // 索引到字符串标签的映射
    size_t* class_counts;   // 每类数量
} label_column;
```

### dataset

```c
typedef struct {
    size_t rows;                     // 样本数
    size_t num_features;             // 特征列数
    size_t num_labels;              // 标签列数
    feature_column* features;       // [num_features]
    label_column* labels;           // [num_labels]
} dataset;
```

## 函数

### 创建和销毁

```c
dataset* create_dataset(size_t num_samples, size_t num_features, size_t num_labels);
void free_dataset(dataset* ds);
```

### CSV 转换

```c
dataset* csv_to_dataset(csv_t* csv, const char** label_cols, size_t num_labels);
```

### 特征访问

```c
double dataset_get_feature(const dataset* ds, size_t feature_idx, size_t sample_idx);
void dataset_set_feature(dataset* ds, size_t feature_idx, size_t sample_idx, double value);
```

### 标签访问

```c
size_t dataset_get_label(const dataset* ds, size_t label_idx, size_t sample_idx);
const char* dataset_get_label_string(const dataset* ds, size_t label_idx, size_t sample_idx);
```

### 训练/测试集划分

```c
typedef struct {
    size_t* train_indices;
    size_t* test_indices;
    size_t train_size;
    size_t test_size;
} Dataset_Split_t;

void train_test_split(const dataset* ds, float test_ratio, unsigned int seed,
                      Dataset_Split_t* split);
```

### 特征缩放

```c
typedef enum {
    SCALING_STANDARD,    // (x - mean) / std_dev
    SCALING_MINMAX,      // (x - min) / (max - min)
    SCALING_NONE
} ScalingType;

ML_ScalingParams_t* ml_fit_scaling(const dataset* ds, size_t* feat_idx,
                                    size_t n_features, size_t* indices,
                                    size_t n_samples, ScalingType type);

dataset* ml_transform_features(const ML_ScalingParams_t* params,
                              const dataset* ds, size_t* feat_idx,
                              size_t n_features, size_t* indices,
                              size_t n_samples);

void ml_free_scaling_params(ML_ScalingParams_t* params);
```

## 示例

```c
#define CSV_IMPLEMENTATION
#include "csv.h"

#define DATASET_IMPLEMENTATION
#include "dataset.h"

// 加载 CSV
csv_t* csv = csv_load("iris.csv");

// 转换为数据集
const char* labels[] = {"species"};
dataset* ds = csv_to_dataset(csv, labels, 1);

// 访问特征
printf("样本 0, 特征 0: %f\n",
       dataset_get_feature(ds, 0, 0));

// 划分数据
Dataset_Split_t split;
train_test_split(ds, 0.2, 42, &split);

// 缩放特征
ML_ScalingParams_t* scaler = ml_fit_scaling(ds, NULL, ds->num_features,
                                            split.train_indices,
                                            split.train_size,
                                            SCALING_STANDARD);
```

## 说明

- 特征按列存储以提高访问效率
- 标签自动转换为整数索引
- 字符串标签存储在 `value_map` 中以便解释
