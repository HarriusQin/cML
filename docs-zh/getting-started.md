# 快速开始

## 编译项目

```bash
make all
```

## 运行测试

```bash
make test
```

## 使用示例

### 加载 CSV 数据

```c
#define CSV_IMPLEMENTATION
#include "csv.h"

csv_t* csv = csv_load("data.csv");
```

### 创建数据集

```c
#define DATASET_IMPLEMENTATION
#include "dataset.h"

dataset* ds = csv_to_dataset(csv, labels, 1);
```

### 训练模型

```c
#define SOFTMAX_REGRESSION_IMPLEMENTATION
#include "softmax_regression.h"

ML_Model_t model = create_softmax_model();
model.methods.fit(&model.config, &model.state, ds, ...);
```

## 文档语言

- [English](../docs/index.md)
- [中文](index.md)
