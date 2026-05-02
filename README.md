# cML - C 语言机器学习库

纯 C 实现的机器学习算法库，支持 OpenCL GPU 加速的深度学习。

## 项目结构

```
cML/
├── dl/                   # 深度学习
│   ├── mlp.h             # 多层感知机
│   ├── tensor.h          # 张量运算
│   ├── opencl_mlp.h      # OpenCL 加速 MLP
│   ├── opencl_tensor.h   # OpenCL 张量运算
│   ├── lenet5.h          # LeNet-5 卷积神经网络
│   ├── lstm.h            # 长短期记忆网络
│   ├── rnn.h             # 循环神经网络
│   └── test/             # 深度学习测试
├── ensemble/             # 集成学习
│   ├── adaboost.h        # AdaBoost
│   ├── randomforest.h    # 随机森林
│   ├── xgboost.h         # XGBoost
│   └── catboost.h        # CatBoost
├── linear/               # 线性模型
│   ├── mlr.h             # 多元线性回归
│   ├── ridge_regression.h      # 岭回归（L2正则化）
│   ├── polynomial_regression.h # 多项式回归
│   ├── softmax_regression.h    # Softmax 回归
│   ├── linear_algebra.h  # 矩阵运算
│   └── wls.h             # 加权最小二乘
├── test/                 # 通用 ML 测试
├── common/               # 通用工具
│   ├── dataset.h         # 数据集结构
│   ├── csv.h             # CSV 解析
│   ├── machine_learning.h      # 统一 ML 接口
│   └── utilities.h       # 辅助函数
├── idx.h                 # MNIST idx 文件格式
├── gnb.h                 # 高斯朴素贝叶斯
└── decision_tree.h       # 决策树
```

## 功能特性

- **深度学习**: MLP、CNN (LeNet-5)、RNN、LSTM，支持 OpenCL GPU 加速
- **集成学习**: AdaBoost、随机森林、XGBoost、CatBoost
- **线性模型**: MLR、岭回归、多项式回归、Softmax、WLS
- **纯 C 实现**: 除标准 C 库外无外部依赖
- **OpenCL 支持**: 张量运算 GPU 加速

## 快速开始

```c
#define CSV_IMPLEMENTATION
#include "csv.h"

#define DATASET_IMPLEMENTATION
#include "dataset.h"

#define SOFTMAX_REGRESSION_IMPLEMENTATION
#include "softmax_regression.h"

// 加载数据
csv_t* csv = csv_load("data.csv");
const char* labels[] = {"label_col"};
dataset* ds = csv_to_dataset(csv, labels, 1);

// 创建并训练模型
ML_Model_t model = create_softmax_model();
model.methods.fit(&model.config, &model.state, ds, ...);

// 预测
int predictions[100];
model.methods.predict(&model.state, test_ds, ..., predictions);
```

## 编译

```bash
make all          # 编译所有目标
make test         # 运行基本测试
make test_dl      # 运行深度学习测试
```

## 文档

### English Documentation | 英文文档
- [Getting Started](docs/getting-started.md)
- [Common Utilities](docs/common/)
- [Linear Models](docs/linear/)
- [Ensemble Methods](docs/ensemble/)
- [Deep Learning](docs/dl/)
- [API Reference](docs/api/full_api_reference.md)

### 中文文档
- [快速开始](docs-zh/getting-started.md)
- [通用工具](docs-zh/common/)
- [线性模型](docs-zh/linear/)
- [集成学习](docs-zh/ensemble/)
- [深度学习](docs-zh/dl/)
- [API 参考](docs-zh/api/full_api_reference.md)
