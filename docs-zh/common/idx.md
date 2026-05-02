# IDX 文件格式

用于 MNIST 及类似数据集的 IDX（索引二进制）格式读取器。

## 概述

IDX 是为 MNIST 手写数字数据库开发的简单二进制格式。支持多种数值类型和任意维度。

## 文件格式

```
[魔数字节] [维度信息] [数据...]
```

### 魔数

```
+------++----------++------------+
| type || n_dims  ||  零字节    |
+------++----------++------------+
  1 字节    1 字节    1 字节
```

- **type**: 数据类型码
  - `0x08`: 无符号字节 (uint8)
  - `0x09`: 有符号字节 (int8)
  - `0x0B`: 短整型 (int16)
  - `0x0C`: 整型 (int32)
  - `0x0D`: 浮点型 (float32)
  - `0x0E`: 双精度浮点型 (float64)

- **n_dims**: 维度数 (0-255)

- **零字节**: 保留字节（通常为 0）

### 维度编码

维度以 32 位大端无符号整数存储。

MNIST 图像示例 (60000 × 28 × 28):
```
魔数:     0x00000803  (type=UBYTE, n_dims=3)
维度 0:   60000       (图像数量)
维度 1:   28          (行数)
维度 2:   28          (列数)
```

## 数据结构

```c
typedef enum {
    IDX_UBYTE = 0x08,   // uint8
    IDX_BYTE  = 0x09,   // int8
    IDX_SHORT = 0x0B,   // int16
    IDX_INT   = 0x0C,    // int32
    IDX_FLOAT = 0x0D,    // float32
    IDX_DOUBLE = 0x0E     // float64
} IDX_TYPE;

typedef struct cIDX {
    IDX_TYPE type;        // 数据类型
    uint8_t n_dims;       // 维度数
    uint32_t* dims;       // 各维度大小
    void* idx_data;       // 原始数据指针
    size_t size;          // 总字节数
} cIDX;
```

## 函数

```c
// 加载 IDX 文件
static cIDX* idx_load(const char* path);

// 释放 IDX 结构
static void free_idx(cIDX* idx);

// 获取元素字节大小
static size_t idx_elem_size(IDX_TYPE type);

// 检查字节序
static int is_little_endian(void);
```

## 字节序处理

IDX 格式使用大端字节序。在小端系统上自动进行字节交换：

```c
static void byte_swap(void* data, size_t elem_size, size_t count) {
    uint8_t* p = (uint8_t*)data;
    for (size_t i = 0; i < count; i++) {
        // 反转每个元素内的字节
        for (size_t j = 0; j < elem_size / 2; j++) {
            uint8_t tmp = p[i * elem_size + j];
            p[i * elem_size + j] = p[i * elem_size + elem_size - 1 - j];
            p[i * elem_size + elem_size - 1 - j] = tmp;
        }
    }
}
```

## MNIST 数据集

MNIST 数据库包含四个 IDX 文件：

| 文件 | 类型 | 形状 | 描述 |
|------|------|-------|-------------|
| train-images-idx3-ubyte | UBYTE | [60000, 28, 28] | 训练图像 |
| train-labels-idx1-ubyte | UBYTE | [60000] | 训练标签 |
| t10k-images-idx3-ubyte | UBYTE | [10000, 28, 28] | 测试图像 |
| t10k-labels-idx1-ubyte | UBYTE | [10000] | 测试标签 |

## 示例

```c
#define IDX_IMPLEMENTATION
#include "idx.h"

// 加载 MNIST 图像
cIDX* train_images = idx_load("data/train-images-idx3-ubyte");
// 返回: type=IDX_UBYTE, n_dims=3
// dims=[60000, 28, 28]
// 访问图像 0, 第 10 行, 第 15 列的像素:
// uint8_t pixel = ((uint8_t*)train_images->idx_data)[0 * 28 * 28 + 10 * 28 + 15];

cIDX* train_labels = idx_load("data/train-labels-idx1-ubyte");
// 返回: type=IDX_UBYTE, n_dims=1
// dims=[60000]
// 访问标签 0:
// uint8_t label = ((uint8_t*)train_labels->idx_data)[0];

// 使用数据...
printf("加载了 %u 张 %ux%u 的图像\n",
       train_images->dims[0],
       train_images->dims[1],
       train_images->dims[2]);

// 释放
free_idx(train_images);
free_idx(train_labels);
```

## 元素大小

| 类型 | 码 | 大小（字节） |
|------|------|--------------|
| IDX_UBYTE | 0x08 | 1 |
| IDX_BYTE | 0x09 | 1 |
| IDX_SHORT | 0x0B | 2 |
| IDX_INT | 0x0C | 4 |
| IDX_FLOAT | 0x0D | 4 |
| IDX_DOUBLE | 0x0E | 8 |

## 说明

- IDX 文件始终是大端（MSB 在前）
- 小端系统自动转换
- 数据以行主序存储
- 3D MNIST 数据：[图像, 行, 列]
- 标签通常存储为标量值（1D 数组）
- UBYTE 类型像素值范围为 [0, 255]
