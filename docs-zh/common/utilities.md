# 工具函数

决策树算法的辅助工具。

## 概述

utilities 模块提供了基于树的算法（随机森林、AdaBoost、XGBoost、CatBoost）使用的辅助函数和比较函数，用于高效的分裂点查找。

## FeatureSample

用于基于排序的分裂点查找：

```c
typedef struct {
    size_t idx;  // 数据集中的样本索引
    double val; // 该样本的特征值
} FeatureSample;
```

每个条目将样本索引与其特征值配对，从而能够高效排序以找到候选分裂阈值。

## 比较函数

```c
static int cmp_FeatureSample(const void* a, const void* b) {
    double da = ((const FeatureSample*)a)->val;
    double db = ((const FeatureSample*)b)->val;
    return (da > db) - (da < db);
}
```

与 `qsort()` 一起使用，按特征值升序排序。

## 分裂点查找算法

对于给定特征，找到最佳分裂点：

```c
static double find_best_split(
    const FeatureSample* samples,
    size_t n_samples,
    const double* targets,
    double* thresholds,
    size_t* left_indices,
    size_t* right_indices,
    double* left_target_sum,
    double* right_target_sum) {

    // 按特征值排序
    qsort(samples, n_samples, sizeof(FeatureSample), cmp_FeatureSample);

    // 在唯一阈值处尝试分裂
    double best_gain = -INFINITY;
    double best_threshold = 0;

    for (size_t i = 0; i < n_samples - 1; i++) {
        // 仅考虑唯一值（跳过重复）
        if (samples[i].val == samples[i + 1].val) continue;

        double threshold = (samples[i].val + samples[i + 1].val) / 2;

        // 计算左右侧和
        double left_sum = 0, right_sum = 0;
        size_t left_count = 0, right_count = 0;

        for (size_t j = 0; j < i + 1; j++) {
            left_sum += targets[samples[j].idx];
            left_count++;
        }
        for (size_t j = i + 1; j < n_samples; j++) {
            right_sum += targets[samples[j].idx];
            right_count++;
        }

        // 计算信息增益（用于分类）
        // ...熵计算...

        if (gain > best_gain) {
            best_gain = gain;
            best_threshold = threshold;
        }
    }

    return best_threshold;
}
```

## 在决策树中的用法

```c
// 对每个特征，找到最佳分裂
for (size_t f = 0; f < n_features; f++) {
    FeatureSample* samples = malloc(n_samples * sizeof(FeatureSample));

    // 构建特征样本数组
    for (size_t i = 0; i < n_samples; i++) {
        samples[i].idx = indices[i];
        samples[i].val = dataset->features[f].data[samples[i].idx];
    }

    // 找到此特征的最佳分裂
    double threshold = find_best_split(samples, n_samples, targets,
                                     thresholds, left, right,
                                     left_sum, right_sum);

    free(samples);
}
```

## 说明

- 基于排序的分裂查找是精确的（找到最优分裂）
- 替代方案：基于直方图的分裂以获得更快的近似分裂
- 对于大数据集，可以对特征和/或样本进行子采样
- `FeatureSample` 是轻量级的（在 64 位系统上为 16 字节）
