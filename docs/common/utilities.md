# Utilities

Helper utilities for tree-based algorithms.

## Overview

The utilities module provides data structures and comparison functions used by decision tree algorithms (Random Forest, AdaBoost, XGBoost, CatBoost) for efficient split finding.

## Feature Sample

Used for sorting-based split finding:

```c
typedef struct {
    size_t idx;  // Sample index in the dataset
    double val; // Feature value for the sample
} FeatureSample;
```

Each entry pairs a sample index with its feature value, enabling efficient sorting to find candidate split thresholds.

## Comparison Function

```c
static int cmp_FeatureSample(const void* a, const void* b) {
    double da = ((const FeatureSample*)a)->val;
    double db = ((const FeatureSample*)b)->val;
    return (da > db) - (da < db);
}
```

Used with `qsort()` for ascending sort by feature value.

## Split Finding Algorithm

For a given feature, find the best split point:

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

    // Sort samples by feature value
    qsort(samples, n_samples, sizeof(FeatureSample), cmp_FeatureSample);

    // Try splits at unique threshold values
    double best_gain = -INFINITY;
    double best_threshold = 0;

    for (size_t i = 0; i < n_samples - 1; i++) {
        // Only consider unique values (skip duplicates)
        if (samples[i].val == samples[i + 1].val) continue;

        double threshold = (samples[i].val + samples[i + 1].val) / 2;

        // Compute left/right sums
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

        // Compute information gain (for classification)
        double left_entropy = 0, right_entropy = 0;
        // ... entropy computation ...

        double gain = left_entropy + right_entropy;

        if (gain > best_gain) {
            best_gain = gain;
            best_threshold = threshold;
        }
    }

    return best_threshold;
}
```

## Usage in Decision Trees

```c
// For each feature, find best split
for (size_t f = 0; f < n_features; f++) {
    FeatureSample* samples = malloc(n_samples * sizeof(FeatureSample));

    // Build feature sample array
    for (size_t i = 0; i < n_samples; i++) {
        samples[i].idx = indices[i];
        samples[i].val = dataset->features[f].data[samples[i].idx];
    }

    // Find best split for this feature
    double threshold = find_best_split(samples, n_samples, targets,
                                     thresholds, left, right,
                                     left_sum, right_sum);

    free(samples);
}
```

## Notes

- Sorting-based split finding is exact (finds optimal split)
- Alternative: histogram-based splitting for faster approximate splits
- For large datasets, subsample features and/or samples
- Duplicate feature values are skipped when computing split candidates
- `FeatureSample` is lightweight (16 bytes on 64-bit systems)
