#ifndef __C_UTILITIES_H__
#define __C_UTILITIES_H__

#include <stddef.h>

/**
 * @brief Feature sample for sorting-based split finding
 *
 * Used by tree-based algorithms (Decision Tree, Random Forest, AdaBoost stumps)
 * to efficiently find optimal split thresholds by sorting feature values.
 */
typedef struct {
    size_t idx;  /**< Sample index in the dataset */
    double val;  /**< Feature value for the sample */
} FeatureSample;

/**
 * @brief Compare two FeatureSample entries by val (ascending)
 *
 * Used with qsort() for threshold finding.
 */
static int cmp_FeatureSample(const void* a, const void* b) {
    double da = ((const FeatureSample*)a)->val;
    double db = ((const FeatureSample*)b)->val;
    return (da > db) - (da < db);
}

#endif /* __C_UTILITIES_H__ */
