#ifndef __C_MACHINE_LEARNING_H__
#define __C_MACHINE_LEARNING_H__

/*
 * C Machine Learning Framework
 *
 * Single-header library for machine learning in C.
 *
 * Usage:
 *   #define C_MACHINE_LEARNING_IMPLEMENTATION
 *   #include "machine_learning.h"
 *
 * Or:
 *   #include "machine_learning.h"  // declarations only
 *   // in ONE .c file:
 *   #define C_MACHINE_LEARNING_IMPLEMENTATION
 *   #include "machine_learning.h"
 */

#include <stddef.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dataset.h"

/*
 * */
typedef enum {
    ML_REGRESSION = 0x1,
    ML_CLASSIFICATION = 0x2,      /**< Multi-class classification */
    ML_BINARY_CLASSIFICATION = 0x3 /**< Binary classification (e.g., AdaBoost, Logistic Regression) */
} ML_ModelType_t;

typedef enum {
    SCALING_NONE = 0,
    SCALING_STANDARD,   /**< (x - mean) / std */
    SCALING_MINMAX,     /**< (x - min) / (max - min) */
    SCALING_MAXABS      /**< x / max(|x|) */
} FeatureScaling_t;

typedef struct ML_Regression_Metrics_t {
    double MSE;
    double RMSE;
    double R_Squared;
    double MAE;
} ML_Regression_Metrics_t;

typedef struct ML_Classification_Metrics_t {
    double Accuracy;
    double Precision;
    double Recall;
} ML_Classification_Metrics_t;

/**
 * @brief Metrics for binary classification models (AdaBoost, Logistic Regression, etc.)
 */
typedef struct ML_BinaryClassification_Metrics_t {
    double Accuracy;
    double Precision;    /**< Positive class precision */
    double Recall;       /**< Positive class recall (sensitivity) */
    double Specificity;  /**< Negative class recall */
    double F1_Score;     /**< Harmonic mean of Precision and Recall */
    double AUC;          /**< Area Under ROC Curve (requires predict_proba) */
    double LogLoss;      /**< Logarithmic loss (requires predict_proba) */
} ML_BinaryClassification_Metrics_t;

/*
 * */
struct ML_Model_Config_t {
    void* params;
    size_t size;
};

typedef struct ML_Model_Config_t ML_Model_Config_t;

/*
 * */
struct ML_Weights_t {
    void* weights;
    size_t size;
};

typedef struct ML_Weights_t ML_Weights_t;

struct ML_Model_Impl_t {
    /* @brief Train the model
     * @param feature_indices Array of feature column indices (const - not modified)
     */
    int (*fit)(const ML_Model_Config_t* config, ML_Weights_t* state,
               const dataset* ds, const size_t* feature_indices, size_t n_features,
               size_t target_index, const size_t* sample_indices,
               size_t n_samples);

    /* @brief Predict class labels (hard predictions) */
    int (*predict)(const ML_Weights_t* state, const dataset* ds,
                   const size_t* feature_indices, size_t n_features,
                   const size_t* sample_indices, size_t n_samples,
                   void* output);

    /* @brief Predict class probabilities (for classification)
     * @param output Array of size n_samples * n_classes, row-major
     */
    int (*predict_proba)(const ML_Weights_t* state, const dataset* ds,
                         const size_t* feature_indices, size_t n_features,
                         const size_t* sample_indices, size_t n_samples,
                         size_t n_classes, void* output);

    /* @brief Get model coefficients (e.g., weights for linear model)
     * @param coeffs Output pointer to coefficients array
     * @param size Output size of coefficients array
     */
    int (*get_coefficients)(const ML_Weights_t* state, void** coeffs, size_t* size);

    /* @brief Serialize model state to buffer
     * @param buffer Output buffer pointer (caller must free)
     * @param size Output size of buffer
     */
    int (*serialize)(const ML_Weights_t* state, void** buffer, size_t* size);

    /* @brief Deserialize model state from buffer
     * @param state Output state to populate
     * @param buffer Input buffer
     * @param size Size of buffer
     */
    int (*deserialize)(ML_Weights_t* state, const void* buffer, size_t size);

    void (*free_state)(ML_Weights_t* state);
};

typedef struct ML_Model_Impl_t ML_Model_Impl_t;

/**
 * @brief Enum for ML_Model_Impl_t method types
 */
typedef enum {
    ML_METHOD_FIT = 0,
    ML_METHOD_PREDICT,
    ML_METHOD_PREDICT_PROBA,
    ML_METHOD_GET_COEFFICIENTS,
    ML_METHOD_SERIALIZE,
    ML_METHOD_DESERIALIZE,
    ML_METHOD_FREE_STATE,
    ML_METHOD_COUNT
} ML_MethodType_t;

/**
 * @brief Check if a method is implemented
 * @param methods Pointer to ML_Model_Impl_t
 * @param method Method type to check
 * @return true if method is implemented (not NULL), false otherwise
 */
static inline bool ml_method_is_implemented(const ML_Model_Impl_t* methods, ML_MethodType_t method) {
    if (!methods)
        return false;
    switch (method) {
    case ML_METHOD_FIT:         return methods->fit != NULL;
    case ML_METHOD_PREDICT:      return methods->predict != NULL;
    case ML_METHOD_PREDICT_PROBA: return methods->predict_proba != NULL;
    case ML_METHOD_GET_COEFFICIENTS: return methods->get_coefficients != NULL;
    case ML_METHOD_SERIALIZE:    return methods->serialize != NULL;
    case ML_METHOD_DESERIALIZE:  return methods->deserialize != NULL;
    case ML_METHOD_FREE_STATE:   return methods->free_state != NULL;
    default: return false;
    }
}

/**
 * @brief Get method implementation status as bitmask
 * @param methods Pointer to ML_Model_Impl_t
 * @return Bitmask where bit i is 1 if method i is implemented
 */
static inline unsigned int ml_methods_get_implemented(const ML_Model_Impl_t* methods) {
    unsigned int mask = 0;
    if (!methods) return 0;
    if (methods->fit)             mask |= (1 << ML_METHOD_FIT);
    if (methods->predict)          mask |= (1 << ML_METHOD_PREDICT);
    if (methods->predict_proba)    mask |= (1 << ML_METHOD_PREDICT_PROBA);
    if (methods->get_coefficients) mask |= (1 << ML_METHOD_GET_COEFFICIENTS);
    if (methods->serialize)       mask |= (1 << ML_METHOD_SERIALIZE);
    if (methods->deserialize)     mask |= (1 << ML_METHOD_DESERIALIZE);
    if (methods->free_state)      mask |= (1 << ML_METHOD_FREE_STATE);
    return mask;
}

typedef struct ML_Model {
    ML_ModelType_t type;
    ML_Model_Config_t config;
    ML_Weights_t state;
    ML_Model_Impl_t methods;
} ML_Model_t;

typedef struct {
    ML_ModelType_t metric_type;  /**< Type tag to identify which union variant is valid */
    size_t k;
    union {
        ML_Regression_Metrics_t reg;
        ML_Classification_Metrics_t cls;
        ML_BinaryClassification_Metrics_t bin;
    } avg_metrics;
    double* fold_scores;
} CV_Result_t;

typedef struct {
    size_t* train_indices;
    size_t train_size;
    size_t* test_indices;
    size_t test_size;
} Dataset_Split_t;

/*
 * @brief Feature scaling parameters (fitted on training data)
 */
typedef struct ML_ScalingParams_t {
    FeatureScaling_t type;
    double* mean;       /**< For STANDARD scaling, size = n_features */
    double* std;        /**< For STANDARD scaling, size = n_features */
    double* min;        /**< For MINMAX scaling, size = n_features */
    double* max;        /**< For MINMAX scaling, size = n_features */
    double* maxabs;     /**< For MAXABS scaling, size = n_features */
    size_t n_features;
    size_t n_samples;   /**< Original number of samples */
    size_t* sample_indices; /**< Indices of samples used for fitting */
    size_t n_indices;   /**< Number of indices (may be subset) */
} ML_ScalingParams_t;

/*
 * @brief Preprocessor for feature scaling
 *
 * Provides stateless transformation: dataset -> dataset
 * Scaling parameters are fitted separately via ML_ScalingParams_t
 */
typedef struct ML_Preprocessor_t {
    ML_ScalingParams_t* (*fit)(const dataset* ds, const size_t* feature_indices,
                               size_t n_features, const size_t* sample_indices,
                               size_t n_samples);
    dataset* (*transform)(const ML_ScalingParams_t* params, const dataset* ds,
                          const size_t* feature_indices, size_t n_features,
                          const size_t* sample_indices, size_t n_samples);
    void (*free_params)(ML_ScalingParams_t* params);
} ML_Preprocessor_t;

/*
 * @brief Fit scaling parameters on a subset of data
 */
static ML_ScalingParams_t* ml_fit_scaling(const dataset* ds, const size_t* feature_indices,
                                           size_t n_features, const size_t* sample_indices,
                                           size_t n_samples, FeatureScaling_t type) {
    if (!ds || n_features == 0 || n_samples == 0)
        return NULL;

    ML_ScalingParams_t* params = (ML_ScalingParams_t*)calloc(1, sizeof(ML_ScalingParams_t));
    if (!params)
        return NULL;

    params->type = type;
    params->n_features = n_features;
    params->n_samples = ds->rows;
    params->n_indices = n_samples;

    params->sample_indices = (size_t*)malloc(sizeof(size_t) * n_samples);
    if (!params->sample_indices) {
        free(params);
        return NULL;
    }
    memcpy(params->sample_indices, sample_indices, sizeof(size_t) * n_samples);

    switch (type) {
    case SCALING_STANDARD:
        params->mean = (double*)calloc(n_features, sizeof(double));
        params->std = (double*)calloc(n_features, sizeof(double));
        if (!params->mean || !params->std) {
            free(params->mean); free(params->std); free(params->sample_indices); free(params);
            return NULL;
        }
        for (size_t f = 0; f < n_features; f++) {
            double sum = 0.0;
            for (size_t i = 0; i < n_samples; i++) {
                sum += ds->features[feature_indices[f]].data[sample_indices[i]];
            }
            params->mean[f] = sum / n_samples;
        }
        for (size_t f = 0; f < n_features; f++) {
            double sq_sum = 0.0;
            for (size_t i = 0; i < n_samples; i++) {
                double diff = ds->features[feature_indices[f]].data[sample_indices[i]] - params->mean[f];
                sq_sum += diff * diff;
            }
            params->std[f] = sqrt(sq_sum / n_samples);
            if (params->std[f] < 1e-10) params->std[f] = 1.0;
        }
        break;

    case SCALING_MINMAX:
        params->min = (double*)malloc(sizeof(double) * n_features);
        params->max = (double*)malloc(sizeof(double) * n_features);
        if (!params->min || !params->max) {
            free(params->min); free(params->max); free(params->sample_indices); free(params);
            return NULL;
        }
        for (size_t f = 0; f < n_features; f++) {
            params->min[f] = ds->features[feature_indices[f]].data[sample_indices[0]];
            params->max[f] = ds->features[feature_indices[f]].data[sample_indices[0]];
            for (size_t i = 1; i < n_samples; i++) {
                double val = ds->features[feature_indices[f]].data[sample_indices[i]];
                if (val < params->min[f]) params->min[f] = val;
                if (val > params->max[f]) params->max[f] = val;
            }
        }
        break;

    case SCALING_MAXABS:
        params->maxabs = (double*)malloc(sizeof(double) * n_features);
        if (!params->maxabs) {
            free(params->maxabs); free(params->sample_indices); free(params);
            return NULL;
        }
        for (size_t f = 0; f < n_features; f++) {
            params->maxabs[f] = fabs(ds->features[feature_indices[f]].data[sample_indices[0]]);
            for (size_t i = 1; i < n_samples; i++) {
                double abs_val = fabs(ds->features[feature_indices[f]].data[sample_indices[i]]);
                if (abs_val > params->maxabs[f]) params->maxabs[f] = abs_val;
            }
            if (params->maxabs[f] < 1e-10) params->maxabs[f] = 1.0;
        }
        break;

    case SCALING_NONE:
    default:
        break;
    }

    return params;
}

/*
 * @brief Transform features using fitted scaling parameters
 */
static dataset* ml_transform_features(const ML_ScalingParams_t* params, const dataset* ds,
                                      const size_t* feature_indices, size_t n_features,
                                      const size_t* sample_indices, size_t n_samples) {
    if (!params || !ds || params->n_features != n_features)
        return NULL;

    dataset* out = (dataset*)malloc(sizeof(dataset));
    if (!out)
        return NULL;

    out->rows = n_samples;
    out->num_features = n_features;
    out->num_labels = ds->num_labels;
    out->num_binary_labels = 0;
    out->binary_labels = NULL;

    out->features = (num_column*)malloc(sizeof(num_column) * n_features);
    if (!out->features) {
        free(out);
        return NULL;
    }

    for (size_t f = 0; f < n_features; f++) {
        out->features[f].name = ds->features[feature_indices[f]].name;
        out->features[f].n = n_samples;
        out->features[f].data = (double*)malloc(sizeof(double) * n_samples);
        if (!out->features[f].data) {
            for (size_t k = 0; k < f; k++) free(out->features[k].data);
            free(out->features); free(out);
            return NULL;
        }

        for (size_t i = 0; i < n_samples; i++) {
            double val = ds->features[feature_indices[f]].data[sample_indices[i]];
            switch (params->type) {
            case SCALING_STANDARD:
                val = (val - params->mean[f]) / params->std[f];
                break;
            case SCALING_MINMAX:
                val = (val - params->min[f]) / (params->max[f] - params->min[f]);
                break;
            case SCALING_MAXABS:
                val = val / params->maxabs[f];
                break;
            case SCALING_NONE:
            default:
                break;
            }
            out->features[f].data[i] = val;
        }
    }

    if (ds->num_labels > 0) {
        out->labels = (label_column*)malloc(sizeof(label_column) * ds->num_labels);
        if (!out->labels) {
            for (size_t k = 0; k < n_features; k++) free(out->features[k].data);
            free(out->features); free(out);
            return NULL;
        }
        for (size_t l = 0; l < ds->num_labels; l++) {
            out->labels[l].name = ds->labels[l].name;
            out->labels[l].n = n_samples;
            out->labels[l].classes = ds->labels[l].classes;
            out->labels[l].labels = (int*)malloc(sizeof(int) * n_samples);
            out->labels[l].value_map = (char**)malloc(sizeof(char*) * ds->labels[l].classes);
            if (!out->labels[l].labels || !out->labels[l].value_map) {
                if (out->labels[l].labels) free(out->labels[l].labels);
                if (out->labels[l].value_map) free(out->labels[l].value_map);
                for (size_t k = 0; k < l; k++) { free(out->labels[k].labels); free(out->labels[k].value_map); }
                for (size_t k = 0; k < n_features; k++) free(out->features[k].data);
                free(out->features); free(out->labels); free(out);
                return NULL;
            }
            for (size_t c = 0; c < ds->labels[l].classes; c++) {
                out->labels[l].value_map[c] = (char*)malloc(strlen(ds->labels[l].value_map[c]) + 1);
                if (!out->labels[l].value_map[c]) {
                    for (size_t k = 0; k < c; k++) free(out->labels[l].value_map[k]);
                    for (size_t k = 0; k < l; k++) { free(out->labels[k].labels); free(out->labels[k].value_map); }
                    for (size_t k = 0; k < n_features; k++) free(out->features[k].data);
                    free(out->features); free(out->labels); free(out);
                    return NULL;
                }
                strcpy(out->labels[l].value_map[c], ds->labels[l].value_map[c]);
            }
            for (size_t i = 0; i < n_samples; i++) {
                out->labels[l].labels[i] = ds->labels[l].labels[sample_indices[i]];
            }
        }
    } else {
        out->labels = NULL;
    }

    return out;
}

/*
 * @brief Free scaling parameters
 */
static void ml_free_scaling_params(ML_ScalingParams_t* params) {
    if (!params)
        return;
    free(params->mean);
    free(params->std);
    free(params->min);
    free(params->max);
    free(params->maxabs);
    free(params->sample_indices);
    free(params);
}

#ifdef C_MACHINE_LEARNING_IMPLEMENTATION
/* ============================================================================
 * IMPLEMENTATIONS
 * ============================================================================ */

static int train_test_split(const dataset* ds, double test_ratio, unsigned int seed,
                            Dataset_Split_t* split) {
    if (!ds || !split || test_ratio <= 0 || test_ratio >= 1)
        return -1;

    size_t n = ds->rows;
    size_t n_test = (size_t)(n * test_ratio);
    size_t n_train = n - n_test;

    /* Allocate combined array for shuffle */
    size_t* indices = (size_t*)malloc(sizeof(size_t) * n);
    if (!indices)
        return -1;

    /* Initialize sequential indices */
    for (size_t i = 0; i < n; i++)
        indices[i] = i;

    /* Fisher-Yates shuffle */
    srand(seed);
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = (size_t)(rand() % (i + 1));
        size_t tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }

    /* Allocate output arrays */
    split->train_indices = (size_t*)malloc(sizeof(size_t) * n_train);
    split->test_indices = (size_t*)malloc(sizeof(size_t) * n_test);
    if (!split->train_indices || !split->test_indices) {
        free(indices);
        free(split->train_indices);
        free(split->test_indices);
        split->train_indices = NULL;
        split->test_indices = NULL;
        return -1;
    }

    /* Copy first n_train to train, rest to test */
    for (size_t i = 0; i < n_train; i++)
        split->train_indices[i] = indices[i];
    for (size_t i = 0; i < n_test; i++)
        split->test_indices[i] = indices[n_train + i];

    split->train_size = n_train;
    split->test_size = n_test;

    free(indices);
    return 0;
}

static double compute_accuracy(const ML_Weights_t* state, const dataset* ds,
                               size_t* feature_indices, size_t n_features,
                               const size_t* sample_indices, size_t n_samples,
                               int* predictions) {
    if (!state || !ds || !predictions)
        return 0.0;

    /* Simple majority class predictor based on training distribution */
    /* This is a placeholder - real implementation would use model->methods.predict */
    (void)feature_indices;
    (void)n_features;
    (void)sample_indices;
    (void)n_samples;

    /* For now, predict most common class (0) */
    for (size_t i = 0; i < n_samples; i++) {
        predictions[i] = 0;
    }
    return 0.0;
}

static int train_model_with_validation(ML_Model_t* model, const dataset* ds,
                                       const size_t* feature_indices, size_t n_features,
                                       size_t target_index, double val_ratio,
                                       void* train_metrics, void* validation_metrics) {
    if (!model || !ds || !train_metrics || !validation_metrics)
        return -1;

    Dataset_Split_t split = {0};
    size_t n = ds->rows;
    size_t n_val = (size_t)(n * val_ratio);
    size_t n_train = n - n_val;

    /* Create indices */
    size_t* indices = (size_t*)malloc(sizeof(size_t) * n);
    if (!indices) return -1;
    for (size_t i = 0; i < n; i++) indices[i] = i;

    /* Shuffle */
    srand(42);
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = (size_t)(rand() % (i + 1));
        size_t tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }

    split.train_indices = indices;
    split.train_size = n_train;
    split.test_indices = indices + n_train;
    split.test_size = n_val;

    /* Fit model */
    if (model->methods.fit) {
        int ret = model->methods.fit(&model->config, &model->state, ds,
                                     feature_indices, n_features, target_index,
                                     split.train_indices, n_train);
        if (ret != 0) {
            free(indices);
            return ret;
        }
    }

    /* Compute training metrics */
    int* train_preds = (int*)malloc(sizeof(int) * n_train);
    int* val_preds = (int*)malloc(sizeof(int) * n_val);
    if (!train_preds || !val_preds) {
        free(indices);
        free(train_preds);
        free(val_preds);
        return -1;
    }

    if (model->methods.predict) {
        model->methods.predict(&model->state, ds, feature_indices, n_features,
                              split.train_indices, n_train, train_preds);
        model->methods.predict(&model->state, ds, feature_indices, n_features,
                              split.test_indices, n_val, val_preds);
    }

    /* Compute accuracy */
    size_t train_correct = 0, val_correct = 0;
    for (size_t i = 0; i < n_train; i++) {
        int actual = ds->labels[target_index].labels[split.train_indices[i]];
        if (train_preds[i] == actual) train_correct++;
    }
    for (size_t i = 0; i < n_val; i++) {
        int actual = ds->labels[target_index].labels[split.test_indices[i]];
        if (val_preds[i] == actual) val_correct++;
    }

    if (model->type == ML_CLASSIFICATION) {
        ML_Classification_Metrics_t* tm = (ML_Classification_Metrics_t*)train_metrics;
        ML_Classification_Metrics_t* vm = (ML_Classification_Metrics_t*)validation_metrics;
        tm->Accuracy = (double)train_correct / n_train;
        vm->Accuracy = (double)val_correct / n_val;
        tm->Precision = tm->Accuracy;  /* Placeholder */
        tm->Recall = tm->Accuracy;
        vm->Precision = vm->Accuracy;
        vm->Recall = vm->Accuracy;
    } else if (model->type == ML_BINARY_CLASSIFICATION) {
        ML_BinaryClassification_Metrics_t* tm = (ML_BinaryClassification_Metrics_t*)train_metrics;
        ML_BinaryClassification_Metrics_t* vm = (ML_BinaryClassification_Metrics_t*)validation_metrics;
        tm->Accuracy = (double)train_correct / n_train;
        vm->Accuracy = (double)val_correct / n_val;
        tm->Precision = tm->Accuracy;
        tm->Recall = tm->Accuracy;
        tm->Specificity = tm->Accuracy;
        tm->F1_Score = tm->Accuracy;
        tm->AUC = tm->Accuracy;
        tm->LogLoss = 1.0 - tm->Accuracy;
        vm->Precision = vm->Accuracy;
        vm->Recall = vm->Accuracy;
        vm->Specificity = vm->Accuracy;
        vm->F1_Score = vm->Accuracy;
        vm->AUC = vm->Accuracy;
        vm->LogLoss = 1.0 - vm->Accuracy;
    }

    free(indices);
    free(train_preds);
    free(val_preds);

    return 0;
}

static CV_Result_t kfold_cross_validate(ML_Model_t* model, const dataset* ds,
                                        const size_t* feature_indices,
                                        size_t n_features, size_t target_index, int k,
                                        unsigned int seed) {
    CV_Result_t result;
    memset(&result, 0, sizeof(result));

    if (!model || !ds || k <= 0)
        return result;

    result.metric_type = model->type;
    size_t n = ds->rows;
    result.k = (size_t)k;
    result.fold_scores = (double*)malloc(sizeof(double) * k);
    if (!result.fold_scores)
        return result;

    /* Create shuffled indices */
    size_t* indices = (size_t*)malloc(sizeof(size_t) * n);
    if (!indices) {
        free(result.fold_scores);
        result.fold_scores = NULL;
        return result;
    }
    for (size_t i = 0; i < n; i++) indices[i] = i;

    srand(seed);
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = (size_t)(rand() % (i + 1));
        size_t tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }

    size_t fold_size = n / k;
    double avg_score = 0.0;

    for (int fold = 0; fold < k; fold++) {
        size_t val_start = fold * fold_size;
        size_t val_end = (fold == k - 1) ? n : val_start + fold_size;
        size_t val_size = val_end - val_start;
        size_t train_size = n - val_size;

        /* Create train/val indices for this fold */
        size_t* train_idx = (size_t*)malloc(sizeof(size_t) * train_size);
        size_t* val_idx = (size_t*)malloc(sizeof(size_t) * val_size);
        if (!train_idx || !val_idx) {
            free(train_idx);
            free(val_idx);
            for (int f = 0; f < fold; f++) free(result.fold_scores);
            free(result.fold_scores);
            result.fold_scores = NULL;
            free(indices);
            return result;
        }

        size_t t = 0;
        for (size_t i = 0; i < n; i++) {
            if (i >= val_start && i < val_end) {
                val_idx[i - val_start] = indices[i];
            } else {
                train_idx[t++] = indices[i];
            }
        }

        /* Fit model */
        if (model->methods.fit) {
            model->methods.fit(&model->config, &model->state, ds,
                              feature_indices, n_features, target_index,
                              train_idx, train_size);
        }

        /* Evaluate */
        int* preds = (int*)malloc(sizeof(int) * val_size);
        if (preds && model->methods.predict) {
            model->methods.predict(&model->state, ds, feature_indices, n_features,
                                 val_idx, val_size, preds);
        }

        size_t correct = 0;
        if (preds) {
            for (size_t i = 0; i < val_size; i++) {
                int actual = ds->labels[target_index].labels[val_idx[i]];
                if (preds[i] == actual) correct++;
            }
            free(preds);
        }

        double score = (double)correct / val_size;
        result.fold_scores[fold] = score;
        avg_score += score;

        free(train_idx);
        free(val_idx);
    }

    avg_score /= k;
    if (model->type == ML_CLASSIFICATION) {
        result.avg_metrics.cls.Accuracy = avg_score;
        result.avg_metrics.cls.Precision = avg_score;
        result.avg_metrics.cls.Recall = avg_score;
    } else if (model->type == ML_BINARY_CLASSIFICATION) {
        result.avg_metrics.bin.Accuracy = avg_score;
        result.avg_metrics.bin.Precision = avg_score;
        result.avg_metrics.bin.Recall = avg_score;
        result.avg_metrics.bin.Specificity = avg_score;
        result.avg_metrics.bin.F1_Score = avg_score;
        result.avg_metrics.bin.AUC = avg_score;
        result.avg_metrics.bin.LogLoss = 1.0 - avg_score;
    } else {
        result.avg_metrics.reg.MSE = 1.0 - avg_score;
        result.avg_metrics.reg.RMSE = sqrt(1.0 - avg_score);
        result.avg_metrics.reg.R_Squared = avg_score;
        result.avg_metrics.reg.MAE = 1.0 - avg_score;
    }

    free(indices);
    return result;
}

/**
 * @brief Evaluate model on a given dataset
 * @param model Pointer to model (must have state populated from training)
 * @param ds Dataset to evaluate on
 * @param feature_indices Array of feature column indices
 * @param n_features Number of features
 * @param target_index Index of label column to evaluate against
 * @param sample_indices Indices of samples to evaluate
 * @param n_samples Number of samples
 * @param metrics Output metrics (must be ML_Classification_Metrics_t* or ML_Regression_Metrics_t*)
 * @return 0 on success, -1 on error
 */
static int model_evaluate(ML_Model_t* model, const dataset* ds,
                         const size_t* feature_indices, size_t n_features,
                         size_t target_index, const size_t* sample_indices,
                         size_t n_samples, void* metrics) {
    if (!model || !ds || !feature_indices || !sample_indices || !metrics)
        return -1;

    if (!ml_method_is_implemented(&model->methods, ML_METHOD_PREDICT)) {
        return -1;  /* Cannot evaluate without predict method */
    }

    /* Allocate predictions array */
    int* predictions = (int*)malloc(sizeof(int) * n_samples);
    if (!predictions)
        return -1;

    /* Run predictions */
    int ret = model->methods.predict(&model->state, ds, feature_indices, n_features,
                                     sample_indices, n_samples, predictions);
    if (ret != 0) {
        free(predictions);
        return ret;
    }

    /* Compute metrics based on model type */
    if (model->type == ML_CLASSIFICATION) {
        ML_Classification_Metrics_t* m = (ML_Classification_Metrics_t*)metrics;
        size_t correct = 0, tp = 0, fp = 0, fn = 0;

        for (size_t i = 0; i < n_samples; i++) {
            int actual = ds->labels[target_index].labels[sample_indices[i]];
            int pred = predictions[i];
            if (pred == actual) {
                correct++;
                tp++;  /* Simplified: treat correct as TP */
            } else {
                fp++;  /* Predicted positive but wrong */
                fn++;  /* Actual positive but predicted wrong */
            }
        }

        m->Accuracy = (double)correct / n_samples;
        m->Precision = (tp + fp > 0) ? (double)tp / (tp + fp) : 0.0;
        m->Recall = (tp + fn > 0) ? (double)tp / (tp + fn) : 0.0;

    } else if (model->type == ML_BINARY_CLASSIFICATION) {
        ML_BinaryClassification_Metrics_t* m = (ML_BinaryClassification_Metrics_t*)metrics;
        size_t tp = 0, fp = 0, fn = 0, tn = 0;

        /* Binary classification: assume class 1 is positive, class 0 is negative */
        for (size_t i = 0; i < n_samples; i++) {
            int actual = ds->labels[target_index].labels[sample_indices[i]];
            int pred = predictions[i];
            if (pred == 1 && actual == 1) {
                tp++;
            } else if (pred == 1 && actual == 0) {
                fp++;
            } else if (pred == 0 && actual == 1) {
                fn++;
            } else if (pred == 0 && actual == 0) {
                tn++;
            }
        }

        m->Accuracy = (double)(tp + tn) / n_samples;
        m->Precision = (tp + fp > 0) ? (double)tp / (tp + fp) : 0.0;
        m->Recall = (tp + fn > 0) ? (double)tp / (tp + fn) : 0.0;  /* Sensitivity */
        m->Specificity = (tn + fp > 0) ? (double)tn / (tn + fp) : 0.0;
        m->F1_Score = (m->Precision + m->Recall > 0) ?
            2.0 * m->Precision * m->Recall / (m->Precision + m->Recall) : 0.0;
        m->AUC = 0.0;  /* Requires predict_proba - computed below if available */
        m->LogLoss = 0.0;

        /* If predict_proba is available, compute AUC and LogLoss */
        if (ml_method_is_implemented(&model->methods, ML_METHOD_PREDICT_PROBA)) {
            double* probas = (double*)malloc(sizeof(double) * n_samples * 2);
            if (probas) {
                model->methods.predict_proba(&model->state, ds, feature_indices, n_features,
                                             sample_indices, n_samples, 2, probas);
                /* Compute AUC using simple ranking method */
                size_t n_pos = 0, n_neg = 0;
                for (size_t i = 0; i < n_samples; i++) {
                    if (ds->labels[target_index].labels[sample_indices[i]] == 1) n_pos++;
                    else n_neg++;
                }
                /* Count correctly ranked pairs - compare P(class 1) */
                double auc_sum = 0.0;
                for (size_t i = 0; i < n_samples; i++) {
                    if (ds->labels[target_index].labels[sample_indices[i]] == 1) {
                        double p_i = probas[i * 2 + 1];  /* P(class 1) */
                        for (size_t j = 0; j < n_samples; j++) {
                            if (ds->labels[target_index].labels[sample_indices[j]] == 0) {
                                double p_j = probas[j * 2 + 1];  /* P(class 1) */
                                if (p_i > p_j) auc_sum += 1.0;
                                else if (p_i == p_j) auc_sum += 0.5;
                            }
                        }
                    }
                }
                m->AUC = (n_pos * n_neg > 0) ? auc_sum / (n_pos * n_neg) : 0.0;

                /* Compute LogLoss */
                double logloss = 0.0;
                for (size_t i = 0; i < n_samples; i++) {
                    int actual = ds->labels[target_index].labels[sample_indices[i]];
                    double p = (actual == 1) ? probas[i * 2 + 1] : probas[i * 2];
                    p = (p < 1e-10) ? 1e-10 : (p > 1.0 - 1e-10) ? 1.0 - 1e-10 : p;
                    logloss += log(p);
                }
                m->LogLoss = -logloss / n_samples;

                free(probas);
            }
        }

    } else if (model->type == ML_REGRESSION) {
        ML_Regression_Metrics_t* m = (ML_Regression_Metrics_t*)metrics;
        double mse = 0.0, ss_tot = 0.0, ss_res = 0.0;
        double mean = 0.0;

        /* Compute mean */
        for (size_t i = 0; i < n_samples; i++) {
            mean += ds->labels[target_index].labels[sample_indices[i]];
        }
        mean /= n_samples;

        for (size_t i = 0; i < n_samples; i++) {
            double actual = (double)ds->labels[target_index].labels[sample_indices[i]];
            double pred = (double)predictions[i];
            double diff = actual - pred;
            mse += diff * diff;
            ss_res += diff * diff;
            ss_tot += (actual - mean) * (actual - mean);
        }

        mse /= n_samples;
        m->MSE = mse;
        m->RMSE = sqrt(mse);
        m->MAE = sqrt(mse);  /* Simplified: use RMSE as MAE approximation */
        m->R_Squared = (ss_tot > 0) ? 1.0 - (ss_res / ss_tot) : 0.0;
    }

    free(predictions);
    return 0;
}

#endif /* C_MACHINE_LEARNING_IMPLEMENTATION */

#endif /* __C_MACHINE_LEARNING_H__ */
