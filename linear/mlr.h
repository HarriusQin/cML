#ifndef __C_MLR_H__
#define __C_MLR_H__

#include "dataset.h"
#include "machine_learning.h"
#include "linear_algebra.h"

typedef struct MLR_State
{
    Matrix* weights;
    Matrix* bias;
} MLR_State;

static int dataset_features_to_matrix(
    const dataset* ds, 
    const size_t* feature_indices, 
    size_t n_features, 
    const size_t* sample_indices, 
    size_t n_samples, 
    Matrix* result
);

static int mlr_adapter_with_bias(
    const dataset* ds, 
    const size_t* feature_indices, 
    size_t n_features, 
    const size_t* sample_indices, 
    size_t n_samples, 
    Matrix* result
);

static int normal_equation(Matrix* X, Matrix* y, Matrix* res);

static int MLR_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
                   const dataset* ds, const size_t* feature_indices, size_t n_features,
                   size_t target_index, const size_t* sample_indices,
                   size_t n_samples);

static int MLR_predict(const ML_Weights_t* state, const dataset* ds,
                       const size_t* feature_indices, size_t n_features,
                       const size_t* sample_indices, size_t n_samples,
                       void* output);

static void MLR_free(ML_Weights_t* state);

#ifdef MLR_IMPLEMENTATION

static int dataset_features_to_matrix(
    const dataset* ds, 
    const size_t* feature_indices, 
    size_t n_features, 
    const size_t* sample_indices, 
    size_t n_samples, 
    Matrix* result
) {
    if (!ds || !feature_indices || !result || n_features == 0 || n_samples == 0)
        return 1;
    if (matrix_init(result, n_samples, n_features) != 0)
        return 1;
    for (size_t i = 0; i < n_samples; i++) {
        size_t sample_idx;
        if (sample_indices == NULL) {
            sample_idx = i;
        } else {
            if (sample_indices[i] >= ds->rows) {
                free(result);
                return 1;
            }
            sample_idx = sample_indices[i];
        }
        for (size_t j = 0; j < n_features; j++) {
            if (feature_indices[j] >= ds->num_features) {
                free_matrix_data(result);
                return 1;
            }
            double value = ds->features[feature_indices[j]].data[sample_idx];
            result->mat[i * result->cols + j] = value;
        }
    }
    return 0; 
}

static int mlr_adapter_with_bias(
    const dataset* ds, 
    const size_t* feature_indices, 
    size_t n_features, 
    const size_t* sample_indices, 
    size_t n_samples, 
    Matrix* result
) {
    if (!ds || !feature_indices || !result || n_features == 0 || n_samples == 0)
        return 1;
    size_t matrix_cols = n_features + 1;
    if (matrix_init(result, n_samples, matrix_cols) != 0) 
        return 1;
    for (size_t i = 0; i < n_samples; i++) {
        size_t sample_idx;
        if (sample_indices == NULL) {
            sample_idx = i;
        } else {
            if (sample_indices[i] >= ds->rows) {
                free_matrix_data(result);
                return 1; // 索引越界
            }
            sample_idx = sample_indices[i];
        }
        for (size_t j = 0; j < matrix_cols; j++) {
            double* dest_ptr = &result->mat[i * result->cols + j];
            /* Bias Column */
            if (j == 0) {
                *dest_ptr = 1.0;
            } else {
                size_t feature_data_idx = feature_indices[j - 1];
                
                if (feature_data_idx >= ds->num_features)
                    return 1;
                *dest_ptr = ds->features[feature_data_idx].data[sample_idx];
            }
        }
    }
    return 0;
}

static int normal_equation(Matrix* X, Matrix* y, Matrix* res) {
    Matrix X_t = {0}, XTX = {0}, XTX_Inv = {0}, XTy = {0};
    if (matrix_transpose(X, &X_t) != 0) return 1;
    if (matrix_multiply(&X_t, X, &XTX) != 0) goto cleanup;
    if (matrix_inverse(&XTX, &XTX_Inv) != 0) {
        // Matrix singular or ill-conditioned, fall back to SVD-based pseudoinverse
        free_matrix_data(&X_t);
        free_matrix_data(&XTX);
        int svd_status = normal_equation_svd(X, y, res);
        return svd_status;
    }
    if (matrix_multiply(&X_t, y, &XTy) != 0) goto cleanup;
    if (matrix_multiply(&XTX_Inv, &XTy, res) != 0) goto cleanup;
cleanup:
    free_matrix_data(&X_t);
    free_matrix_data(&XTX);
    free_matrix_data(&XTX_Inv);
    free_matrix_data(&XTy);
    return 0;
}

static int MLR_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
            const dataset* ds, const size_t* feature_indices, size_t n_features,
            size_t target_index, const size_t* sample_indices,
            size_t n_samples) {
    Matrix X = {0}, y = {0}, theta = {0};
    int status = 0;
    if (mlr_adapter_with_bias(ds, feature_indices, n_features,
                              sample_indices, n_samples, &X) != 0)
        return 1;
    if (matrix_init(&y, n_samples, 1) != 0) {
        free_matrix_data(&X);
        return 1;
    }

    for (size_t i = 0; i < n_samples; i++) {
        size_t sample_idx;
        if (sample_indices == NULL) {
            sample_idx = i;
        } else {
            if (sample_indices[i] >= ds->rows) {
                status = 1; break;
            }
            sample_idx = sample_indices[i];
        }
        y.mat[i] = ds->features[target_index].data[sample_idx];
    }

    if (status != 0) {
        free_matrix_data(&X);
        free_matrix_data(&y);
        return 1;
    }

    // 求解正规方程：theta = (X^T X)^{-1} X^T y
    if (normal_equation(&X, &y, &theta) != 0) {
        status = 1; goto cleanup;
    }

    // 分离 bias 和 weights
    if (matrix_init(((MLR_State*)state->weights)->bias, 1, 1) != 0) {
        status = 1; goto cleanup;
    }
    ((MLR_State*)state->weights)->bias->mat[0] = theta.mat[0];

    if (matrix_init(((MLR_State*)state->weights)->weights, 1, n_features) != 0) {
        status = 1; goto cleanup;
    }

    for (size_t i = 0; i < n_features; i++)
        ((MLR_State*)state->weights)->weights->mat[i] = theta.mat[i + 1];

cleanup:
    free_matrix_data(&X);
    free_matrix_data(&y);
    free_matrix_data(&theta);

    return status;
}

static int MLR_predict(const ML_Weights_t* state, const dataset* ds,
                       const size_t* feature_indices, size_t n_features,
                       const size_t* sample_indices, size_t n_samples,
                       void* output) {
    Matrix X_test = {0}, result = {0};
    int status = 0;

    if (!state || !((MLR_State*)state->weights)->weights || !((MLR_State*)state->weights)->bias || !ds || !output)
        return 1;
    if (((MLR_State*)state->weights)->weights->cols != n_features)
        return 1;

    if (dataset_features_to_matrix(ds, feature_indices, n_features,
                                   sample_indices, n_samples, &X_test) != 0)
        return 1;

    if (matrix_init(&result, n_samples, 1) != 0) {
        free_matrix(&X_test);
        return 1;
    }

    if (matrix_multiply(&X_test, ((MLR_State*)state->weights)->weights, &result) != 0) {
        status = 1; goto cleanup;
    }

    double bias_val = ((MLR_State*)state->weights)->bias->mat[0];
    for (size_t i = 0; i < n_samples; i++)
        result.mat[i] += bias_val;

    double* result_array = (double*)output;
    for (size_t i = 0; i < n_samples; i++)
        result_array[i] = result.mat[i];

cleanup:
    free_matrix_data(&X_test);
    free_matrix_data(&result);
    return status;
}

static void MLR_free(ML_Weights_t* state) {
    free_matrix(((MLR_State*)state->weights)->bias);
    free_matrix(((MLR_State*)state->weights)->weights);
}

#endif /* MLR_IMPLEMENTATION */

#endif /* __C_MLR_H__ */