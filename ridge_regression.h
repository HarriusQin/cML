#ifndef __C_RIDGE_REGRESSION_H__
#define __C_RIDGE_REGRESSION_H__

#include "dataset.h"
#include "machine_learning.h"
#include "linear_algebra.h"

typedef struct Ridge_State {
    Matrix *weights;
    Matrix *bias;
    double lambda;
} Ridge_State;

static int ridge_fit(const ML_Model_Config_t *config, ML_Weights_t *state,
                    const dataset *ds, const size_t *feature_indices,
                    size_t n_features, size_t target_index,
                    const size_t *sample_indices, size_t n_samples,
                    double lambda);

static int ridge_predict(const ML_Weights_t *state, const dataset *ds,
                        const size_t *feature_indices, size_t n_features,
                        const size_t *sample_indices, size_t n_samples,
                        void *output);

static void ridge_free(ML_Weights_t *state);

#ifdef RIDGE_REGRESSION_IMPLEMENTATION

static int ridge_fit(const ML_Model_Config_t *config, ML_Weights_t *state,
                    const dataset *ds, const size_t *feature_indices,
                    size_t n_features, size_t target_index,
                    const size_t *sample_indices, size_t n_samples,
                    double lambda) {
    Matrix X = {0}, X_aug = {0}, y = {0}, theta = {0};
    int status = 0;

    // 构造 X_aug = [1 | X]（带偏置）
    {
        Matrix X_raw = {0};
        extern int mlr_adapter_with_bias(const dataset *, const size_t *, size_t,
                                         const size_t *, size_t, Matrix *);
        if (mlr_adapter_with_bias(ds, feature_indices, n_features,
                                  sample_indices, n_samples, &X_raw) != 0)
            return 1;
        if (matrix_init(&X_aug, n_samples, n_features + 1) != 0) {
            free_matrix_data(&X_raw); return 1;
        }
        memcpy(X_aug.mat, X_raw.mat, n_samples * (n_features + 1) * sizeof(double));
        free_matrix_data(&X_raw);
    }

    // 构造 y
    if (matrix_init(&y, n_samples, 1) != 0) {
        free_matrix_data(&X_aug); return 1;
    }
    for (size_t i = 0; i < n_samples; i++) {
        size_t idx = sample_indices ? sample_indices[i] : i;
        y.mat[i] = ds->features[target_index].data[idx];
    }

    // Ridge: theta = (XᵀX + λ·I)⁻¹ Xᵀy
    {
        Matrix X_t = {0}, XTX = {0}, XTX_Inv = {0}, XTy = {0};
        if (matrix_transpose(&X_aug, &X_t) != 0) { status = 1; goto cleanup; }
        if (matrix_multiply(&X_t, &X_aug, &XTX) != 0) { status = 1; goto cleanup; }

        // XTX = XTX + λ·I（对角线加 λ）
        size_t n = XTX.cols;
        for (size_t i = 0; i < n; i++)
            XTX.mat[i * n + i] += lambda;

        if (matrix_inverse(&XTX, &XTX_Inv) != 0) { status = 1; goto cleanup; }
        if (matrix_multiply(&X_t, &y, &XTy) != 0) { status = 1; goto cleanup; }
        if (matrix_multiply(&XTX_Inv, &XTy, &theta) != 0) { status = 1; goto cleanup; }

        free_matrix_data(&X_t); free_matrix_data(&XTX); free_matrix_data(&XTX_Inv); free_matrix_data(&XTy);
        goto skip_cleanup;
cleanup:
        free_matrix_data(&X_t); free_matrix_data(&XTX); free_matrix_data(&XTX_Inv); free_matrix_data(&XTy);
        free_matrix_data(&X_aug); free_matrix_data(&y); return status;
skip_cleanup:
        (void)0;
    }

    ((Ridge_State *)state->weights)->lambda = lambda;
    ((Ridge_State *)state->weights)->bias = calloc(1, sizeof(Matrix));
    matrix_init(((Ridge_State *)state->weights)->bias, 1, 1);
    ((Ridge_State *)state->weights)->bias->mat[0] = theta.mat[0];

    ((Ridge_State *)state->weights)->weights = calloc(1, sizeof(Matrix));
    if (matrix_init(((Ridge_State *)state->weights)->weights, 1, n_features) != 0) {
        free_matrix_data(&theta); free_matrix_data(&X_aug); free_matrix_data(&y); return 1;
    }
    for (size_t i = 0; i < n_features; i++)
        ((Ridge_State *)state->weights)->weights->mat[i] = theta.mat[i + 1];

    free_matrix_data(&X_aug); free_matrix_data(&y); free_matrix_data(&theta);
    return 0;
}

static int ridge_predict(const ML_Weights_t *state, const dataset *ds,
                        const size_t *feature_indices, size_t n_features,
                        const size_t *sample_indices, size_t n_samples,
                        void *output) {
    Matrix X = {0};
    double *result = (double *)output;
    Ridge_State *s = (Ridge_State *)state->weights;

    extern int dataset_features_to_matrix(const dataset *, const size_t *, size_t,
                                         const size_t *, size_t, Matrix *);
    if (dataset_features_to_matrix(ds, feature_indices, n_features,
                                  sample_indices, n_samples, &X) != 0)
        return 1;

    Matrix pred = {0};
    if (matrix_init(&pred, n_samples, 1) != 0) { free_matrix_data(&X); return 1; }
    if (matrix_multiply(&X, s->weights, &pred) != 0) { free_matrix_data(&X); free_matrix_data(&pred); return 1; }

    for (size_t i = 0; i < n_samples; i++)
        result[i] = pred.mat[i] + s->bias->mat[0];

    free_matrix_data(&X); free_matrix_data(&pred);
    return 0;
}

static void ridge_free(ML_Weights_t *state) {
    if (!state) return;
    Ridge_State *s = (Ridge_State *)state->weights;
    if (s->bias) free_matrix(s->bias);
    if (s->weights) free_matrix(s->weights);
}

#endif /* RIDGE_REGRESSION_IMPLEMENTATION */

#endif /* __C_RIDGE_REGRESSION_H__ */
