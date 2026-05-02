#ifndef __C_WLS_H__
#define __C_WLS_H__

#include "dataset.h"
#include "machine_learning.h"
#include "linear_algebra.h"

typedef struct WLS_State {
    Matrix *weights;
    double *bias;
} WLS_State;

static int wls_fit(const ML_Model_Config_t *config, ML_Weights_t *state,
                   const dataset *ds, const size_t *feature_indices,
                   size_t n_features, size_t target_index,
                   const size_t *sample_indices, size_t n_samples,
                   const double *weights);

static int wls_predict(const ML_Weights_t *state, const dataset *ds,
                      const size_t *feature_indices, size_t n_features,
                      const size_t *sample_indices, size_t n_samples,
                      void *output);

static void wls_free(ML_Weights_t *state);

#ifdef WLS_IMPLEMENTATION

// 对角加权矩阵 W^{1/2}：result[i] = sqrt(weights[i])
static int weighted_X(const Matrix *X, const double *w_sqrt, Matrix *result) {
    if (matrix_init(result, X->rows, X->cols) != 0) return 1;
    for (size_t i = 0; i < X->rows; i++) {
        double wi = w_sqrt[i];
        for (size_t j = 0; j < X->cols; j++)
            result->mat[i * X->cols + j] = X->mat[i * X->cols + j] * wi;
    }
    return 0;
}

static int wls_fit(const ML_Model_Config_t *config, ML_Weights_t *state,
                   const dataset *ds, const size_t *feature_indices,
                   size_t n_features, size_t target_index,
                   const size_t *sample_indices, size_t n_samples,
                   const double *weights) {
    Matrix X_raw = {0}, X = {0}, y = {0}, y_weighted = {0}, theta = {0};
    double *w_sqrt = malloc(n_samples * sizeof(double));
    if (!w_sqrt) return 1;
    for (size_t i = 0; i < n_samples; i++) w_sqrt[i] = sqrt(weights[i]);

    int status = 0;
    extern int mlr_adapter_with_bias(const dataset *, const size_t *, size_t,
                                     const size_t *, size_t, Matrix *);
    if (mlr_adapter_with_bias(ds, feature_indices, n_features,
                              sample_indices, n_samples, &X_raw) != 0) {
        free(w_sqrt); return 1;
    }

    if (weighted_X(&X_raw, w_sqrt, &X) != 0) {
        free_matrix_data(&X_raw); free(w_sqrt); return 1;
    }
    free_matrix_data(&X_raw);

    if (matrix_init(&y, n_samples, 1) != 0) {
        free_matrix_data(&X); free(w_sqrt); return 1;
    }
    for (size_t i = 0; i < n_samples; i++) {
        size_t idx = sample_indices ? sample_indices[i] : i;
        y.mat[i] = ds->features[target_index].data[idx];
    }

    if (weighted_X(&y, w_sqrt, &y_weighted) != 0) {
        free_matrix_data(&X); free_matrix_data(&y); free(w_sqrt); return 1;
    }

    // 正规方程：theta = (XᵀX)⁻¹ Xᵀy_weighted
    {
        Matrix X_t = {0}, XTX = {0}, XTX_Inv = {0}, XTy = {0};
        if (matrix_transpose(&X, &X_t) != 0) { status = 1; goto cleanup; }
        if (matrix_multiply(&X_t, &X, &XTX) != 0) { status = 1; goto cleanup; }
        if (matrix_inverse(&XTX, &XTX_Inv) != 0) {
            // Matrix ill-conditioned or singular, use SVD-based pseudoinverse
            free_matrix_data(&X_t); free_matrix_data(&XTX);
            // For WLS: theta = X^+ * y_weighted where X is the weighted design matrix
            if (normal_equation_svd(&X, &y_weighted, &theta) != 0) { status = 1; goto cleanup; }
            goto wls_skip_cleanup;
        }
        if (matrix_multiply(&X_t, &y_weighted, &XTy) != 0) { status = 1; goto cleanup; }
        if (matrix_multiply(&XTX_Inv, &XTy, &theta) != 0) { status = 1; goto cleanup; }
        free_matrix_data(&X_t); free_matrix_data(&XTX); free_matrix_data(&XTX_Inv); free_matrix_data(&XTy);

wls_skip_cleanup:
        (void)0;
    }
cleanup:
        free_matrix_data(&X); free_matrix_data(&y); free_matrix_data(&y_weighted);
        free(w_sqrt);
        if (status != 0) return status;
    }

    ((WLS_State *)state->weights)->bias = malloc(sizeof(double));
    ((WLS_State *)state->weights)->weights = calloc(1, sizeof(Matrix));
    *((WLS_State *)state->weights)->bias = theta.mat[0];
    if (matrix_init(((WLS_State *)state->weights)->weights, 1, n_features) != 0) {
        free_matrix_data(&theta); return 1;
    }
    for (size_t i = 0; i < n_features; i++)
        ((WLS_State *)state->weights)->weights->mat[i] = theta.mat[i + 1];

    free_matrix_data(&theta);
    return 0;
}

static int wls_predict(const ML_Weights_t *state, const dataset *ds,
                      const size_t *feature_indices, size_t n_features,
                      const size_t *sample_indices, size_t n_samples,
                      void *output) {
    Matrix X = {0};
    double *result = (double *)output;
    WLS_State *s = (WLS_State *)state->weights;

    extern int dataset_features_to_matrix(const dataset *, const size_t *, size_t,
                                         const size_t *, size_t, Matrix *);
    if (dataset_features_to_matrix(ds, feature_indices, n_features,
                                  sample_indices, n_samples, &X) != 0)
        return 1;

    Matrix pred = {0};
    if (matrix_init(&pred, n_samples, 1) != 0) { free_matrix_data(&X); return 1; }
    if (matrix_multiply(&X, s->weights, &pred) != 0) { free_matrix_data(&X); free_matrix_data(&pred); return 1; }

    for (size_t i = 0; i < n_samples; i++)
        result[i] = pred.mat[i] + s->bias[0];

    free_matrix_data(&X); free_matrix_data(&pred);
    return 0;
}

static void wls_free(ML_Weights_t *state) {
    if (!state) return;
    WLS_State *s = (WLS_State *)state->weights;
    if (s->bias) free(s->bias);
    if (s->weights) free_matrix(s->weights);
}

#endif /* WLS_IMPLEMENTATION */

#endif /* __C_WLS_H__ */
