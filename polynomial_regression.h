#ifndef __C_POLYNOMIAL_REGRESSION_H__
#define __C_POLYNOMIAL_REGRESSION_H__

#include "dataset.h"
#include "machine_learning.h"
#include "linear_algebra.h"

typedef struct PolyReg_State {
    Matrix *weights;
    double *bias;
    size_t degree;
} PolyReg_State;

static int polynomial_features(
    const double *x, size_t n_samples,
    size_t degree,
    Matrix *result
);

static int poly_fit(const ML_Model_Config_t *config, ML_Weights_t *state,
                    const double *x, const double *y, size_t n_samples,
                    size_t degree);

static int poly_predict(const ML_Weights_t *state,
                       const double *x, size_t n_samples,
                       double *output);

static void poly_free(ML_Weights_t *state);

#ifdef POLYNOMIAL_REGRESSION_IMPLEMENTATION

static int polynomial_features(
    const double *x, size_t n_samples,
    size_t degree,
    Matrix *result
) {
    if (!x || !result || degree < 1) return 1;
    if (matrix_init(result, n_samples, degree) != 0) return 1;
    for (size_t i = 0; i < n_samples; i++) {
        double x_pow = x[i];  // x^1
        for (size_t d = 0; d < degree; d++) {
            result->mat[i * degree + d] = x_pow;
            x_pow *= x[i];    // x^(d+1) -> x^(d+2)
        }
    }
    return 0;
}

static int poly_fit(const ML_Model_Config_t *config, ML_Weights_t *state,
                    const double *x, const double *y, size_t n_samples,
                    size_t degree) {
    Matrix X = {0}, X_aug = {0}, y_mat = {0}, theta = {0};
    if (polynomial_features(x, n_samples, degree, &X) != 0) return 1;

    // 添加偏置列: X_aug = [1 | x^1 | x^2 | ...]
    if (matrix_init(&X_aug, n_samples, degree + 1) != 0) {
        free_matrix(&X); return 1;
    }
    for (size_t i = 0; i < n_samples; i++) {
        X_aug.mat[i * X_aug.cols] = 1.0;  // 偏置列
        for (size_t j = 0; j < degree; j++)
            X_aug.mat[i * X_aug.cols + j + 1] = X.mat[i * degree + j];
    }

    if (matrix_init(&y_mat, n_samples, 1) != 0) {
        free_matrix(&X); free_matrix(&X_aug); return 1;
    }
    for (size_t i = 0; i < n_samples; i++)
        y_mat.mat[i] = y[i];

    // 正规方程: theta = (X^T X)^{-1} X^T y
    {
        Matrix X_t = {0}, XTX = {0}, XTX_Inv = {0}, XTy = {0};
        if (matrix_transpose(&X_aug, &X_t) != 0) goto cleanup;
        if (matrix_multiply(&X_t, &X_aug, &XTX) != 0) { free_matrix_data(&X_t); goto cleanup; }
        if (matrix_inverse(&XTX, &XTX_Inv) != 0) {
            // Matrix ill-conditioned or singular, use SVD-based pseudoinverse
            free_matrix_data(&X_t); free_matrix_data(&XTX);
            if (normal_equation_svd(&X_aug, &y_mat, &theta) != 0) goto cleanup;
            goto skip_cleanup;
        }
        if (matrix_multiply(&X_t, &y_mat, &XTy) != 0) { free_matrix_data(&X_t); free_matrix_data(&XTX); free_matrix_data(&XTX_Inv); goto cleanup; }
        if (matrix_multiply(&XTX_Inv, &XTy, &theta) != 0) { free_matrix_data(&X_t); free_matrix_data(&XTX); free_matrix_data(&XTX_Inv); free_matrix_data(&XTy); goto cleanup; }
        free_matrix_data(&X_t); free_matrix_data(&XTX); free_matrix_data(&XTX_Inv); free_matrix_data(&XTy);
    skip_cleanup:
        (void)0;
    }

    ((PolyReg_State *)state->weights)->bias = malloc(sizeof(double));
    ((PolyReg_State *)state->weights)->weights = calloc(1, sizeof(Matrix));
    ((PolyReg_State *)state->weights)->degree = degree;
    *((PolyReg_State *)state->weights)->bias = theta.mat[0];
    if (matrix_init(((PolyReg_State *)state->weights)->weights, 1, degree) != 0) {
        free(((PolyReg_State *)state->weights)->bias); goto cleanup;
    }
    for (size_t i = 0; i < degree; i++)
        ((PolyReg_State *)state->weights)->weights->mat[i] = theta.mat[i + 1];

cleanup:
    free_matrix_data(&X); free_matrix_data(&X_aug); free_matrix_data(&y_mat); free_matrix_data(&theta);
    return 0;
}

static int poly_predict(const ML_Weights_t *state,
                       const double *x, size_t n_samples,
                       double *output) {
    if (!state || !x || !output) return 1;
    PolyReg_State *s = (PolyReg_State *)state->weights;
    Matrix X = {0};
    if (polynomial_features(x, n_samples, s->degree, &X) != 0) return 1;
    for (size_t i = 0; i < n_samples; i++) {
        double pred = s->bias[0];
        for (size_t j = 0; j < s->degree; j++)
            pred += s->weights->mat[j] * X.mat[i * s->degree + j];
        output[i] = pred;
    }
    free_matrix_data(&X);
    return 0;
}

static void poly_free(ML_Weights_t *state) {
    if (!state) return;
    PolyReg_State *s = (PolyReg_State *)state->weights;
    if (s->bias) free(s->bias);
    if (s->weights) free_matrix(s->weights);
}

#endif /* POLYNOMIAL_REGRESSION_IMPLEMENTATION */

#endif /* __C_POLYNOMIAL_REGRESSION_H__ */
