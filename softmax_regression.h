#ifndef __C_SOFTMAX_REGRESSION_H__
#define __C_SOFTMAX_REGRESSION_H__

#include "dataset.h"
#include "machine_learning.h"
#include "linear_algebra.h"

typedef struct SoftmaxReg_State {
    Matrix *weights;     // n_features × n_classes weight matrix
    double *bias;        // n_classes bias vector
    size_t n_classes;    // number of classes
    size_t n_features;   // number of features
} SoftmaxReg_State;

typedef struct SoftmaxReg_Config {
    double learning_rate;    // learning rate (default: 0.01)
    size_t max_iter;         // max iterations (default: 1000)
    double tolerance;        // convergence tolerance (default: 1e-6)
    bool verbose;            // print progress
} SoftmaxReg_Config;

static int softmax_fit(const ML_Model_Config_t *config, ML_Weights_t *state,
                     const dataset *ds, const size_t *feature_indices,
                     size_t n_features, size_t target_index,
                     const size_t *sample_indices, size_t n_samples);

static int softmax_fit_gd(const ML_Model_Config_t *config, ML_Weights_t *state,
                        const dataset *ds, const size_t *feature_indices,
                        size_t n_features, size_t target_index,
                        const size_t *sample_indices, size_t n_samples);

static int softmax_predict(const ML_Weights_t *state, const dataset *ds,
                         const size_t *feature_indices, size_t n_features,
                         const size_t *sample_indices, size_t n_samples,
                         void *output);

static int softmax_predict_proba(const ML_Weights_t *state, const dataset *ds,
                               const size_t *feature_indices, size_t n_features,
                               const size_t *sample_indices, size_t n_samples,
                               size_t n_classes, void *output);

static int softmax_get_coefficients(const ML_Weights_t *state, void **coeffs, size_t *size);

static void softmax_free(ML_Weights_t *state);

static ML_Model_t create_softmax_model(void);
static ML_Model_t create_softmax_model_gd(void);

#ifdef SOFTMAX_REGRESSION_IMPLEMENTATION

static void softmax(double *z, size_t n_classes) {
    double max_z = z[0];
    for (size_t k = 1; k < n_classes; k++)
        if (z[k] > max_z) max_z = z[k];

    double sum = 0.0;
    for (size_t k = 0; k < n_classes; k++) {
        z[k] = exp(z[k] - max_z);
        sum += z[k];
    }
    for (size_t k = 0; k < n_classes; k++)
        z[k] /= sum;
}

// Softmax regression using normal equation (closed-form solution)
// Trains K binary classifiers using One-vs-All strategy
static int softmax_fit(const ML_Model_Config_t *config, ML_Weights_t *state,
                     const dataset *ds, const size_t *feature_indices,
                     size_t n_features, size_t target_index,
                     const size_t *sample_indices, size_t n_samples) {
    (void)config;

    label_column *labels = &ds->labels[target_index];
    size_t K = labels->classes;

    // Allocate state
    SoftmaxReg_State *s = calloc(1, sizeof(SoftmaxReg_State));
    if (!s) return 1;
    s->n_classes = K;
    s->n_features = n_features;
    s->weights = calloc(1, sizeof(Matrix));
    s->bias = calloc(K, sizeof(double));
    if (!s->weights || !s->bias) { goto error; }
    if (matrix_init(s->weights, n_features, K) != 0) { goto error; }

    // Prepare feature matrix X (with bias column)
    Matrix X_raw = {0};
    extern int mlr_adapter_with_bias(const dataset*, const size_t*, size_t,
                                    const size_t*, size_t, Matrix*);
    if (mlr_adapter_with_bias(ds, feature_indices, n_features,
                              sample_indices, n_samples, &X_raw) != 0) {
        goto error_raw;
    }

    // For each class k, train a binary classifier using normal equation
    for (size_t k = 0; k < K; k++) {
        double *y_k = malloc(n_samples * sizeof(double));
        if (!y_k) { goto error_k; }
        for (size_t i = 0; i < n_samples; i++) {
            size_t idx = sample_indices ? sample_indices[i] : i;
            int label = labels->labels[idx];
            y_k[i] = (label == (int)k) ? 1.0 : 0.0;
        }

        Matrix y_mat = {0};
        if (matrix_init(&y_mat, n_samples, 1) != 0) {
            free(y_k);
            goto error_k;
        }
        memcpy(y_mat.mat, y_k, n_samples * sizeof(double));
        free(y_k);

        Matrix theta = {0};
        extern int normal_equation_svd(const Matrix*, const Matrix*, Matrix*);
        if (normal_equation_svd(&X_raw, &y_mat, &theta) != 0) {
            free_matrix_data(&y_mat);
            goto error_k;
        }

        // Store bias and weights
        s->bias[k] = theta.mat[0];
        for (size_t j = 0; j < n_features; j++)
            s->weights->mat[j * K + k] = theta.mat[j + 1];

        free_matrix_data(&y_mat);
        free_matrix_data(&theta);
        continue;

    error_k:
        free_matrix_data(&X_raw);
        goto error;
    }

    free_matrix_data(&X_raw);
    state->weights = s;
    state->size = sizeof(SoftmaxReg_State);
    return 0;

error_raw:
    free_matrix_data(&X_raw);
error:
    if (s->weights) { free_matrix_data(s->weights); free(s->weights); }
    if (s->bias) free(s->bias);
    free(s);
    return 1;
}

static int softmax_fit_gd(const ML_Model_Config_t *config, ML_Weights_t *state,
                        const dataset *ds, const size_t *feature_indices,
                        size_t n_features, size_t target_index,
                        const size_t *sample_indices, size_t n_samples) {
    // Parse config
    double lr = 0.01;
    size_t max_iter = 1000;
    double tol = 1e-6;
    int verbose = 0;
    if (config && config->params) {
        SoftmaxReg_Config *c = (SoftmaxReg_Config *)config->params;
        lr = c->learning_rate > 0 ? c->learning_rate : 0.01;
        max_iter = c->max_iter > 0 ? c->max_iter : 1000;
        tol = c->tolerance > 0 ? c->tolerance : 1e-6;
        verbose = c->verbose;
    }

    label_column *labels = &ds->labels[target_index];
    size_t K = labels->classes;

    // Allocate state
    SoftmaxReg_State *s = calloc(1, sizeof(SoftmaxReg_State));
    if (!s) return 1;
    s->n_classes = K;
    s->n_features = n_features;
    s->weights = calloc(1, sizeof(Matrix));
    s->bias = calloc(K, sizeof(double));
    if (!s->weights || !s->bias) { goto error; }
    if (matrix_init(s->weights, n_features, K) != 0) { goto error; }

    // Initialize weights and bias to zero
    for (size_t i = 0; i < n_features * K; i++)
        s->weights->mat[i] = 0.0;
    for (size_t k = 0; k < K; k++)
        s->bias[k] = 0.0;

    // Extract feature matrix X (n_samples × n_features)
    Matrix X = {0};
    extern int dataset_features_to_matrix(const dataset*, const size_t*, size_t,
                                         const size_t*, size_t, Matrix*);
    if (dataset_features_to_matrix(ds, feature_indices, n_features,
                                 sample_indices, n_samples, &X) != 0) {
        goto error;
    }

    // Extract labels
    int *y = malloc(n_samples * sizeof(int));
    if (!y) { free_matrix_data(&X); goto error; }
    for (size_t i = 0; i < n_samples; i++) {
        size_t idx = sample_indices ? sample_indices[i] : i;
        y[i] = labels->labels[idx];
    }

    // Gradient descent iterations
    double prev_loss = 1e300;
    double *probs = malloc(n_samples * K * sizeof(double));
    double *d_weights = calloc(n_features * K, sizeof(double));
    double *d_bias = calloc(K, sizeof(double));

    if (!probs || !d_weights || !d_bias) {
        free(probs); free(d_weights); free(d_bias);
        free(y); free_matrix_data(&X);
        goto error;
    }

    for (size_t iter = 0; iter < max_iter; iter++) {
        // Forward pass: compute probabilities
        for (size_t i = 0; i < n_samples; i++) {
            double z[256];  // max 256 classes
            for (size_t k = 0; k < K; k++) {
                z[k] = s->bias[k];
                for (size_t j = 0; j < n_features; j++)
                    z[k] += X.mat[i * n_features + j] * s->weights->mat[j * K + k];
            }
            softmax(z, K);
            for (size_t k = 0; k < K; k++)
                probs[i * K + k] = z[k];
        }

        // Compute cross-entropy loss
        double loss = 0.0;
        for (size_t i = 0; i < n_samples; i++) {
            double p = probs[i * K + y[i]];
            if (p > 1e-15)  // avoid log(0)
                loss -= log(p);
        }
        loss /= n_samples;

        // Check convergence
        if (fabs(prev_loss - loss) < tol) {
            if (verbose) printf("Converged at iter %zu, loss=%.6f\n", iter, loss);
            break;
        }
        prev_loss = loss;

        if (verbose && (iter % 100 == 0))
            printf("iter %zu: loss=%.6f\n", iter, loss);

        // Compute gradients
        // d_weights[j*K + k] = sum_i x_ij * (p_ik - y_ik) / n
        // d_bias[k] = sum_i (p_ik - y_ik) / n
        memset(d_weights, 0, n_features * K * sizeof(double));
        memset(d_bias, 0, K * sizeof(double));

        for (size_t i = 0; i < n_samples; i++) {
            for (size_t k = 0; k < K; k++) {
                double delta = probs[i * K + k] - (k == y[i] ? 1.0 : 0.0);
                d_bias[k] += delta;
                for (size_t j = 0; j < n_features; j++)
                    d_weights[j * K + k] += X.mat[i * n_features + j] * delta;
            }
        }

        // Normalize gradients
        for (size_t k = 0; k < K; k++) {
            d_bias[k] /= n_samples;
            for (size_t j = 0; j < n_features; j++)
                d_weights[j * K + k] /= n_samples;
        }

        // Update weights and bias
        for (size_t k = 0; k < K; k++) {
            s->bias[k] -= lr * d_bias[k];
            for (size_t j = 0; j < n_features; j++)
                s->weights->mat[j * K + k] -= lr * d_weights[j * K + k];
        }
    }

    free(probs);
    free(d_weights);
    free(d_bias);
    free(y);
    free_matrix_data(&X);
    state->weights = s;
    state->size = sizeof(SoftmaxReg_State);
    return 0;

error:
    if (s->weights) { free_matrix_data(s->weights); free(s->weights); }
    if (s->bias) free(s->bias);
    free(s);
    return 1;
}

static int softmax_predict_proba(const ML_Weights_t *state, const dataset *ds,
                               const size_t *feature_indices, size_t n_features,
                               const size_t *sample_indices, size_t n_samples,
                               size_t n_classes, void *output) {
    if (!state || !state->weights || !ds || !output) return 1;
    SoftmaxReg_State *s = (SoftmaxReg_State *)state->weights;

    Matrix X = {0};
    extern int dataset_features_to_matrix(const dataset*, const size_t*, size_t,
                                         const size_t*, size_t, Matrix*);
    if (dataset_features_to_matrix(ds, feature_indices, n_features,
                                 sample_indices, n_samples, &X) != 0)
        return 1;

    double *probs = (double *)output;

    for (size_t i = 0; i < n_samples; i++) {
        double z[256];
        for (size_t k = 0; k < n_classes; k++) {
            z[k] = s->bias[k];
            for (size_t j = 0; j < n_features; j++)
                z[k] += X.mat[i * n_features + j] * s->weights->mat[j * n_classes + k];
        }
        softmax(z, n_classes);
        for (size_t k = 0; k < n_classes; k++)
            probs[i * n_classes + k] = z[k];
    }

    free_matrix_data(&X);
    return 0;
}

static int softmax_predict(const ML_Weights_t *state, const dataset *ds,
                         const size_t *feature_indices, size_t n_features,
                         const size_t *sample_indices, size_t n_samples,
                         void *output) {
    if (!state || !state->weights || !ds || !output) return 1;
    SoftmaxReg_State *s = (SoftmaxReg_State *)state->weights;

    int *predictions = (int *)output;
    double *probs = malloc(n_samples * s->n_classes * sizeof(double));
    if (!probs) return 1;

    if (softmax_predict_proba(state, ds, feature_indices, n_features,
                             sample_indices, n_samples, s->n_classes, probs) != 0) {
        free(probs);
        return 1;
    }

    for (size_t i = 0; i < n_samples; i++) {
        size_t best_class = 0;
        double best_prob = probs[i * s->n_classes];
        for (size_t k = 1; k < s->n_classes; k++) {
            if (probs[i * s->n_classes + k] > best_prob) {
                best_prob = probs[i * s->n_classes + k];
                best_class = k;
            }
        }
        predictions[i] = (int)best_class;
    }

    free(probs);
    return 0;
}

static int softmax_get_coefficients(const ML_Weights_t *state, void **coeffs, size_t *size) {
    if (!state || !state->weights) return 1;
    SoftmaxReg_State *s = (SoftmaxReg_State *)state->weights;

    size_t total = s->n_classes + s->n_features * s->n_classes;
    double *c = malloc(total * sizeof(double));
    if (!c) return 1;

    for (size_t k = 0; k < s->n_classes; k++)
        c[k] = s->bias[k];

    for (size_t k = 0; k < s->n_classes; k++) {
        for (size_t j = 0; j < s->n_features; j++)
            c[s->n_classes + k * s->n_features + j] = s->weights->mat[j * s->n_classes + k];
    }

    *coeffs = c;
    *size = total;
    return 0;
}

static void softmax_free(ML_Weights_t *state) {
    if (!state || !state->weights) return;
    SoftmaxReg_State *s = (SoftmaxReg_State *)state->weights;
    if (s->weights) { free_matrix_data(s->weights); free(s->weights); }
    if (s->bias) free(s->bias);
    free(s);
    state->weights = NULL;
    state->size = 0;
}

static ML_Model_t create_softmax_model_gd(void) {
    return (ML_Model_t){
        .type = ML_CLASSIFICATION,
        .config = {NULL, 0},
        .state = {NULL, 0},
        .methods = (ML_Model_Impl_t){
            .fit = softmax_fit_gd,
            .predict = softmax_predict,
            .predict_proba = softmax_predict_proba,
            .get_coefficients = softmax_get_coefficients,
            .serialize = NULL,
            .deserialize = NULL,
            .free_state = softmax_free
        }
    };
}

static ML_Model_t create_softmax_model(void) {
    return (ML_Model_t){
        .type = ML_CLASSIFICATION,
        .config = {NULL, 0},
        .state = {NULL, 0},
        .methods = (ML_Model_Impl_t){
            .fit = softmax_fit,
            .predict = softmax_predict,
            .predict_proba = softmax_predict_proba,
            .get_coefficients = softmax_get_coefficients,
            .serialize = NULL,
            .deserialize = NULL,
            .free_state = softmax_free
        }
    };
}

#endif /* SOFTMAX_REGRESSION_IMPLEMENTATION */

#endif /* __C_SOFTMAX_REGRESSION_H__ */
