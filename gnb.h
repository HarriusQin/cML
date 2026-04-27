#ifndef __C_GNB_H__
#define __C_GNB_H__

#include <stddef.h>
#include <stdlib.h>
#include <math.h>

#include "machine_learning.h"

/* ============================================================================
 * Gaussian Naive Bayes
 * ============================================================================ */

typedef struct GNB_State {
    double* means;
    double* variances;
    size_t n_classes;
    size_t n_features;
    double* class_priors;
} GNB_State;

static int gnb_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
                   const dataset* ds, const size_t* feature_indices, size_t n_features,
                   size_t target_index, const size_t* sample_indices, size_t n_samples) {
    (void)config;
    size_t n_classes = ds->labels[target_index].classes;
    GNB_State* gnb = (GNB_State*)malloc(sizeof(GNB_State));
    if (!gnb) return -1;

    gnb->n_classes = n_classes;
    gnb->n_features = n_features;
    gnb->means = (double*)calloc(n_classes * n_features, sizeof(double));
    gnb->variances = (double*)calloc(n_classes * n_features, sizeof(double));
    gnb->class_priors = (double*)calloc(n_classes, sizeof(double));

    size_t* class_counts = (size_t*)calloc(n_classes, sizeof(size_t));
    for (size_t i = 0; i < n_samples; i++) {
        int label = ds->labels[target_index].labels[sample_indices[i]];
        class_counts[label]++;
    }

    for (size_t c = 0; c < n_classes; c++)
        gnb->class_priors[c] = (double)class_counts[c] / n_samples;

    for (size_t i = 0; i < n_samples; i++) {
        int label = ds->labels[target_index].labels[sample_indices[i]];
        for (size_t f = 0; f < n_features; f++)
            gnb->means[label * n_features + f] += ds->features[feature_indices[f]].data[sample_indices[i]];
    }
    for (size_t c = 0; c < n_classes; c++)
        if (class_counts[c] > 0)
            for (size_t f = 0; f < n_features; f++)
                gnb->means[c * n_features + f] /= class_counts[c];

    for (size_t i = 0; i < n_samples; i++) {
        int label = ds->labels[target_index].labels[sample_indices[i]];
        for (size_t f = 0; f < n_features; f++) {
            double diff = ds->features[feature_indices[f]].data[sample_indices[i]] - gnb->means[label * n_features + f];
            gnb->variances[label * n_features + f] += diff * diff;
        }
    }
    for (size_t c = 0; c < n_classes; c++)
        if (class_counts[c] > 0)
            for (size_t f = 0; f < n_features; f++) {
                gnb->variances[c * n_features + f] /= class_counts[c];
                if (gnb->variances[c * n_features + f] < 1e-10) gnb->variances[c * n_features + f] = 1.0;
            }

    free(class_counts);
    state->weights = gnb;
    state->size = sizeof(GNB_State);
    return 0;
}

static double gnb_gaussian_pdf(double x, double mean, double variance) {
    double coeff = 1.0 / sqrt(2.0 * 3.14159265358979 * variance);
    double exponent = -((x - mean) * (x - mean)) / (2.0 * variance);
    return coeff * exp(exponent);
}

static int gnb_predict(const ML_Weights_t* state, const dataset* ds,
                       const size_t* feature_indices, size_t n_features,
                       const size_t* sample_indices, size_t n_samples, void* output) {
    GNB_State* gnb = (GNB_State*)state->weights;
    int* predictions = (int*)output;
    for (size_t s = 0; s < n_samples; s++) {
        size_t idx = sample_indices[s];
        double best_log_prob = -1e300;
        int best_class = 0;
        for (size_t c = 0; c < gnb->n_classes; c++) {
            double log_prob = log(gnb->class_priors[c] + 1e-10);
            for (size_t f = 0; f < n_features; f++) {
                double x = ds->features[feature_indices[f]].data[idx];
                log_prob += log(gnb_gaussian_pdf(x, gnb->means[c * n_features + f],
                                                  gnb->variances[c * n_features + f]) + 1e-10);
            }
            if (log_prob > best_log_prob) { best_log_prob = log_prob; best_class = (int)c; }
        }
        predictions[s] = best_class;
    }
    return 0;
}

static void gnb_free(ML_Weights_t* state) {
    if (state->weights) {
        GNB_State* gnb = (GNB_State*)state->weights;
        free(gnb->means); free(gnb->variances); free(gnb->class_priors); free(gnb);
        state->weights = NULL;
    }
}

static ML_Model_t create_gnb_model(void) {
    return (ML_Model_t){
        .type = ML_CLASSIFICATION,
        .config = {NULL, 0}, .state = {NULL, 0},
        .methods = { .fit = gnb_fit, .predict = gnb_predict,
                     .predict_proba = NULL, .get_coefficients = NULL,
                     .serialize = NULL, .deserialize = NULL, .free_state = gnb_free }
    };
}

#endif /* __C_GNB_H__ */