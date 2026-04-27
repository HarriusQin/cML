#ifndef __C_ADABOOST_H__
#define __C_ADABOOST_H__

#define MAX_CLASSES 256
#define ADABOOST_MAX_ESTIMATORS 50

#include "dataset.h"
#include "machine_learning.h"
#include "utilities.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

typedef struct AdaBoost_WeakLearner {
    int (*fit)(struct AdaBoost_WeakLearner* self, const dataset* ds,
               const size_t* indices, size_t n, const int* labels,
               const double* weights);

    int (*predict)(const struct AdaBoost_WeakLearner* self, const dataset* ds,
                   size_t sample_idx);

    void (*free_learner)(struct AdaBoost_WeakLearner* self);

    void* data;
} AdaBoost_WeakLearner;

typedef struct {
    AdaBoost_WeakLearner** learners;
    double* alphas;
    size_t n_estimators;
} AdaBoost_State;

typedef struct {
    size_t feature_index;
    double threshold;
    int left_polarity;
    int right_polarity;
} Stump;

static int stump_fit(struct AdaBoost_WeakLearner* self, const dataset* ds,
                     const size_t* indices, size_t n, const int* labels,
                     const double* weights);

static int stump_predict(const struct AdaBoost_WeakLearner* self,
                         const dataset* ds, size_t sample_idx);

static void stump_free(struct AdaBoost_WeakLearner* self);

static AdaBoost_WeakLearner* create_stump(void);

static int AdaBoost_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
                        const dataset* ds, const size_t* feature_indices,
                        size_t n_features, size_t target_index,
                        const size_t* sample_indices, size_t n_samples);

static int Adaboost_Predict(const ML_Weights_t* state, const dataset* ds,
                            const size_t* feature_indices, size_t n_features,
                            const size_t* sample_indices, size_t n_samples,
                            void* output);

static void AdaBoost_free(ML_Weights_t* state);

static ML_Model_t create_adaboost_model(void);

#ifdef ADABOOST_IMPLEMENTATION

static int stump_fit(struct AdaBoost_WeakLearner* self, const dataset* ds,
                     const size_t* indices, size_t n, const int* labels,
                     const double* weights) {
    Stump* s = (Stump*)malloc(sizeof(Stump));
    if (!s)
        return -1;

    double best_error = 1e300;
    int best_feat = 0;
    double best_thresh = 0.0;
    int best_left_pol = 1, best_right_pol = -1;

    for (size_t f = 0; f < ds->num_features; ++f) {
        FeatureSample* sorted =
            (FeatureSample*)malloc(sizeof(FeatureSample) * n);
        for (size_t i = 0; i < n; ++i) {
            sorted[i].idx = indices[i];
            sorted[i].val = ds->features[f].data[indices[i]];
        }
        qsort(sorted, n, sizeof(FeatureSample), cmp_FeatureSample);

        for (size_t spilt = 0; spilt < n - 1; ++spilt) {
            double v_curr = sorted[spilt].val, v_next = sorted[spilt + 1].val;
            if (v_curr == v_next)
                continue;

            double threshold = (v_curr + v_next) / 2.0;

            for (int pol = 1; pol >= -1; pol -= 2) {
                int left_pol = pol, right_pol = -pol;

                double err = 0.0;
                for (size_t i = 0; i < n; ++i) {
                    double v = sorted[i].val;
                    int y = labels[sorted[i].idx];
                    int pred = (v <= threshold) ? left_pol : right_pol;
                    if (pred != y)
                        err += weights[sorted[i].idx];
                }

                if (err < best_error) {
                    best_error = err;
                    best_feat = f;
                    best_thresh = threshold;
                    best_left_pol = left_pol;
                    best_right_pol = right_pol;
                }
            }
        }
        free(sorted);
    }
    s->feature_index = best_feat;
    s->left_polarity = best_left_pol;
    s->right_polarity = best_right_pol;
    s->threshold = best_thresh;
    self->data = s;
    return 0;
}

static int stump_predict(const struct AdaBoost_WeakLearner* self,
                         const dataset* ds, size_t sample_idx) {
    const Stump* s = (const Stump*)self->data;
    double v = ds->features[s->feature_index].data[sample_idx];
    return (v <= s->threshold) ? s->left_polarity : s->right_polarity;
}

static void stump_free(struct AdaBoost_WeakLearner* self) {
    if (self->data) {
        free(self->data);
        self->data = NULL;
    }
}

static AdaBoost_WeakLearner* create_stump(void) {
    AdaBoost_WeakLearner* wl =
        (AdaBoost_WeakLearner*)malloc(sizeof(AdaBoost_WeakLearner));
    if (!wl) {
        return NULL;
    }
    wl->fit = stump_fit;
    wl->predict = stump_predict;
    wl->free_learner = stump_free;
    wl->data = NULL;
    return wl;
}

static int AdaBoost_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
                        const dataset* ds, const size_t* feature_indices,
                        size_t n_features, size_t target_index,
                        const size_t* sample_indices, size_t n_samples) {
    const binary_label_column* bin_labels = &ds->binary_labels[target_index];

    size_t n_estimators = 50;

    AdaBoost_State* adaboost = (AdaBoost_State*)malloc(sizeof(AdaBoost_State));
    adaboost->learners = (AdaBoost_WeakLearner**)malloc(
        sizeof(AdaBoost_WeakLearner*) * ADABOOST_MAX_ESTIMATORS);
    adaboost->alphas =
        (double*)malloc(sizeof(double) * ADABOOST_MAX_ESTIMATORS);

    double* weights = (double*)malloc(sizeof(double) * n_samples);
    for (size_t i = 0; i < n_samples; ++i)
        weights[i] = 1.0 / n_samples;

    size_t patient_counter = 0;
    size_t final_n_estimators = 0;

    for (size_t t = 0; t < ADABOOST_MAX_ESTIMATORS; ++t) {
        double weight_sum = 0.0;
        for (size_t i = 0; i < n_samples; ++i)
            weight_sum += weights[i];
        for (size_t i = 0; i < n_samples; ++i)
            weights[i] /= weight_sum;

        adaboost->learners[t] = create_stump();
        adaboost->learners[t]->fit(adaboost->learners[t], ds, sample_indices,
                                   n_samples, bin_labels->labels, weights);

        double e = 0.0;
        for (size_t i = 0; i < n_samples; ++i) {
            int pred = adaboost->learners[t]->predict(adaboost->learners[t], ds,
                                                      sample_indices[i]);
            if (pred != bin_labels->labels[sample_indices[i]])
                e += weights[i];
        }

        if (e < 1e-10)
            e = 1e-10;

        if (e >= 0.5) {
            for (size_t j = 0; j < t; ++j)
                adaboost->learners[j]->free_learner(adaboost->learners[j]);
            free(adaboost->learners);
            free(adaboost->alphas);
            free(adaboost);
            free(weights);
            state->weights = NULL;
            state->size = 0;
            return -1;
        }

        adaboost->alphas[t] = 0.5 * log((1.0 - e) / e);

        int all_correct = 1;
        for (size_t i = 0; i < n_samples; ++i) {
            int pred = adaboost->learners[t]->predict(adaboost->learners[t], ds,
                                                      sample_indices[i]);
            int y = bin_labels->labels[sample_indices[i]];
            if (pred == y)
                weights[i] *= exp(-adaboost->alphas[t]);
            else {
                weights[i] *= exp(adaboost->alphas[t]);
                all_correct = 0;
            }
        }

        if (all_correct) {
            final_n_estimators = t + 1;
            break;
        }

        final_n_estimators = t + 1;
    }
    free(weights);
    adaboost->n_estimators = final_n_estimators;
    state->weights = adaboost;
    state->size = sizeof(AdaBoost_State);

    return 0;
}

static int Adaboost_Predict(const ML_Weights_t* state, const dataset* ds,
                            const size_t* feature_indices, size_t n_features,
                            const size_t* sample_indices, size_t n_samples,
                            void* output) {

    AdaBoost_State* adaboost = (AdaBoost_State*)state->weights;
    int* predictions = (int*)output;

    for (size_t s = 0; s < n_samples; s++) {
        size_t idx = sample_indices[s];

        double weighted_sum = 0.0;
        for (size_t t = 0; t < adaboost->n_estimators; t++) {
            int pred =
                adaboost->learners[t]->predict(adaboost->learners[t], ds, idx);
            weighted_sum += adaboost->alphas[t] * pred;
        }

        predictions[s] = (weighted_sum >= 0) ? 1 : -1;
    }
    return 0;
}

static void AdaBoost_free(ML_Weights_t* state) {
    if (state->weights) {
        AdaBoost_State* adaboost = (AdaBoost_State*)state->weights;
        for (size_t t = 0; t < adaboost->n_estimators; t++) {
            adaboost->learners[t]->free_learner(adaboost->learners[t]);
            free(adaboost->learners[t]);
        }
        free(adaboost->learners);
        free(adaboost->alphas);
        free(adaboost);
        state->weights = NULL;
    }
}

static ML_Model_t create_adaboost_model(void) {
    return (ML_Model_t){.type = ML_BINARY_CLASSIFICATION,
                        .config = {NULL, 0},
                        .state = {NULL, 0},
                        .methods = {.fit = AdaBoost_fit,
                                    .predict = Adaboost_Predict,
                                    .predict_proba = NULL,
                                    .get_coefficients = NULL,
                                    .serialize = NULL,
                                    .deserialize = NULL,
                                    .free_state = AdaBoost_free}};
}

#endif /* ADABOOST_IMPLEMENTATION */

#endif /* __C_ADABOOST_H__ */
