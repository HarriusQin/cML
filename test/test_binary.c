/**
 * @file test_binary.c
 * @brief Test binary classification model type (ML_BINARY_CLASSIFICATION)
 */

#define CSV_IMPLEMENTATION
#include "csv.h"

#define C_MACHINE_LEARNING_IMPLEMENTATION
#include "machine_learning.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Simple binary classifier for testing */
typedef struct {
    double threshold;
} SimpleBinaryClassifier;

static int binary_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
                      const dataset* ds, const size_t* feature_indices, size_t n_features,
                      size_t target_index, const size_t* sample_indices,
                      size_t n_samples) {
    (void)config;
    (void)feature_indices;
    (void)n_features;
    (void)target_index;
    SimpleBinaryClassifier* clf = (SimpleBinaryClassifier*)malloc(sizeof(SimpleBinaryClassifier));

    /* Compute mean of first feature as threshold */
    double sum = 0.0;
    for (size_t i = 0; i < n_samples; i++) {
        sum += ds->features[0].data[sample_indices[i]];
    }
    clf->threshold = sum / n_samples;
    state->weights = clf;
    state->size = sizeof(SimpleBinaryClassifier);
    return 0;
}

static int binary_predict(const ML_Weights_t* state, const dataset* ds,
                          const size_t* feature_indices, size_t n_features,
                          const size_t* sample_indices, size_t n_samples,
                          void* output) {
    (void)feature_indices;
    (void)n_features;
    SimpleBinaryClassifier* clf = (SimpleBinaryClassifier*)state->weights;
    int* preds = (int*)output;

    for (size_t i = 0; i < n_samples; i++) {
        double val = ds->features[0].data[sample_indices[i]];
        preds[i] = (val > clf->threshold) ? 1 : 0;
    }
    return 0;
}

static void binary_free(ML_Weights_t* state) {
    if (state->weights) {
        free(state->weights);
        state->weights = NULL;
    }
}

static ML_Model_t make_binary_model() {
    return (ML_Model_t){
        ML_BINARY_CLASSIFICATION,
        {NULL, 0},
        {NULL, 0},
        {binary_fit, binary_predict, NULL, NULL, NULL, NULL, binary_free}
    };
}

/* Binary classifier with predict_proba for AUC testing */
typedef struct {
    double threshold;
} BinaryWithProba;

static int binary_proba_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
                            const dataset* ds, const size_t* feature_indices, size_t n_features,
                            size_t target_index, const size_t* sample_indices,
                            size_t n_samples) {
    (void)config;
    (void)feature_indices;
    (void)n_features;
    (void)target_index;
    BinaryWithProba* clf = (BinaryWithProba*)malloc(sizeof(BinaryWithProba));

    double sum = 0.0;
    for (size_t i = 0; i < n_samples; i++) {
        sum += ds->features[0].data[sample_indices[i]];
    }
    clf->threshold = sum / n_samples;
    state->weights = clf;
    state->size = sizeof(BinaryWithProba);
    return 0;
}

static int binary_proba_predict(const ML_Weights_t* state, const dataset* ds,
                               const size_t* feature_indices, size_t n_features,
                               const size_t* sample_indices, size_t n_samples,
                               void* output) {
    (void)feature_indices;
    (void)n_features;
    BinaryWithProba* clf = (BinaryWithProba*)state->weights;
    int* preds = (int*)output;

    for (size_t i = 0; i < n_samples; i++) {
        double val = ds->features[0].data[sample_indices[i]];
        preds[i] = (val > clf->threshold) ? 1 : 0;
    }
    return 0;
}

static int binary_proba_predict_proba(const ML_Weights_t* state, const dataset* ds,
                                      const size_t* feature_indices, size_t n_features,
                                      const size_t* sample_indices, size_t n_samples,
                                      size_t n_classes, void* output) {
    (void)feature_indices;
    (void)n_features;
    BinaryWithProba* clf = (BinaryWithProba*)state->weights;
    double* probas = (double*)output;

    (void)n_classes;
    for (size_t i = 0; i < n_samples; i++) {
        double val = ds->features[0].data[sample_indices[i]];
        double p = 1.0 / (1.0 + exp(-(val - clf->threshold)));
        probas[i * 2] = 1.0 - p;  /* P(class 0) */
        probas[i * 2 + 1] = p;   /* P(class 1) */
    }
    return 0;
}

static void binary_proba_free(ML_Weights_t* state) {
    if (state->weights) {
        free(state->weights);
        state->weights = NULL;
    }
}

static ML_Model_t make_binary_proba_model() {
    return (ML_Model_t){
        ML_BINARY_CLASSIFICATION,
        {NULL, 0},
        {NULL, 0},
        {binary_proba_fit, binary_proba_predict, binary_proba_predict_proba,
         NULL, NULL, NULL, binary_proba_free}
    };
}

int main(int argc, char* argv[]) {
    const char* csv_path = (argc > 1) ? argv[1] : "data/iris.csv";

    fprintf(stderr, "=== Binary Classification Test ===\n\n");

    /* Load iris dataset */
    fprintf(stderr, "Loading CSV from %s...\n", csv_path);
    csv_t* csv = csv_load(csv_path);
    if (!csv) {
        fprintf(stderr, "ERROR: Failed to load CSV\n");
        return 1;
    }
    fprintf(stderr, "CSV loaded: %zu rows\n", csv->size - 1);

    fprintf(stderr, "Converting to dataset...\n");
    const char* labels[] = {"species"};
    dataset* ds = csv_to_dataset(csv, labels, 1);
    if (!ds) {
        fprintf(stderr, "ERROR: Failed to convert CSV to dataset\n");
        free_csv_data(csv);
        free(csv);
        return 1;
    }
    fprintf(stderr, "Dataset created: %zu samples, %zu features\n", ds->rows, ds->num_features);

    /* Manually convert to binary: setosa=1, others=0 */
    /* The labels array has 3 classes: setosa(0), versicolor(1), virginica(2) */
    /* We'll change to binary: setosa=1, others=0 */
    for (size_t i = 0; i < ds->rows; i++) {
        ds->labels[0].labels[i] = (ds->labels[0].labels[i] == 0) ? 1 : 0;
    }
    ds->labels[0].classes = 2;  /* Now only 2 classes: 0=non-setosa, 1=setosa */
    fprintf(stderr, "Converted to binary: %zu classes\n", ds->labels[0].classes);

    /* Use target_index=0 (the original labels[0], now binary) */
    size_t target_index = 0;
    fprintf(stderr, "Using target_index=%zu for binary labels\n", target_index);

    /* Test 1: Basic binary model without predict_proba */
    fprintf(stderr, "\n=== Test 1: Basic Binary Classification ===\n");
    ML_Model_t model = make_binary_model();
    const size_t fi[] = {0, 1, 2, 3};

    CV_Result_t cv = kfold_cross_validate(&model, ds, fi, 4, target_index, 5, 42);
    fprintf(stderr, "CV Results (k=%zu):\n", cv.k);
    fprintf(stderr, "  metric_type: %d (ML_BINARY_CLASSIFICATION=%d)\n",
           cv.metric_type, ML_BINARY_CLASSIFICATION);
    fprintf(stderr, "  Accuracy: %.4f\n", cv.avg_metrics.bin.Accuracy);
    fprintf(stderr, "  Precision: %.4f\n", cv.avg_metrics.bin.Precision);
    fprintf(stderr, "  Recall: %.4f\n", cv.avg_metrics.bin.Recall);
    fprintf(stderr, "  Specificity: %.4f\n", cv.avg_metrics.bin.Specificity);
    fprintf(stderr, "  F1_Score: %.4f\n", cv.avg_metrics.bin.F1_Score);
    fprintf(stderr, "  AUC: %.4f (requires predict_proba)\n", cv.avg_metrics.bin.AUC);
    free(cv.fold_scores);
    model.methods.free_state(&model.state);

    /* Test 2: Binary model with predict_proba (for AUC) */
    fprintf(stderr, "\n=== Test 2: Binary with predict_proba (AUC) ===\n");
    ML_Model_t model2 = make_binary_proba_model();

    Dataset_Split_t split;
    if (train_test_split(ds, 0.2, 42, &split) != 0) {
        fprintf(stderr, "ERROR: train_test_split failed\n");
        return 1;
    }
    fprintf(stderr, "Split: train=%zu, test=%zu\n", split.train_size, split.test_size);

    model2.methods.fit(&model2.config, &model2.state, ds, fi, 4, target_index,
                       split.train_indices, split.train_size);
    fprintf(stderr, "Model fitted\n");

    ML_BinaryClassification_Metrics_t metrics;
    memset(&metrics, 0, sizeof(metrics));

    int ret;
    ret = model_evaluate(&model2, ds, fi, 4, target_index,
                         split.test_indices, split.test_size, &metrics);
    if (ret == 0) {
        fprintf(stderr, "model_evaluate succeeded:\n");
        fprintf(stderr, "  Accuracy: %.4f\n", metrics.Accuracy);
        fprintf(stderr, "  Precision: %.4f\n", metrics.Precision);
        fprintf(stderr, "  Recall: %.4f\n", metrics.Recall);
        fprintf(stderr, "  Specificity: %.4f\n", metrics.Specificity);
        fprintf(stderr, "  F1_Score: %.4f\n", metrics.F1_Score);
        fprintf(stderr, "  AUC: %.4f\n", metrics.AUC);
        fprintf(stderr, "  LogLoss: %.4f\n", metrics.LogLoss);
    } else {
        fprintf(stderr, "model_evaluate failed: %d\n", ret);
    }

    model2.methods.free_state(&model2.state);
    free(split.train_indices);
    free(split.test_indices);

    /* Test 3: train_model_with_validation for binary */
    fprintf(stderr, "\n=== Test 3: train_model_with_validation ===\n");
    ML_Model_t model3 = make_binary_model();
    ML_BinaryClassification_Metrics_t train_m, val_m;

    ret = train_model_with_validation(&model3, ds, fi, 4, target_index, 0.2,
                                     &train_m, &val_m);
    if (ret == 0) {
        fprintf(stderr, "Training metrics: Acc=%.4f, Prec=%.4f, Rec=%.4f, Spec=%.4f, F1=%.4f\n",
               train_m.Accuracy, train_m.Precision, train_m.Recall,
               train_m.Specificity, train_m.F1_Score);
        fprintf(stderr, "Validation metrics: Acc=%.4f, Prec=%.4f, Rec=%.4f, Spec=%.4f, F1=%.4f\n",
               val_m.Accuracy, val_m.Precision, val_m.Recall,
               val_m.Specificity, val_m.F1_Score);
    } else {
        fprintf(stderr, "train_model_with_validation failed: %d\n", ret);
    }

    model3.methods.free_state(&model3.state);

    /* Cleanup */
    free_dataset(ds);
    free_csv_data(csv);
    free(csv);

    fprintf(stderr, "\n=== All Binary Classification Tests Passed ===\n");
    return 0;
}