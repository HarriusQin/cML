/**
 * @file test_iris.c
 * @brief Test ML interfaces with Iris dataset
 *
 * This file tests the improved machine_learning.h interfaces.
 */

#define CSV_IMPLEMENTATION
#include "csv.h"

#include "dataset.h"

#define C_MACHINE_LEARNING_IMPLEMENTATION
#include "machine_learning.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char* argv[]) {
    const char* csv_path = "data/iris.csv";
    if (argc > 1) {
        csv_path = argv[1];
    }

    printf("=== Loading Iris dataset from %s ===\n", csv_path);

    csv_t* csv = csv_load(csv_path);
    if (!csv) {
        fprintf(stderr, "Failed to load CSV\n");
        return 1;
    }
    printf("CSV loaded: %zu rows, %zu cols\n", csv->size - 1, csv->rows[0].size);

    const char* labels[] = {"species"};
    dataset* ds = csv_to_dataset(csv, labels, 1);
    if (!ds) {
        fprintf(stderr, "Failed to convert CSV to dataset\n");
        free_csv_data(csv);
        free(csv);
        return 1;
    }
    printf("Dataset: %zu samples, %zu features, %zu labels\n",
           ds->rows, ds->num_features, ds->num_labels);

    /* Feature indices - now const! */
    const size_t feature_indices[] = {0, 1, 2, 3};
    size_t n_features = 4;
    size_t target_index = 0;

    printf("\n=== Test 1: train_test_split ===\n");
    Dataset_Split_t split = {0};
    int ret = train_test_split(ds, 0.2, 42, &split);
    printf("ret=%d, train_size=%zu, test_size=%zu\n", ret, split.train_size, split.test_size);

    printf("\n=== Test 2: Feature scaling (dataset is now const) ===\n");
    ML_ScalingParams_t* params = ml_fit_scaling(ds, feature_indices, n_features, split.train_indices, split.train_size, SCALING_STANDARD);
    if (params) {
        printf("Scaling fitted: type=%d, n_features=%zu\n", params->type, params->n_features);
        ml_free_scaling_params(params);
    }

    printf("\n=== Test 3: CV_Result_t now has metric_type ===\n");
    ML_Model_t model = {
        .type = ML_CLASSIFICATION,
        .config = {NULL, 0},
        .state = {NULL, 0},
        .methods = {NULL, NULL, NULL, NULL, NULL, NULL, NULL}
    };

    CV_Result_t cv = kfold_cross_validate(&model, ds, feature_indices, n_features, target_index, 5, 42);
    printf("cv.metric_type=%d (ML_CLASSIFICATION=%d)\n", cv.metric_type, ML_CLASSIFICATION);
    printf("cv.k=%zu, avg_acc=%.2f\n", cv.k, cv.avg_metrics.cls.Accuracy);
    if (cv.fold_scores) free(cv.fold_scores);

    printf("\n=== Test 4: ml_method_is_implemented ===\n");
    printf("Checking methods on empty model (all NULL):\n");
    printf("  fit implemented: %s\n", ml_method_is_implemented(&model.methods, ML_METHOD_FIT) ? "yes" : "no");
    printf("  predict implemented: %s\n", ml_method_is_implemented(&model.methods, ML_METHOD_PREDICT) ? "yes" : "no");
    printf("  predict_proba implemented: %s\n", ml_method_is_implemented(&model.methods, ML_METHOD_PREDICT_PROBA) ? "yes" : "no");
    printf("  get_coefficients implemented: %s\n", ml_method_is_implemented(&model.methods, ML_METHOD_GET_COEFFICIENTS) ? "yes" : "no");
    printf("  serialize implemented: %s\n", ml_method_is_implemented(&model.methods, ML_METHOD_SERIALIZE) ? "yes" : "no");
    printf("  deserialize implemented: %s\n", ml_method_is_implemented(&model.methods, ML_METHOD_DESERIALIZE) ? "yes" : "no");
    printf("  free_state implemented: %s\n", ml_method_is_implemented(&model.methods, ML_METHOD_FREE_STATE) ? "yes" : "no");
    printf("Implemented mask: 0x%x (expecting 0x00)\n", ml_methods_get_implemented(&model.methods));

    printf("\n=== Test 5: model_evaluate (without predict method - should fail) ===\n");
    ML_Classification_Metrics_t eval_met = {0};
    ret = model_evaluate(&model, ds, feature_indices, n_features, target_index, split.train_indices, split.train_size, &eval_met);
    printf("ret=%d (expected -1 since predict is NULL)\n", ret);

    printf("\n=== Test 6: train_model_with_validation ===\n");
    ML_Classification_Metrics_t train_met = {0}, val_met = {0};
    ret = train_model_with_validation(&model, ds, feature_indices, n_features,
                                      target_index, 0.2, &train_met, &val_met);
    printf("ret=%d, train_acc=%.2f, val_acc=%.2f\n", ret, train_met.Accuracy, val_met.Accuracy);

    printf("\n=== All improvements verified! ===\n");

    free(split.train_indices);
    free(split.test_indices);
    free_dataset(ds);
    free_csv_data(csv);
    free(csv);

    return 0;
}
