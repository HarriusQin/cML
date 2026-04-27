/**
 * @file test_dt.c
 * @brief Test Decision Tree model
 */

#define CSV_IMPLEMENTATION
#include "csv.h"

#define C_MACHINE_LEARNING_IMPLEMENTATION
#include "machine_learning.h"

#define DECISION_TREE_IMPLEMENTATION
#include "decision_tree.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    const char* csv_path = "data/iris.csv";
    if (argc > 1) csv_path = argv[1];

    printf("=== Decision Tree Test ===\n");
    csv_t* csv = csv_load(csv_path);
    if (!csv) { fprintf(stderr, "Failed to load CSV\n"); return 1; }

    const char* labels[] = {"species"};
    dataset* ds = csv_to_dataset(csv, labels, 1);
    if (!ds) { fprintf(stderr, "Failed to convert CSV\n"); free_csv_data(csv); free(csv); return 1; }

    const size_t feat_idx[] = {0, 1, 2, 3};
    size_t n_feat = 4;

    Dataset_Split_t split;
    train_test_split(ds, 0.2, 42, &split);

    ML_ScalingParams_t* scaler = ml_fit_scaling(ds, feat_idx, n_feat, split.train_indices, split.train_size, SCALING_STANDARD);
    dataset* scaled_train = ml_transform_features(scaler, ds, feat_idx, n_feat, split.train_indices, split.train_size);
    dataset* scaled_test = ml_transform_features(scaler, ds, feat_idx, n_feat, split.test_indices, split.test_size);

    size_t* train_idx = (size_t*)malloc(sizeof(size_t) * split.train_size);
    for (size_t i = 0; i < split.train_size; i++) train_idx[i] = i;
    size_t* test_idx = (size_t*)malloc(sizeof(size_t) * split.test_size);
    for (size_t i = 0; i < split.test_size; i++) test_idx[i] = i;

    ML_Model_t dt = create_dt_model();
    dt.methods.fit(&dt.config, &dt.state, scaled_train, feat_idx, n_feat, 0, train_idx, split.train_size);

    int* preds = (int*)malloc(sizeof(int) * split.test_size);
    dt.methods.predict(&dt.state, scaled_test, feat_idx, n_feat, test_idx, split.test_size, preds);

    size_t correct = 0;
    for (size_t i = 0; i < split.test_size; i++) {
        int pred_label = preds[i];
        int true_label = (int)ds->labels[0].labels[split.test_indices[i]];
        if (pred_label == true_label) correct++;
    }
    printf("Test Accuracy: %.2f%%\n", 100.0 * correct / split.test_size);

    if (100.0 * correct / split.test_size < 50.0) { fprintf(stderr, "Accuracy too low!\n"); return 1; }

    dt.methods.free_state(&dt.state);

    CV_Result_t cv = kfold_cross_validate(&(ML_Model_t){.type = ML_CLASSIFICATION, .config = {NULL, 0}, .state = {NULL, 0},
        .methods = { .fit = dt_fit, .predict = dt_predict, .predict_proba = NULL, .get_coefficients = NULL,
                     .serialize = NULL, .deserialize = NULL, .free_state = dt_free }},
        ds, feat_idx, n_feat, 0, 5, 42);
    printf("CV Accuracy: %.2f%% (k=%zu)\n", 100.0 * cv.avg_metrics.cls.Accuracy, cv.k);
    if (cv.fold_scores) free(cv.fold_scores);

    free(preds);
    free(train_idx);
    free(test_idx);
    ml_free_scaling_params(scaler);
    free_dataset(scaled_train);
    free_dataset(scaled_test);
    free(split.train_indices);
    free(split.test_indices);
    free_dataset(ds);
    free_csv_data(csv);
    free(csv);
    printf("=== Decision Tree Test Passed ===\n");
    return 0;
}