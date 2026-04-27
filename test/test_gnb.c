/**
 * @file test_gnb.c
 * @brief Test Gaussian Naive Bayes model
 */

#define CSV_IMPLEMENTATION
#include "csv.h"

#define C_MACHINE_LEARNING_IMPLEMENTATION
#include "machine_learning.h"

#define GNB_IMPLEMENTATION
#include "gnb.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    const char* csv_path = "data/iris.csv";
    if (argc > 1) csv_path = argv[1];

    printf("=== GNB Test ===\n");
    csv_t* csv = csv_load(csv_path);
    if (!csv) { fprintf(stderr, "Failed to load CSV\n"); return 1; }

    const char* labels[] = {"species"};
    dataset* ds = csv_to_dataset(csv, labels, 1);
    if (!ds) { fprintf(stderr, "Failed to convert CSV\n"); free_csv_data(csv); free(csv); return 1; }

    const size_t feat_idx[] = {0, 1, 2, 3};
    size_t n_feat = 4;

    ML_Model_t model = create_gnb_model();
    size_t* indices = (size_t*)malloc(sizeof(size_t) * ds->rows);
    for (size_t i = 0; i < ds->rows; i++) indices[i] = i;

    int ret = model.methods.fit(&model.config, &model.state, ds, feat_idx, n_feat, 0, indices, ds->rows);
    if (ret != 0) { fprintf(stderr, "Fit failed\n"); return 1; }

    int preds[150];
    ret = model.methods.predict(&model.state, ds, feat_idx, n_feat, indices, ds->rows, preds);
    if (ret != 0) { fprintf(stderr, "Predict failed\n"); return 1; }

    size_t correct = 0;
    for (size_t i = 0; i < ds->rows; i++)
        if (preds[i] == ds->labels[0].labels[i]) correct++;
    printf("Accuracy: %.2f%%\n", 100.0 * correct / ds->rows);

    if (100.0 * correct / ds->rows < 90.0) { fprintf(stderr, "Accuracy too low!\n"); return 1; }

    model.methods.free_state(&model.state);
    free(indices);
    free_dataset(ds);
    free_csv_data(csv);
    free(csv);
    printf("=== GNB Test Passed ===\n");
    return 0;
}