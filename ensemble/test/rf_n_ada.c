#define CSV_IMPLEMENTATION
#define C_MACHINE_LEARNING_IMPLEMENTATION

#define MAX_CLASSES 256

#define ADABOOST_MAX_ESTIMATORS 50
#define RANDOMFOREST_MAX_ESTIMATORS 200

#include "dataset.h"
#include "machine_learning.h"
#include "utilities.h"
#define ADABOOST_IMPLEMENTATION
#include "adaboost.h"

#define RANDOMFOREST_IMPLEMENTATION
#include "randomforest.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

int main(int argc, char const* argv[]) {
    csv_t* csv_data = csv_load("data/iris.csv");
    const char* labels[] = {"species"};
    dataset* ds = csv_to_dataset(csv_data, labels, 1);

    /* Feature indices: use all 4 features */
    size_t feature_indices[4];
    for (size_t i = 0; i < ds->num_features; i++)
        feature_indices[i] = i;
    size_t n_features = ds->num_features;

    /* Train/test split (80/20) */
    Dataset_Split_t split = {0};
    train_test_split(ds, 0.2, 42, &split);

    size_t* train_indices = split.train_indices;
    size_t n_train = split.train_size;
    size_t* test_indices = split.test_indices;
    size_t n_test = split.test_size;

    /* ==================== RandomForest (Multiclass) ==================== */
    ML_Weights_t rf_state = {0};
    int ret = RandomForest_fit(NULL, &rf_state, ds, feature_indices, n_features,
                          0, train_indices, n_train);
    if (ret != 0) {
        printf("RandomForest training failed\n");
        free(split.train_indices);
        free(split.test_indices);
        free_dataset(ds);
        free_csv_data(csv_data);
        return 1;
    }

    int* rf_preds = (int*)malloc(sizeof(int) * n_test);
    RandomForest_predict(&rf_state, ds, feature_indices, n_features,
                        test_indices, n_test, rf_preds);

    size_t rf_correct = 0;
    for (size_t i = 0; i < n_test; i++) {
        int true_label = ds->labels[0].labels[test_indices[i]];
        if (rf_preds[i] == true_label)
            rf_correct++;
    }
    double rf_accuracy = (double)rf_correct / n_test;

    printf("=== RandomForest Multiclass Classification ===\n");
    printf("Train samples: %zu, Test samples: %zu, Trees used: %zu\n",
           n_train, n_test, ((RandomForest_State*)rf_state.weights)->n_trees);
    printf("Accuracy: %.2f%%\n", rf_accuracy * 100);

    RandomForest_free(&rf_state);
    free(rf_preds);

    /* ==================== AdaBoost (Binary: setosa vs others) ==================== */
    dataset_expand_binary_labels(ds, 0, BINARY_SIGNED);
    size_t target_index = 0; /* species_setosa binary column */

    ML_Model_Config_t config = {0};
    ML_Weights_t ada_state = {0};

    ret = AdaBoost_fit(&config, &ada_state, ds, feature_indices, n_features,
                       target_index, train_indices, n_train);
    if (ret != 0) {
        printf("AdaBoost training failed\n");
        free(split.train_indices);
        free(split.test_indices);
        free_dataset(ds);
        free_csv_data(csv_data);
        return 1;
    }

    int* ada_preds = (int*)malloc(sizeof(int) * n_test);
    Adaboost_Predict(&ada_state, ds, feature_indices, n_features, test_indices,
                     n_test, ada_preds);

    const binary_label_column* bin_labels = &ds->binary_labels[target_index];
    size_t ada_correct = 0;
    for (size_t i = 0; i < n_test; i++) {
        if (ada_preds[i] == bin_labels->labels[test_indices[i]])
            ada_correct++;
    }
    double ada_accuracy = (double)ada_correct / n_test;

    size_t tp = 0, fp = 0, fn = 0;
    for (size_t i = 0; i < n_test; i++) {
        int pred = ada_preds[i];
        int actual = bin_labels->labels[test_indices[i]];
        if (pred == 1 && actual == 1) tp++;
        else if (pred == 1 && actual == -1) fp++;
        else if (pred == -1 && actual == 1) fn++;
    }
    double precision = (tp + fp > 0) ? (double)tp / (tp + fp) : 0.0;
    double recall = (tp + fn > 0) ? (double)tp / (tp + fn) : 0.0;
    double f1 = (precision + recall > 0) ? 2.0 * precision * recall / (precision + recall) : 0.0;

    printf("\n=== AdaBoost Binary Classification (setosa vs others) ===\n");
    printf("Train samples: %zu, Test samples: %zu\n", n_train, n_test);
    printf("Accuracy:  %.2f%%\n", ada_accuracy * 100);
    printf("Precision: %.2f%%\n", precision * 100);
    printf("Recall:    %.2f%%\n", recall * 100);
    printf("F1 Score:  %.2f%%\n", f1 * 100);

    /* Cleanup */
    AdaBoost_free(&ada_state);
    free(ada_preds);
    free(split.train_indices);
    free(split.test_indices);
    free_dataset(ds);
    free_csv_data(csv_data);

    return 0;
}
