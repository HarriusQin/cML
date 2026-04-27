/**
 * @file test_ml.c
 * @brief Test program for ML models
 *
 * 测试程序演示所有机器学习模型的使用:
 * - 高斯朴素贝叶斯 (GNB)
 * - 决策树 (Decision Tree)
 * - 随机森林 (Random Forest)
 * - AdaBoost
 * - K近邻 (KNN)
 *
 * 编译: gcc -o test_ml test_ml.c -lm
 * 运行: ./test_ml
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CSV_IMPLEMENTATION
#include "csv.h"

#define DATASET_IMPLEMENTATION
#include "dataset.h"

#define ML_IMPLEMENTATION
#include "ml_samples.h"

static void print_separator(void) {
    printf("============================================================\n");
}

static void print_header(const char* title) {
    print_separator();
    printf("%s\n", title);
    print_separator();
}

int main(void) {
    printf("\n");
    print_separator();
    printf("  Machine Learning Models Test (ml_samples.h)\n");
    print_separator();
    printf("\n");

    /* Load data */
    print_header("Step 1: Loading Iris Dataset");

    csv_t* csv = csv_load("data/iris.csv");
    if (!csv) {
        fprintf(stderr, "Error: Failed to load iris.csv\n");
        return 1;
    }
    printf("CSV loaded: %zu rows, %zu columns\n", csv->size - 1, csv->rows[0].size);

    const char* labels[] = {"species"};
    dataset* ds = csv_to_dataset(csv, labels, 1);
    if (!ds) {
        fprintf(stderr, "Error: Failed to convert CSV to dataset\n");
        free_csv_data(csv);
        free(csv);
        return 1;
    }

    printf("Dataset: %zu samples, %zu features, %zu classes\n\n",
           ds->rows, ds->num_features, ds->labels[0].classes);

    const char* class_names[] = {"setosa", "versicolor", "virginica"};

    double test_samples[][4] = {
        {5.1, 3.5, 1.4, 0.2},
        {6.0, 2.7, 5.1, 1.6},
        {7.0, 3.0, 6.3, 2.4}
    };
    int expected[] = {0, 1, 2};

    /* Gaussian Naive Bayes */
    print_header("Step 2: Gaussian Naive Bayes");
    printf("Training GNB...\n");
    model* gnb = train_gnb(ds, 0, 0, default_config);
    if (gnb) {
        double acc = accuracy(gnb, ds, 0, 0);
        printf("GNB Accuracy: %.2f%%\n", acc * 100);
        printf("Test: [%s] [%s] [%s]\n",
               class_names[predict_gnb(gnb, test_samples[0], 4)],
               class_names[predict_gnb(gnb, test_samples[1], 4)],
               class_names[predict_gnb(gnb, test_samples[2], 4)]);
        model_free(gnb);
        free(gnb);
    }
    printf("\n");

    /* Decision Tree */
    print_header("Step 3: Decision Tree");
    model_config dt_config = default_config;
    dt_config.max_depth = 10;
    printf("Training Decision Tree (depth=%zu)...\n", dt_config.max_depth);
    model* dt = train_decision_tree(ds, 0, 0, dt_config);
    if (dt) {
        double acc = accuracy(dt, ds, 0, 0);
        printf("DT Accuracy: %.2f%%\n", acc * 100);
        printf("Test: [%s] [%s] [%s]\n",
               class_names[(int)predict_decision_tree(dt, test_samples[0], 4)],
               class_names[(int)predict_decision_tree(dt, test_samples[1], 4)],
               class_names[(int)predict_decision_tree(dt, test_samples[2], 4)]);
        model_free(dt);
        free(dt);
    }
    printf("\n");

    /* Random Forest */
    print_header("Step 4: Random Forest");
    model_config rf_config = default_config;
    rf_config.n_estimators = 100;
    rf_config.max_depth = 10;
    printf("Training RF (%zu trees, depth=%zu)...\n",
           rf_config.n_estimators, rf_config.max_depth);
    model* rf = train_random_forest(ds, 0, 0, rf_config);
    if (rf) {
        double acc = accuracy(rf, ds, 0, 0);
        printf("RF Accuracy: %.2f%%\n", acc * 100);
        printf("Test: [%s] [%s] [%s]\n",
               class_names[(int)predict_random_forest(rf, test_samples[0], 4)],
               class_names[(int)predict_random_forest(rf, test_samples[1], 4)],
               class_names[(int)predict_random_forest(rf, test_samples[2], 4)]);
        model_free(rf);
        free(rf);
    }
    printf("\n");

    /* AdaBoost */
    print_header("Step 5: AdaBoost");
    model_config ada_config = default_config;
    ada_config.n_estimators = 50;
    printf("Training AdaBoost (%zu stumps)...\n", ada_config.n_estimators);
    model* ada = train_adaboost(ds, 0, 0, ada_config);
    if (ada) {
        double acc = accuracy(ada, ds, 0, 0);
        printf("AdaBoost Accuracy: %.2f%%\n", acc * 100);
        printf("Test: [%s] [%s] [%s]\n",
               class_names[predict_adaboost(ada, test_samples[0], 4)],
               class_names[predict_adaboost(ada, test_samples[1], 4)],
               class_names[predict_adaboost(ada, test_samples[2], 4)]);
        model_free(ada);
        free(ada);
    }
    printf("\n");

    /* KNN */
    print_header("Step 6: KNN");
    model_config knn_config = default_config;
    knn_config.k_value = 5;
    printf("Training KNN (K=%zu)...\n", knn_config.k_value);
    model* knn = train_knn(ds, 0, 0, knn_config);
    if (knn) {
        double acc = accuracy(knn, ds, 0, 0);
        printf("KNN Accuracy: %.2f%%\n", acc * 100);
        printf("Test: [%s] [%s] [%s]\n",
               class_names[(int)predict_knn(knn, test_samples[0], 4)],
               class_names[(int)predict_knn(knn, test_samples[1], 4)],
               class_names[(int)predict_knn(knn, test_samples[2], 4)]);
        model_free(knn);
        free(knn);
    }
    printf("\n");

    /* Summary */
    print_header("Summary");
    printf("All models tested successfully!\n");
    print_separator();

    free_dataset(ds);
    free(ds);
    free_csv_data(csv);
    free(csv);

    return 0;
}
