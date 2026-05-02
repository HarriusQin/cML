/**
 * @file test_mlp_iris.c
 * @brief Test MLP on Iris dataset using dataset adapter
 *
 * This demonstrates:
 * - Converting dataset (from CSV) to Tensor format
 * - Feature normalization
 * - Train/test split
 * - MLP training and evaluation
 */

#define MLP_IMPLEMENTATION
#include "dl/mlp.h"

#define CSV_IMPLEMENTATION
#include "csv.h"

#define DATASET_IMPLEMENTATION
#include "dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/* ============================================================================
 * Dataset to Tensor Adapter
 * ============================================================================ */

/**
 * Convert dataset features to a single Tensor [n_samples, n_features]
 * Features are normalized using min-max scaling to [0, 1]
 */
static Tensor* dataset_features_to_tensor(const dataset* ds, size_t feature_idx) {
    size_t n_samples = ds->rows;
    size_t n_features = ds->num_features;

    size_t shape[] = {n_samples, n_features};
    Tensor* t = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 2);
    float* data = (float*)t->data;

    // Find min/max for each feature
    double* min_vals = malloc(sizeof(double) * n_features);
    double* max_vals = malloc(sizeof(double) * n_features);

    for (size_t f = 0; f < n_features; f++) {
        min_vals[f] = ds->features[f].data[0];
        max_vals[f] = ds->features[f].data[0];
        for (size_t i = 1; i < n_samples; i++) {
            double val = ds->features[f].data[i];
            if (val < min_vals[f]) min_vals[f] = val;
            if (val > max_vals[f]) max_vals[f] = val;
        }
    }

    // Normalize to [0, 1]
    for (size_t i = 0; i < n_samples; i++) {
        for (size_t f = 0; f < n_features; f++) {
            double range = max_vals[f] - min_vals[f];
            double val = ds->features[f].data[i];
            if (range > 0) {
                data[i * n_features + f] = (float)((val - min_vals[f]) / range);
            } else {
                data[i * n_features + f] = 0.0f;
            }
        }
    }

    free(min_vals);
    free(max_vals);
    return t;
}

/**
 * Convert dataset labels (single label column) to a Tensor [n_samples]
 * Labels are stored as class indices (0, 1, 2, ...)
 */
static Tensor* dataset_labels_to_tensor(const dataset* ds, size_t label_idx) {
    size_t n_samples = ds->rows;

    size_t shape[] = {n_samples};
    Tensor* t = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 1);
    float* data = (float*)t->data;

    for (size_t i = 0; i < n_samples; i++) {
        data[i] = (float)ds->labels[label_idx].labels[i];
    }

    return t;
}

/**
 * Get the number of classes in a label column
 */
static size_t dataset_num_classes(const dataset* ds, size_t label_idx) {
    return ds->labels[label_idx].classes;
}

/* ============================================================================
 * Train/Test Split
 * ============================================================================ */

typedef struct {
    size_t* train_indices;
    size_t* test_indices;
    size_t train_size;
    size_t test_size;
} TrainTestSplit;

static void train_test_split_indices(size_t n_samples, float test_ratio, unsigned int seed,
                                     size_t** train_out, size_t** test_out,
                                     size_t* train_size_out, size_t* test_size_out) {
    size_t test_size = (size_t)(n_samples * test_ratio);
    size_t train_size = n_samples - test_size;

    size_t* indices = malloc(sizeof(size_t) * n_samples);
    for (size_t i = 0; i < n_samples; i++) indices[i] = i;

    // Shuffle using Fisher-Yates
    srand(seed);
    for (size_t i = n_samples - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        size_t tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }

    *train_out = malloc(sizeof(size_t) * train_size);
    *test_out = malloc(sizeof(size_t) * test_size);
    *train_size_out = train_size;
    *test_size_out = test_size;

    memcpy(*train_out, indices, sizeof(size_t) * train_size);
    memcpy(*test_out, indices + train_size, sizeof(size_t) * test_size);

    free(indices);
}

static void free_split(TrainTestSplit* split) {
    if (split->train_indices) free(split->train_indices);
    if (split->test_indices) free(split->test_indices);
}

/**
 * Get a batch of features from a tensor
 */
static Tensor* get_feature_batch(const Tensor* features, const size_t* indices, size_t batch_size) {
    size_t n_features = features->shape[1];
    size_t shape[] = {batch_size, n_features};
    Tensor* batch = tensor_create(TENSOR_DTYPE_F32, features->layout, shape, 2);
    float* dst = (float*)batch->data;
    float* src = (float*)features->data;

    for (size_t i = 0; i < batch_size; i++) {
        memcpy(dst + i * n_features, src + indices[i] * n_features, sizeof(float) * n_features);
    }
    return batch;
}

/**
 * Get a batch of labels from a tensor
 */
static Tensor* get_label_batch(const Tensor* labels, const size_t* indices, size_t batch_size) {
    size_t shape[] = {batch_size};
    Tensor* batch = tensor_create(TENSOR_DTYPE_F32, labels->layout, shape, 1);
    float* dst = (float*)batch->data;
    float* src = (float*)labels->data;

    for (size_t i = 0; i < batch_size; i++) {
        dst[i] = src[indices[i]];
    }
    return batch;
}

/**
 * Evaluate MLP accuracy on a dataset
 */
static float evaluate_accuracy(MLP* mlp, const Tensor* features, const Tensor* labels,
                              const size_t* indices, size_t n_samples, size_t n_classes) {
    size_t batch_size = 32;
    size_t correct = 0;

    for (size_t i = 0; i < n_samples; i += batch_size) {
        size_t actual_batch = (i + batch_size > n_samples) ? (n_samples - i) : batch_size;
        Tensor* X_batch = get_feature_batch(features, indices + i, actual_batch);

        Tensor* pred = mlp_predict(mlp, X_batch);
        float* pred_data = (float*)pred->data;

        for (size_t b = 0; b < actual_batch; b++) {
            // Find predicted class (argmax) - only check n_classes
            size_t pred_class = 0;
            float pred_max = pred_data[b * n_classes];  // first class
            for (size_t c = 1; c < n_classes; c++) {
                if (pred_data[b * n_classes + c] > pred_max) {
                    pred_max = pred_data[b * n_classes + c];
                    pred_class = c;
                }
            }

            // Get true class
            size_t true_class = (size_t)((float*)labels->data)[indices[i + b]];

            if (pred_class == true_class) correct++;
        }

        tensor_free(X_batch);
        tensor_free(pred);
    }

    return (float)correct / (float)n_samples;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char* argv[]) {
    const char* csv_path = "data/iris.csv";
    if (argc > 1) csv_path = argv[1];

    printf("=== MLP Iris Classification ===\n\n");

    // Load CSV
    printf("Loading CSV from %s...\n", csv_path);
    csv_t* csv = csv_load(csv_path);
    if (!csv) {
        fprintf(stderr, "Failed to load CSV\n");
        return 1;
    }
    printf("CSV: %zu rows, %zu cols\n", csv->size - 1, csv->rows[0].size);

    // Convert to dataset
    const char* labels[] = {"species"};
    dataset* ds = csv_to_dataset(csv, labels, 1);
    if (!ds) {
        fprintf(stderr, "Failed to convert to dataset\n");
        free_csv_data(csv);
        free(csv);
        return 1;
    }
    printf("Dataset: %zu samples, %zu features, %zu classes\n",
           ds->rows, ds->num_features, dataset_num_classes(ds, 0));

    // Print class names
    printf("Classes: ");
    for (size_t c = 0; c < ds->labels[0].classes; c++) {
        printf("%s ", ds->labels[0].value_map[c]);
    }
    printf("\n\n");

    // Convert to tensors
    printf("Converting to tensors...\n");
    Tensor* X = dataset_features_to_tensor(ds, 0);
    Tensor* y = dataset_labels_to_tensor(ds, 0);
    tensor_print(X, "X");
    tensor_print(y, "y");

    // Train/test split
    printf("\nCreating train/test split (80/20)...\n");
    TrainTestSplit split = {0};
    train_test_split_indices(ds->rows, 0.2f, 42,
                             &split.train_indices, &split.test_indices,
                             &split.train_size, &split.test_size);
    printf("Train size: %zu, Test size: %zu\n", split.train_size, split.test_size);

    // Create MLP: input_dim -> hidden_dim -> output_dim
    size_t input_dim = ds->num_features;  // 4 features
    size_t hidden_dim = 16;
    size_t output_dim = dataset_num_classes(ds, 0);  // 3 classes
    size_t num_layers = 2;
    size_t n_classes = output_dim;

    printf("\nCreating MLP: %zu -> %zu -> %zu\n", input_dim, hidden_dim, output_dim);
    MLP* mlp = mlp_create(input_dim, hidden_dim, output_dim, num_layers);

    // Create SGD optimizer - lower learning rate for Iris
    SGDOptimizer* opt = sgd_create(mlp, 0.01f, 0.9f, 0.0001f);

    // Training
    size_t batch_size = 16;
    size_t epochs = 200;
    size_t batches_per_epoch = split.train_size / batch_size;

    printf("\nTraining: epochs=%zu, batch_size=%zu\n", epochs, batch_size);

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0.0f;

        for (size_t b = 0; b < batches_per_epoch; b++) {
            size_t start = b * batch_size;
            Tensor* X_batch = get_feature_batch(X, split.train_indices + start, batch_size);
            Tensor* y_batch = get_label_batch(y, split.train_indices + start, batch_size);

            float loss = mlp_train_step(mlp, opt, X_batch, y_batch);
            epoch_loss += loss;

            tensor_free(X_batch);
            tensor_free(y_batch);
        }

        epoch_loss /= batches_per_epoch;

        if (epoch % 20 == 0 || epoch == epochs - 1) {
            float train_acc = evaluate_accuracy(mlp, X, y, split.train_indices, split.train_size, n_classes);
            float test_acc = evaluate_accuracy(mlp, X, y, split.test_indices, split.test_size, n_classes);
            printf("Epoch %zu, Loss: %.4f, Train Acc: %.2f%%, Test Acc: %.2f%%\n",
                   epoch, epoch_loss, 100.0f * train_acc, 100.0f * test_acc);
        }
    }

    // Final evaluation
    printf("\n=== Final Evaluation ===\n");
    float train_acc = evaluate_accuracy(mlp, X, y, split.train_indices, split.train_size, n_classes);
    float test_acc = evaluate_accuracy(mlp, X, y, split.test_indices, split.test_size, n_classes);
    printf("Training Accuracy: %.2f%%\n", 100.0f * train_acc);
    printf("Test Accuracy: %.2f%%\n", 100.0f * test_acc);

    // Confusion matrix
    printf("\nConfusion Matrix (test set):\n");
    size_t** confusion = calloc(n_classes, sizeof(size_t*));
    for (size_t c = 0; c < n_classes; c++) confusion[c] = calloc(n_classes, sizeof(size_t));

    size_t batch_size_eval = 32;
    for (size_t i = 0; i < split.test_size; i += batch_size_eval) {
        size_t actual_batch = (i + batch_size_eval > split.test_size) ?
                              (split.test_size - i) : batch_size_eval;
        Tensor* X_batch = get_feature_batch(X, split.test_indices + i, actual_batch);
        Tensor* pred = mlp_predict(mlp, X_batch);
        float* pred_data = (float*)pred->data;

        for (size_t b = 0; b < actual_batch; b++) {
            size_t pred_class = 0;
            float pred_max = pred_data[b * n_classes];
            for (size_t c = 1; c < n_classes; c++) {
                if (pred_data[b * n_classes + c] > pred_max) {
                    pred_max = pred_data[b * n_classes + c];
                    pred_class = c;
                }
            }
            size_t true_class = (size_t)((float*)y->data)[split.test_indices[i + b]];
            if (pred_class < n_classes && true_class < n_classes)
                confusion[true_class][pred_class]++;
        }

        tensor_free(X_batch);
        tensor_free(pred);
    }

    printf("%-12s", "True\\Pred");
    for (size_t c = 0; c < n_classes; c++) {
        printf("%-12zu", c);
    }
    printf("\n");
    for (size_t t = 0; t < n_classes; t++) {
        printf("%-12zu", t);
        for (size_t p = 0; p < n_classes; p++) {
            printf("%-12zu", confusion[t][p]);
        }
        printf("  (%s)\n", ds->labels[0].value_map[t]);
    }

    // Cleanup
    for (size_t c = 0; c < n_classes; c++) free(confusion[c]);
    free(confusion);

    free_split(&split);
    sgd_free(opt);
    mlp_free(mlp);
    tensor_free(X);
    tensor_free(y);
    free_dataset(ds);
    free_csv_data(csv);
    free(csv);

    printf("\n=== Done ===\n");
    return 0;
}
