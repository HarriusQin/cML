/**
 * @file test_mlp.c
 * @brief Test MLP on MNIST dataset
 */

#define MLP_IMPLEMENTATION
#include "dl/mlp.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char* argv[]) {
    const char* data_dir = "data";
    if (argc > 1) data_dir = argv[1];

    printf("=== MLP MNIST Test ===\n");

    // Load MNIST data
    printf("Loading MNIST from %s...\n", data_dir);
    MNISTData* mnist = mnist_load(data_dir);
    if (!mnist) {
        fprintf(stderr, "Failed to load MNIST\n");
        return 1;
    }
    printf("Loaded: %zu train images, %zu test images\n",
           mnist->train_images->shape[0], mnist->test_images->shape[0]);

    // Create MLP: 784 -> 256 -> 128 -> 10 (3-layer architecture)
    size_t input_dim = 784;
    size_t hidden_dim = 256;
    size_t hidden_dim2 = 128;
    size_t output_dim = 10;
    size_t num_layers = 3;

    printf("Creating MLP: %zu -> %zu -> %zu -> %zu\n",
           input_dim, hidden_dim, hidden_dim2, output_dim);

    MLP* mlp = mlp_create(input_dim, hidden_dim, output_dim, num_layers);

    // Create SGD optimizer with momentum for deeper network
    SGDOptimizer* opt = sgd_create(mlp, 0.01f, 0.9f, 0.0001f);

    size_t batch_size = 128;
    size_t epochs = 10;
    size_t batches_per_epoch = mnist->train_images->shape[0] / batch_size;

    printf("\nTraining: epochs=%zu, batch_size=%zu, batches_per_epoch=%zu\n",
           epochs, batch_size, batches_per_epoch);

    // Quick sanity check: train on one batch and see if loss decreases
    printf("\n=== Sanity Check: Overfit Single Batch ===\n");
    Tensor* X_batch = get_batch(mnist->train_images, 0, batch_size);
    Tensor* y_batch = get_labels_batch(mnist->train_labels, 0, batch_size);

    // Print first few labels
    float* y_data = (float*)y_batch->data;
    printf("First 10 labels: ");
    for (size_t i = 0; i < 10; i++) printf("%.0f ", y_data[i]);
    printf("\n");

    // Check weight before training
    FCLayer* l0 = mlp->layers[0];
    FCLayer* l1 = mlp->layers[1];

    for (size_t iter = 0; iter < 50; iter++) {
        float loss = mlp_train_step(mlp, opt, X_batch, y_batch);

        if (iter % 10 == 0) {
            float acc = mlp_accuracy(mlp, X_batch, y_batch);

            // Debug: check prediction distribution
            Tensor* pred = mlp_predict(mlp, X_batch);
            float* pred_data = (float*)pred->data;

            // Check mean output for each class
            float class_sums[10] = {0};
            for (size_t b = 0; b < batch_size; b++) {
                for (size_t c = 0; c < 10; c++) {
                    class_sums[c] += pred_data[b * 10 + c];
                }
            }
            for (size_t c = 0; c < 10; c++) class_sums[c] /= batch_size;

            printf("Iter %zu, Loss: %.4f, Acc: %.2f%%, class_means: [", iter, loss, 100.0f * acc);
            for (size_t c = 0; c < 10; c++) printf("%.3f ", class_sums[c]);
            printf("]\n");

            tensor_free(pred);
        }
    }

    tensor_free(X_batch);
    tensor_free(y_batch);
    printf("=== Sanity Check Complete ===\n\n");

    clock_t start = clock();

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0.0f;

        // Shuffle training data (simple shuffle)
        // For simplicity, we iterate in order; real implementation should shuffle

        for (size_t batch = 0; batch < batches_per_epoch; batch++) {
            size_t start_idx = batch * batch_size;

            // Get mini-batch
            Tensor* X_batch = get_batch(mnist->train_images, start_idx, batch_size);
            Tensor* y_batch = get_labels_batch(mnist->train_labels, start_idx, batch_size);

            // Training step
            float loss = mlp_train_step(mlp, opt, X_batch, y_batch);
            epoch_loss += loss;

            if (batch % 100 == 0) {
                // Debug: check first layer weight gradient stats
                FCLayer* l0 = mlp->layers[0];
                float* gw = (float*)l0->grad_w->data;
                float grad_max = gw[0], grad_min = gw[0], grad_sum = 0;
                for (size_t i = 0; i < l0->grad_w->size; i++) {
                    if (gw[i] > grad_max) grad_max = gw[i];
                    if (gw[i] < grad_min) grad_min = gw[i];
                    grad_sum += gw[i];
                }

                // Debug: check prediction distribution
                Tensor* pred = mlp_predict(mlp, X_batch);
                float* pred_data = (float*)pred->data;
                float pred_sum = 0, pred_max_val = pred_data[0];
                for (size_t i = 0; i < pred->size; i++) {
                    pred_sum += pred_data[i];
                    if (pred_data[i] > pred_max_val) pred_max_val = pred_data[i];
                }
                tensor_free(pred);

                printf("Epoch %zu, Batch %zu/%zu, Loss: %.4f, |grad_w[0]|: max=%.6f, min=%.6f, mean=%.8f, pred_sum=%.4f, pred_max=%.4f\n",
                       epoch + 1, batch, batches_per_epoch, loss, grad_max, grad_min, grad_sum / l0->grad_w->size, pred_sum, pred_max_val);
            }

            tensor_free(X_batch);
            tensor_free(y_batch);
        }

        epoch_loss /= batches_per_epoch;

        // Evaluate on test set
        size_t test_batches = mnist->test_images->shape[0] / batch_size;
        float test_acc = 0.0f;
        for (size_t b = 0; b < test_batches; b++) {
            size_t start_idx = b * batch_size;
            Tensor* X_batch = get_batch(mnist->test_images, start_idx, batch_size);
            Tensor* y_batch = get_labels_batch(mnist->test_labels, start_idx, batch_size);

            test_acc += mlp_accuracy(mlp, X_batch, y_batch);

            tensor_free(X_batch);
            tensor_free(y_batch);
        }
        test_acc /= test_batches;

        printf("Epoch %zu complete - Loss: %.4f, Test Accuracy: %.2f%%\n",
               epoch + 1, epoch_loss, 100.0f * test_acc);
    }

    clock_t end = clock();
    printf("\nTraining time: %.2f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    // Final evaluation
    printf("\n=== Final Evaluation ===\n");

    // Training accuracy
    size_t train_batches = mnist->train_images->shape[0] / batch_size;
    float train_acc = 0.0f;
    for (size_t b = 0; b < train_batches; b++) {
        size_t start_idx = b * batch_size;
        Tensor* X_batch = get_batch(mnist->train_images, start_idx, batch_size);
        Tensor* y_batch = get_labels_batch(mnist->train_labels, start_idx, batch_size);

        train_acc += mlp_accuracy(mlp, X_batch, y_batch);

        tensor_free(X_batch);
        tensor_free(y_batch);
    }
    train_acc /= train_batches;
    printf("Training Accuracy: %.2f%%\n", 100.0f * train_acc);

    // Test accuracy
    size_t test_batches = mnist->test_images->shape[0] / batch_size;
    float test_acc = 0.0f;
    for (size_t b = 0; b < test_batches; b++) {
        size_t start_idx = b * batch_size;
        Tensor* X_batch = get_batch(mnist->test_images, start_idx, batch_size);
        Tensor* y_batch = get_labels_batch(mnist->test_labels, start_idx, batch_size);

        test_acc += mlp_accuracy(mlp, X_batch, y_batch);

        tensor_free(X_batch);
        tensor_free(y_batch);
    }
    test_acc /= test_batches;
    printf("Test Accuracy: %.2f%%\n", 100.0f * test_acc);

    // Cleanup
    sgd_free(opt);
    mlp_free(mlp);
    mnist_free(mnist);

    printf("\n=== MLP Test Complete ===\n");
    return 0;
}
