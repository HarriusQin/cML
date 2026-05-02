/**
 * @file test_opencl_mlp_mnist.c
 * @brief Test OpenCL MLP on MNIST dataset
 */

#include "../opencl_mlp.h"
#include "idx.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MNIST_NUM_TRAIN 60000
#define MNIST_NUM_TEST 10000
#define MNIST_IMAGE_SIZE 784  // 28x28

/* ============================================================================
 * MNIST Data Loading
 * ============================================================================ */

typedef struct {
    float* train_images;
    float* train_labels;
    float* test_images;
    float* test_labels;
    size_t num_train;
    size_t num_test;
} CLMNISTData;

static CLMNISTData* cl_mnist_load(const char* data_dir) {
    CLMNISTData* data = (CLMNISTData*)malloc(sizeof(CLMNISTData));

    char path[256];

    // Load training images
    snprintf(path, sizeof(path), "%s/train-images-idx3-ubyte", data_dir);
    cIDX* idx_images = idx_load(path);
    if (!idx_images) {
        printf("Failed to load %s\n", path);
        free(data);
        return NULL;
    }

    data->num_train = MNIST_NUM_TRAIN;
    data->train_images = (float*)malloc(MNIST_NUM_TRAIN * MNIST_IMAGE_SIZE * sizeof(float));
    uint8_t* img_data = (uint8_t*)idx_images->idx_data;
    for (size_t i = 0; i < MNIST_NUM_TRAIN * MNIST_IMAGE_SIZE; i++) {
        data->train_images[i] = img_data[i] / 255.0f;
    }
    free_idx(idx_images);

    // Load training labels
    snprintf(path, sizeof(path), "%s/train-labels-idx1-ubyte", data_dir);
    cIDX* idx_labels = idx_load(path);
    if (!idx_labels) {
        printf("Failed to load %s\n", path);
        free(data->train_images);
        free(data);
        return NULL;
    }

    data->train_labels = (float*)malloc(MNIST_NUM_TRAIN * sizeof(float));
    uint8_t* lbl_data = (uint8_t*)idx_labels->idx_data;
    for (size_t i = 0; i < MNIST_NUM_TRAIN; i++) {
        data->train_labels[i] = (float)lbl_data[i];
    }
    free_idx(idx_labels);

    // Load test images
    snprintf(path, sizeof(path), "%s/t10k-images-idx3-ubyte", data_dir);
    idx_images = idx_load(path);
    if (!idx_images) {
        printf("Failed to load %s\n", path);
        free(data->train_images);
        free(data->train_labels);
        free(data);
        return NULL;
    }

    data->num_test = MNIST_NUM_TEST;
    data->test_images = (float*)malloc(MNIST_NUM_TEST * MNIST_IMAGE_SIZE * sizeof(float));
    img_data = (uint8_t*)idx_images->idx_data;
    for (size_t i = 0; i < MNIST_NUM_TEST * MNIST_IMAGE_SIZE; i++) {
        data->test_images[i] = img_data[i] / 255.0f;
    }
    free_idx(idx_images);

    // Load test labels
    snprintf(path, sizeof(path), "%s/t10k-labels-idx1-ubyte", data_dir);
    idx_labels = idx_load(path);
    if (!idx_labels) {
        printf("Failed to load %s\n", path);
        free(data->train_images);
        free(data->train_labels);
        free(data->test_images);
        free(data);
        return NULL;
    }

    data->test_labels = (float*)malloc(MNIST_NUM_TEST * sizeof(float));
    lbl_data = (uint8_t*)idx_labels->idx_data;
    for (size_t i = 0; i < MNIST_NUM_TEST; i++) {
        data->test_labels[i] = (float)lbl_data[i];
    }
    free_idx(idx_labels);

    return data;
}

static void cl_mnist_free(CLMNISTData* data) {
    if (!data) return;
    free(data->train_images);
    free(data->train_labels);
    free(data->test_images);
    free(data->test_labels);
    free(data);
}

/* --------------------------------------------------------------------------
 * Create batch CLTensor from data
 * -------------------------------------------------------------------------- */
static CLTensor* cl_get_batch(CLOpenCL* cl, const float* images, const float* labels,
                               size_t start, size_t batch_size) {
    size_t shape[] = {batch_size, MNIST_IMAGE_SIZE};
    CLTensor* X = cl_tensor_create_from_host(cl, CL_TENSOR_DTYPE_F32,
                                              CL_TENSOR_LAYOUT_NCHW, shape, 2,
                                              images + start * MNIST_IMAGE_SIZE);
    (void)labels;  // labels not embedded in tensor for efficiency
    return X;
}

static CLTensor* cl_get_labels(CLOpenCL* cl, const float* labels, size_t start, size_t batch_size) {
    size_t shape[] = {batch_size};
    CLTensor* y = cl_tensor_create_from_host(cl, CL_TENSOR_DTYPE_F32,
                                              CL_TENSOR_LAYOUT_NCHW, shape, 1,
                                              labels + start);
    return y;
}

/* ============================================================================
 * Main Test
 * ============================================================================ */

int main(int argc, char* argv[]) {
    const char* data_dir = "data";
    if (argc > 1) data_dir = argv[1];

    printf("=== OpenCL MLP MNIST Test ===\n\n");

    // Initialize OpenCL
    CLOpenCL cl;
    memset(&cl, 0, sizeof(cl));

    if (!cl_init(&cl, CL_DEVICE_TYPE_GPU)) {
        printf("No GPU found, trying CPU...\n");
        if (!cl_init(&cl, CL_DEVICE_TYPE_CPU)) {
            printf("FAILED: Cannot initialize OpenCL\n");
            return 1;
        }
    }
    cl_print_device_info(cl.device);
    printf("\n");

    // Load MNIST data
    printf("Loading MNIST from %s...\n", data_dir);
    CLMNISTData* mnist = cl_mnist_load(data_dir);
    if (!mnist) {
        fprintf(stderr, "Failed to load MNIST from %s\n", data_dir);
        cl_release(&cl);
        return 1;
    }
    printf("Loaded: %zu train images, %zu test images\n\n",
           mnist->num_train, mnist->num_test);

    // Create MLP: 784 -> 256 -> 128 -> 10 (3-layer architecture)
    size_t input_dim = 784;
    size_t hidden_dim = 256;
    size_t hidden_dim2 = 128;
    size_t output_dim = 10;
    size_t num_layers = 3;

    printf("Creating OpenCL MLP: %zu -> %zu -> %zu -> %zu\n",
           input_dim, hidden_dim, hidden_dim2, output_dim);

    CLOpenCLMLP* mlp = cl_mlp_create(&cl, input_dim, hidden_dim, output_dim, num_layers);

    // Create SGD optimizer with momentum
    CLSGDOptimizer* opt = cl_sgd_create(&cl, mlp, 0.01f, 0.9f, 0.0001f);

    size_t batch_size = 64;
    size_t epochs = 5;
    size_t batches_per_epoch = mnist->num_train / batch_size;

    printf("\nTraining: epochs=%zu, batch_size=%zu, batches_per_epoch=%zu\n",
           epochs, batch_size, batches_per_epoch);

    // Quick sanity check: train on one batch first
    printf("\n=== Sanity Check: Overfit Single Batch ===\n");

    CLTensor* X_batch = cl_get_batch(&cl, mnist->train_images, mnist->train_labels, 0, batch_size);
    CLTensor* y_batch = cl_get_labels(&cl, mnist->train_labels, 0, batch_size);

    for (size_t iter = 0; iter < 30; iter++) {
        float loss = cl_mlp_train_step(&cl, &cl.kernel_cache, mlp, opt, X_batch, (const int*)(mnist->train_labels));
        float acc = cl_mlp_accuracy(&cl, &cl.kernel_cache, mlp, X_batch, (const int*)(mnist->train_labels));

        if (iter % 10 == 0) {
            printf("Iter %zu: Loss=%.4f, Acc=%.2f%%\n", iter, loss, 100.0f * acc);
        }
    }

    cl_tensor_free(X_batch);
    cl_tensor_free(y_batch);
    printf("=== Sanity Check Complete ===\n\n");

    // Full training
    printf("=== Full Training ===\n");
    clock_t start = clock();

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0.0f;

        // Shuffle indices (simple shuffle)
        size_t* indices = (size_t*)malloc(mnist->num_train * sizeof(size_t));
        for (size_t i = 0; i < mnist->num_train; i++) indices[i] = i;
        for (size_t i = mnist->num_train - 1; i > 0; i--) {
            size_t j = rand() % (i + 1);
            size_t tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }

        for (size_t b = 0; b < batches_per_epoch; b++) {
            size_t start = indices[b * batch_size];

            CLTensor* X_batch = cl_get_batch(&cl, mnist->train_images, mnist->train_labels, start, batch_size);

            float loss = cl_mlp_train_step(&cl, &cl.kernel_cache, mlp, opt, X_batch, (const int*)(mnist->train_labels + start));
            epoch_loss += loss;

            cl_tensor_free(X_batch);
        }

        free(indices);

        printf("Epoch %zu/%zu: avg_loss=%.4f\n", epoch + 1, epochs, epoch_loss / batches_per_epoch);

        // Evaluate on test set every epoch
        CLTensor* X_test_eval = cl_get_batch(&cl, mnist->test_images, mnist->test_labels, 0, 1000);
        float* y_test_labels = mnist->test_labels;
        float test_acc = cl_mlp_accuracy(&cl, &cl.kernel_cache, mlp, X_test_eval, (const int*)y_test_labels);
        printf("  Test accuracy: %.2f%%\n", 100.0f * test_acc);
        cl_tensor_free(X_test_eval);
    }

    clock_t end = clock();
    printf("\nTraining time: %.2f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    // Final evaluation
    printf("\n=== Final Evaluation ===\n");

    // Test on full test set in batches
    size_t test_batch = 500;
    size_t correct = 0;
    size_t total = 0;

    for (size_t i = 0; i < MNIST_NUM_TEST; i += test_batch) {
        size_t batch = (i + test_batch <= MNIST_NUM_TEST) ? test_batch : MNIST_NUM_TEST - i;
        CLTensor* X_test = cl_get_batch(&cl, mnist->test_images, mnist->test_labels, i, batch);
        CLTensor* y_test = cl_get_labels(&cl, mnist->test_labels, i, batch);

        CLTensor* pred = cl_mlp_predict(&cl, &cl.kernel_cache, mlp, X_test);

        float* h_pred = (float*)malloc(pred->nbytes);
        float* h_test = (float*)malloc(y_test->nbytes);
        cl_tensor_download(pred, h_pred);
        cl_tensor_download(y_test, h_test);

        for (size_t j = 0; j < batch; j++) {
            size_t pred_class = 0;
            float pred_max = h_pred[j * 10];
            for (size_t c = 1; c < 10; c++) {
                if (h_pred[j * 10 + c] > pred_max) {
                    pred_max = h_pred[j * 10 + c];
                    pred_class = c;
                }
            }
            size_t target_class = (size_t)h_test[j];
            if (pred_class == target_class) correct++;
            total++;
        }

        free(h_pred);
        free(h_test);
        cl_tensor_free(pred);
        cl_tensor_free(X_test);
        cl_tensor_free(y_test);
    }

    printf("Test Accuracy: %.2f%% (%zu/%zu correct)\n",
           100.0f * correct / total, correct, total);

    // Cleanup
    cl_mlp_free(mlp);
    cl_sgd_free(opt);
    cl_mnist_free(mnist);
    cl_release(&cl);

    printf("\n=== Test Complete ===\n");
    return 0;
}