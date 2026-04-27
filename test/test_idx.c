/**
 * @file test_idx.c
 * @brief Test IDX file format reader
 */

#define IDX_IMPLEMENTATION
#include "idx.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    (void)argc; (void)argv;

    printf("=== IDX Format Test ===\n");

    cIDX* train_images = idx_load("data/train-images-idx3-ubyte");
    if (!train_images) { fprintf(stderr, "Failed to load train-images\n"); return 1; }
    printf("train-images: type=%d, n_dims=%u, dims=[%u, %u, %u], size=%zu\n",
           train_images->type, train_images->n_dims,
           train_images->dims[0], train_images->dims[1], train_images->dims[2],
           train_images->size);

    cIDX* train_labels = idx_load("data/train-labels-idx1-ubyte");
    if (!train_labels) { fprintf(stderr, "Failed to load train-labels\n"); free_idx(train_images); return 1; }
    printf("train-labels: type=%d, n_dims=%u, dims=[%u], size=%zu\n",
           train_labels->type, train_labels->n_dims,
           train_labels->dims[0], train_labels->size);

    cIDX* test_images = idx_load("data/t10k-images-idx3-ubyte");
    if (!test_images) { fprintf(stderr, "Failed to load t10k-images\n"); free_idx(train_images); free_idx(train_labels); return 1; }
    printf("t10k-images: type=%d, n_dims=%u, dims=[%u, %u, %u], size=%zu\n",
           test_images->type, test_images->n_dims,
           test_images->dims[0], test_images->dims[1], test_images->dims[2],
           test_images->size);

    cIDX* test_labels = idx_load("data/t10k-labels-idx1-ubyte");
    if (!test_labels) { fprintf(stderr, "Failed to load t10k-labels\n"); free_idx(train_images); free_idx(train_labels); free_idx(test_images); return 1; }
    printf("t10k-labels: type=%d, n_dims=%u, dims=[%u], size=%zu\n",
           test_labels->type, test_labels->n_dims,
           test_labels->dims[0], test_labels->size);

    // Verify data integrity
    uint8_t* img_data = (uint8_t*)train_images->idx_data;
    uint8_t* lbl_data = (uint8_t*)train_labels->idx_data;
    printf("First image pixel at [0,0,0] = %u\n", img_data[0]);
    printf("First 10 labels: ");
    for (int i = 0; i < 10; i++) printf("%u ", lbl_data[i]);
    printf("\n");

    free_idx(train_images);
    free_idx(train_labels);
    free_idx(test_images);
    free_idx(test_labels);

    printf("=== IDX Test Passed ===\n");
    return 0;
}
