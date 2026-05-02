/**
 * @file test_mlp_framework.c
 * @brief Test MLP as ML Framework model
 */

#define C_MACHINE_LEARNING_H
#include "machine_learning.h"
#include "dataset.h"

#define MLP_IMPLEMENTATION
#include "dl/mlp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char* argv[]) {
    printf("=== MLP Framework Adapter Test ===\n\n");

    // Create MLP model using framework adapter
    printf("Creating MLP model via ML framework...\n");

    // Use the ML framework adapter to create MLP model
    // 4 features -> 16 hidden -> 3 classes
    ML_Model_t* model = mlp_model_create(
        /*feature_dim*/ 4,
        /*n_classes*/ 3,
        /*hidden_dim*/ 16,
        /*num_layers*/ 2,
        /*lr*/ 0.01f,
        /*momentum*/ 0.9f,
        /*weight_decay*/ 0.0001f,
        /*epochs*/ 100,
        /*batch_size*/ 16
    );
    if (!model) {
        fprintf(stderr, "Failed to create ML model\n");
        return 1;
    }

    printf("ML_Model_t created successfully!\n");
    printf("  methods.fit: %p\n", (void*)model->methods.fit);
    printf("  methods.predict: %p\n", (void*)model->methods.predict);
    printf("  methods.predict_proba: %p\n", (void*)model->methods.predict_proba);
    printf("  methods.get_coefficients: %p\n", (void*)model->methods.get_coefficients);
    printf("  methods.free_state: %p\n", (void*)model->methods.free_state);

    // Cleanup
    printf("\nCleaning up...\n");
    if (model->methods.free_state) {
        model->methods.free_state(&model->state);
    }
    free(model);

    printf("\n=== Test Complete ===\n");
    return 0;
}