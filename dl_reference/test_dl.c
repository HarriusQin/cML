/**
 * @file test_dl.c
 * @brief Test program for deep learning library
 *
 * Tests basic functionality:
 * - Tensor creation and operations
 * - Autograd
 * - MLP model
 */

#include "dl_tensor.h"
#include "dl_layers.h"
#include "dl_loss.h"
#include "dl_optimizer.h"
#include "dl_model.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ANSI_COLOR_GREEN "\x1b[32m"
#define ANSI_COLOR_RED "\x1b[31m"
#define ANSI_COLOR_YELLOW "\x1b[33m"
#define ANSI_COLOR_RESET "\x1b[0m"

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (cond) { \
        printf(ANSI_COLOR_GREEN "[PASS]" ANSI_COLOR_RESET " %s\n", msg); \
        tests_passed++; \
    } else { \
        printf(ANSI_COLOR_RED "[FAIL]" ANSI_COLOR_RESET " %s\n", msg); \
        tests_failed++; \
    } \
} while(0)

/* ============================================================================
 * TENSOR TESTS
 * ============================================================================ */

void test_tensor_creation(void) {
    printf("\n=== Tensor Creation Tests ===\n");

    size_t shape2d[] = {3, 4};
    Tensor *t = tensor_create(2, shape2d, DL_DEVICE_CPU, false);
    ASSERT(t != NULL && t->size == 12, "tensor_create 2D");

    Tensor *zeros = tensor_zeros(2, shape2d, DL_DEVICE_CPU, false);
    ASSERT(zeros != NULL, "tensor_zeros");

    /* Check all zeros */
    bool all_zeros = true;
    for (size_t i = 0; i < zeros->size; i++) {
        if (zeros->data[i] != 0) all_zeros = false;
    }
    ASSERT(all_zeros, "tensor_zeros values");
    tensor_free(t);
    tensor_free(zeros);

    Tensor *ones = tensor_ones(2, shape2d, DL_DEVICE_CPU, false);
    ASSERT(ones != NULL, "tensor_ones");
    tensor_free(ones);

    Tensor *full = tensor_full(2, shape2d, 5.0, DL_DEVICE_CPU, false);
    ASSERT(full != NULL, "tensor_full");
    bool all_five = true;
    for (size_t i = 0; i < full->size; i++) {
        if (full->data[i] != 5.0) all_five = false;
    }
    ASSERT(all_five, "tensor_full values");
    tensor_free(full);

    /* Random tensor */
    Tensor *randn = tensor_randn(2, shape2d, 0.0, 1.0, false);
    ASSERT(randn != NULL, "tensor_randn");
    tensor_free(randn);

    printf("  Created %d tests, %d passed, %d failed\n",
           tests_passed + tests_failed, tests_passed, tests_failed);
}

void test_tensor_operations(void) {
    printf("\n=== Tensor Operations Tests ===\n");

    size_t shape[] = {2, 3};
    double a_data[] = {1, 2, 3, 4, 5, 6};
    double b_data[] = {6, 5, 4, 3, 2, 1};

    Tensor *a = tensor_create(2, shape, DL_DEVICE_CPU, false);
    Tensor *b = tensor_create(2, shape, DL_DEVICE_CPU, false);
    for (size_t i = 0; i < 6; i++) {
        a->data[i] = a_data[i];
        b->data[i] = b_data[i];
    }

    /* Addition */
    Tensor *sum = tensor_add(a, b);
    ASSERT(sum != NULL, "tensor_add");
    bool sum_correct = true;
    for (size_t i = 0; i < 6; i++) {
        if (sum->data[i] != a_data[i] + b_data[i]) sum_correct = false;
    }
    ASSERT(sum_correct, "tensor_add values");
    tensor_free(sum);

    /* Subtraction */
    Tensor *diff = tensor_sub(a, b);
    ASSERT(diff != NULL, "tensor_sub");
    tensor_free(diff);

    /* Element-wise multiplication */
    Tensor *prod = tensor_mul(a, b);
    ASSERT(prod != NULL, "tensor_mul");
    bool prod_correct = true;
    for (size_t i = 0; i < 6; i++) {
        if (prod->data[i] != a_data[i] * b_data[i]) prod_correct = false;
    }
    ASSERT(prod_correct, "tensor_mul values");
    tensor_free(prod);

    /* Scalar multiplication */
    Tensor *scaled = tensor_mul_scalar(a, 2.0);
    ASSERT(scaled != NULL, "tensor_mul_scalar");
    bool scaled_correct = true;
    for (size_t i = 0; i < 6; i++) {
        if (scaled->data[i] != a_data[i] * 2.0) scaled_correct = false;
    }
    ASSERT(scaled_correct, "tensor_mul_scalar values");
    tensor_free(scaled);

    /* Sum */
    double sum_val = tensor_sum(a);
    ASSERT(sum_val == 21.0, "tensor_sum");

    /* Mean */
    double mean_val = tensor_mean(a);
    ASSERT(mean_val == 3.5, "tensor_mean");

    tensor_free(a);
    tensor_free(b);

    printf("  Created %d tests, %d passed, %d failed\n",
           tests_passed + tests_failed, tests_passed, tests_failed);
}

void test_matrix_operations(void) {
    printf("\n=== Matrix Operations Tests ===\n");

    /* Matrix multiplication: A [2x3] @ B [3x2] = C [2x2] */
    size_t a_shape[] = {2, 3};
    size_t b_shape[] = {3, 2};
    double a_vals[] = {1, 2, 3, 4, 5, 6};
    double b_vals[] = {1, 2, 3, 4, 5, 6};

    Tensor *A = tensor_create(2, a_shape, DL_DEVICE_CPU, false);
    Tensor *B = tensor_create(2, b_shape, DL_DEVICE_CPU, false);
    for (size_t i = 0; i < 6; i++) {
        A->data[i] = a_vals[i];
        B->data[i] = b_vals[i];
    }

    Tensor *C = tensor_matmul(A, B);
    ASSERT(C != NULL, "tensor_matmul");
    ASSERT(C->shape[0] == 2 && C->shape[1] == 2, "tensor_matmul shape");

    /* Expected: [[22, 28], [49, 64]] */
    double expected[] = {22, 28, 49, 64};
    bool correct = true;
    for (size_t i = 0; i < 4; i++) {
        if (fabs(C->data[i] - expected[i]) > 1e-10) correct = false;
    }
    ASSERT(correct, "tensor_matmul values");

    /* Transpose */
    Tensor *A_t = tensor_transpose(A);
    ASSERT(A_t->shape[0] == 3 && A_t->shape[1] == 2, "tensor_transpose shape");
    tensor_free(A_t);

    /* Dot product */
    double dot = tensor_dot(A, B);
    ASSERT(dot > 0, "tensor_dot");

    tensor_free(A);
    tensor_free(B);
    tensor_free(C);

    printf("  Created %d tests, %d passed, %d failed\n",
           tests_passed + tests_failed, tests_passed, tests_failed);
}

/* ============================================================================
 * ACTIVATION TESTS
 * ============================================================================ */

void test_activations(void) {
    printf("\n=== Activation Function Tests ===\n");

    size_t shape[] = {5};
    Tensor *x = tensor_create(1, shape, DL_DEVICE_CPU, false);
    x->data[0] = -1.0;
    x->data[1] = 0.0;
    x->data[2] = 1.0;
    x->data[3] = 2.0;
    x->data[4] = -2.0;

    /* ReLU: max(0, x) */
    Tensor *relu = tensor_relu(x);
    ASSERT(relu != NULL, "tensor_relu");
    ASSERT(relu->data[0] == 0.0 && relu->data[2] == 1.0, "tensor_relu values");
    tensor_free(relu);

    /* Sigmoid: 1 / (1 + exp(-x)) */
    Tensor *sigmoid = tensor_sigmoid(x);
    ASSERT(sigmoid != NULL, "tensor_sigmoid");
    ASSERT(sigmoid->data[2] > 0.7 && sigmoid->data[2] < 0.8, "tensor_sigmoid values");
    tensor_free(sigmoid);

    /* Tanh */
    Tensor *tanh = tensor_tanh_activation(x);
    ASSERT(tanh != NULL, "tensor_tanh");
    tensor_free(tanh);

    /* GELU approximation */
    Tensor *gelu = tensor_gelu(x);
    ASSERT(gelu != NULL, "tensor_gelu");
    tensor_free(gelu);

    tensor_free(x);

    printf("  Created %d tests, %d passed, %d failed\n",
           tests_passed + tests_failed, tests_passed, tests_failed);
}

/* ============================================================================
 * MLP TESTS
 * ============================================================================ */

void test_mlp(void) {
    printf("\n=== MLP Tests ===\n");

    /* Create a simple MLP: 4 -> 8 -> 3 */
    size_t hidden_sizes[] = {8};
    DL_MLPConfig config = {
        .input_size = 4,
        .hidden_sizes = hidden_sizes,
        .n_hidden = 1,
        .output_size = 3,
        .use_bias = true,
        .activation = DL_ACTIVATION_RELU
    };

    DL_MLP *mlp = dl_mlp_create(&config);
    ASSERT(mlp != NULL, "dl_mlp_create");
    ASSERT(mlp->n_layers == 2, "dl_mlp layers count");

    /* Forward pass */
    size_t input_shape[] = {2, 4};  /* batch=2, features=4 */
    Tensor *input = tensor_create(2, input_shape, DL_DEVICE_CPU, false);
    for (size_t i = 0; i < input->size; i++) {
        input->data[i] = (double)i / (double)input->size;
    }

    Tensor *output = dl_mlp_forward(mlp, input);
    ASSERT(output != NULL, "dl_mlp_forward");
    ASSERT(output->shape[0] == 2 && output->shape[1] == 3, "dl_mlp_forward shape");

    tensor_free(input);
    tensor_free(output);
    dl_mlp_free(mlp);

    printf("  Created %d tests, %d passed, %d failed\n",
           tests_passed + tests_failed, tests_passed, tests_failed);
}

/* ============================================================================
 * OPTIMIZER TESTS
 * ============================================================================ */

void test_optimizer(void) {
    printf("\n=== Optimizer Tests ===\n");

    /* Create a simple parameter */
    size_t shape[] = {3};
    Tensor *param = tensor_create(1, shape, DL_DEVICE_CPU, true);
    param->data[0] = 1.0;
    param->data[1] = 2.0;
    param->data[2] = 3.0;

    /* Create gradient */
    Tensor *grad = tensor_create(1, shape, DL_DEVICE_CPU, false);
    grad->data[0] = 0.1;
    grad->data[1] = 0.2;
    grad->data[2] = 0.3;
    param->grad = grad;

    /* Create SGD optimizer */
    DL_Optimizer *sgd = dl_optimizer_create(DL_OPT_SGD, 0.1);
    ASSERT(sgd != NULL, "dl_optimizer_create SGD");
    dl_optimizer_register_params(sgd, &param, 1);

    /* Save original values */
    double orig0 = param->data[0], orig1 = param->data[1], orig2 = param->data[2];

    /* Step */
    dl_optimizer_step(sgd, NULL, 0);

    /* Check parameters were updated */
    bool updated = (param->data[0] != orig0 || param->data[1] != orig1 || param->data[2] != orig2);
    ASSERT(updated, "SGD parameter update");

    dl_optimizer_free(sgd);

    /* Test Adam */
    DL_Optimizer *adam = dl_optimizer_create(DL_OPT_ADAM, 0.01);
    ASSERT(adam != NULL, "dl_optimizer_create Adam");

    /* Reset param */
    param->data[0] = 1.0;
    param->data[1] = 2.0;
    param->data[2] = 3.0;

    dl_optimizer_register_params(adam, &param, 1);
    dl_optimizer_step(adam, NULL, 0);

    updated = (param->data[0] != orig0 || param->data[1] != orig1 || param->data[2] != orig2);
    ASSERT(updated, "Adam parameter update");

    dl_optimizer_free(adam);
    tensor_free(param);

    printf("  Created %d tests, %d passed, %d failed\n",
           tests_passed + tests_failed, tests_passed, tests_failed);
}

/* ============================================================================
 * LOSS FUNCTION TESTS
 * ============================================================================ */

void test_losses(void) {
    printf("\n=== Loss Function Tests ===\n");

    size_t shape[] = {4};
    Tensor *pred = tensor_create(1, shape, DL_DEVICE_CPU, false);
    Tensor *target = tensor_create(1, shape, DL_DEVICE_CPU, false);

    pred->data[0] = 0.9;
    pred->data[1] = 0.1;
    pred->data[2] = 0.8;
    pred->data[3] = 0.2;

    target->data[0] = 1.0;
    target->data[1] = 0.0;
    target->data[2] = 1.0;
    target->data[3] = 0.0;

    /* MSE Loss */
    DL_LossResult mse = dl_mse_loss(pred, target);
    ASSERT(mse.grad != NULL, "MSE loss gradient");
    ASSERT(mse.value > 0, "MSE loss value");
    dl_loss_free(&mse);

    /* Binary Cross-Entropy */
    DL_LossResult bce = dl_binary_cross_entropy_loss(pred, target, "mean");
    ASSERT(bce.grad != NULL, "BCE loss gradient");
    ASSERT(bce.value > 0, "BCE loss value");
    dl_loss_free(&bce);

    tensor_free(pred);
    tensor_free(target);

    printf("  Created %d tests, %d passed, %d failed\n",
           tests_passed + tests_failed, tests_passed, tests_failed);
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    srand((unsigned int)time(NULL));

    printf("================================================================================\n");
    printf("              Deep Learning Library (cDL) Test Suite                            \n");
    printf("================================================================================\n");

    test_tensor_creation();
    test_tensor_operations();
    test_matrix_operations();
    test_activations();
    test_mlp();
    test_optimizer();
    test_losses();

    printf("\n================================================================================\n");
    printf("                           Test Summary                                        \n");
    printf("================================================================================\n");
    printf("\nTotal:  %d tests\n", tests_passed + tests_failed);
    printf(ANSI_COLOR_GREEN "Passed:  %d" ANSI_COLOR_RESET "\n", tests_passed);
    printf(ANSI_COLOR_RED "Failed:  %d" ANSI_COLOR_RESET "\n", tests_failed);

    if (tests_failed == 0) {
        printf("\n" ANSI_COLOR_GREEN "All tests passed!" ANSI_COLOR_RESET "\n");
        return 0;
    } else {
        printf("\n" ANSI_COLOR_RED "Some tests failed." ANSI_COLOR_RESET "\n");
        return 1;
    }
}
