/**
 * @file dl_loss.c
 * @brief Loss function implementations
 */

#include "dl_loss.h"
#include "dl_layers.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * MSE LOSS
 * ============================================================================ */

DL_LossResult dl_mse_loss(const Tensor *predictions, const Tensor *targets) {
    DL_LossResult result = {0};

    if (!predictions || !targets) return result;

    size_t size = predictions->size;
    double sum = 0.0;

    for (size_t i = 0; i < size; i++) {
        double diff = predictions->data[i] - targets->data[i];
        sum += diff * diff;
    }

    result.value = sum / (double)size;

    /* Gradient: d(LMSE)/d(pred) = 2 * (pred - target) / n */
    result.grad = tensor_sub(predictions, targets);
    Tensor *scale = tensor_full(1, (size_t[]){1}, 2.0 / (double)size, DL_DEVICE_CPU, false);
    Tensor *scaled_grad = tensor_mul_scalar(result.grad, 2.0 / (double)size);
    tensor_free(result.grad);
    result.grad = scaled_grad;

    tensor_free(scale);
    return result;
}

/* ============================================================================
 * CROSS-ENTROPY LOSS
 * ============================================================================ */

static double log_sum_exp(const Tensor *x, size_t axis) {
    /* Compute log(sum(exp(x))) with numerical stability */
    size_t n = x->shape[axis];
    double max_val = -1e10;

    for (size_t i = 0; i < n; i++) {
        if (x->data[i] > max_val) max_val = x->data[i];
    }

    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += exp(x->data[i] - max_val);
    }

    return max_val + log(sum);
}

DL_LossResult dl_cross_entropy_loss(const Tensor *predictions, const Tensor *targets,
                                    const char *reduction) {
    DL_LossResult result = {0};

    if (!predictions || !targets) return result;

    size_t batch_size = predictions->shape[0];
    size_t num_classes = predictions->shape[1];

    /* Compute softmax probabilities */
    Tensor *exp_preds = tensor_create(predictions->ndim, predictions->shape,
                                       predictions->device, false);
    double *exp_data = (double *)malloc(sizeof(double) * predictions->size);

    /* Subtract max for numerical stability */
    for (size_t b = 0; b < batch_size; b++) {
        double max_val = -1e10;
        for (size_t c = 0; c < num_classes; c++) {
            size_t idx = b * num_classes + c;
            if (predictions->data[idx] > max_val) max_val = predictions->data[idx];
        }

        double sum = 0.0;
        for (size_t c = 0; c < num_classes; c++) {
            size_t idx = b * num_classes + c;
            exp_data[idx] = exp(predictions->data[idx] - max_val);
            sum += exp_data[idx];
        }

        for (size_t c = 0; c < num_classes; c++) {
            size_t idx = b * num_classes + c;
            exp_data[idx] /= sum;
        }
    }

    memcpy(exp_preds->data, exp_data, sizeof(double) * predictions->size);
    free(exp_data);

    /* Compute cross-entropy loss */
    double total_loss = 0.0;
    result.grad = tensor_create(predictions->ndim, predictions->shape,
                                predictions->device, false);

    for (size_t b = 0; b < batch_size; b++) {
        int target_class = (int)targets->data[b];
        size_t idx = b * num_classes + target_class;

        /* Add small epsilon for numerical stability */
        double prob = exp_preds->data[idx] + 1e-8;
        total_loss -= log(prob);

        /* Gradient: softmax_i - target_i */
        for (size_t c = 0; c < num_classes; c++) {
            size_t grad_idx = b * num_classes + c;
            double target = (c == (size_t)target_class) ? 1.0 : 0.0;
            result.grad->data[grad_idx] = exp_preds->data[grad_idx] - target;
        }
    }

    /* Apply reduction */
    if (strcmp(reduction, "mean") == 0) {
        result.value = total_loss / (double)batch_size;
        Tensor *scale = tensor_full(1, (size_t[]){1}, 1.0 / (double)batch_size,
                                    DL_DEVICE_CPU, false);
        Tensor *scaled_grad = tensor_mul_scalar(result.grad, 1.0 / (double)batch_size);
        tensor_free(result.grad);
        result.grad = scaled_grad;
        tensor_free(scale);
    } else if (strcmp(reduction, "sum") == 0) {
        result.value = total_loss;
    } else {
        result.value = total_loss;
    }

    tensor_free(exp_preds);
    return result;
}

/* ============================================================================
 * BINARY CROSS-ENTROPY LOSS
 * ============================================================================ */

DL_LossResult dl_binary_cross_entropy_loss(const Tensor *predictions, const Tensor *targets,
                                            const char *reduction) {
    DL_LossResult result = {0};

    if (!predictions || !targets) return result;

    size_t size = predictions->size;
    double total_loss = 0.0;

    result.grad = tensor_create(predictions->ndim, predictions->shape,
                                predictions->device, false);

    for (size_t i = 0; i < size; i++) {
        double pred = predictions->data[i];
        double target = targets->data[i];

        /* Clip predictions for numerical stability */
        pred = pred < 1e-8 ? 1e-8 : (pred > 1 - 1e-8 ? 1 - 1e-8 : pred);

        total_loss -= target * log(pred) + (1 - target) * log(1 - pred);

        /* Gradient: (pred - target) / (pred * (1 - pred)) */
        result.grad->data[i] = (pred - target) / (pred * (1 - pred));
    }

    if (strcmp(reduction, "mean") == 0) {
        result.value = total_loss / (double)size;
        Tensor *scale = tensor_full(1, (size_t[]){1}, 1.0 / (double)size, DL_DEVICE_CPU, false);
        Tensor *scaled_grad = tensor_mul_scalar(result.grad, 1.0 / (double)size);
        tensor_free(result.grad);
        result.grad = scaled_grad;
        tensor_free(scale);
    } else if (strcmp(reduction, "sum") == 0) {
        result.value = total_loss;
    } else {
        result.value = total_loss / (double)size;
    }

    return result;
}

/* ============================================================================
 * HINGE LOSS
 * ============================================================================ */

DL_LossResult dl_hinge_loss(const Tensor *predictions, const Tensor *targets, double margin) {
    DL_LossResult result = {0};

    if (!predictions || !targets) return result;

    size_t size = predictions->size;
    double total_loss = 0.0;

    result.grad = tensor_create(predictions->ndim, predictions->shape,
                                predictions->device, false);

    for (size_t i = 0; i < size; i++) {
        double pred = predictions->data[i];
        double target = targets->data[i];

        double loss_val = fmax(0, margin - target * pred);
        total_loss += loss_val;

        /* Gradient: -target if loss > 0, else 0 */
        result.grad->data[i] = (loss_val > 0) ? -target : 0;
    }

    result.value = total_loss / (double)size;
    Tensor *scale = tensor_full(1, (size_t[]){1}, 1.0 / (double)size, DL_DEVICE_CPU, false);
    Tensor *scaled_grad = tensor_mul_scalar(result.grad, 1.0 / (double)size);
    tensor_free(result.grad);
    result.grad = scaled_grad;
    tensor_free(scale);

    return result;
}

/* ============================================================================
 * SMOOTH L1 (HUBER) LOSS
 * ============================================================================ */

DL_LossResult dl_smooth_l1_loss(const Tensor *predictions, const Tensor *targets, double beta) {
    DL_LossResult result = {0};

    if (!predictions || !targets) return result;

    size_t size = predictions->size;
    double total_loss = 0.0;

    result.grad = tensor_create(predictions->ndim, predictions->shape,
                                predictions->device, false);

    for (size_t i = 0; i < size; i++) {
        double diff = predictions->data[i] - targets->data[i];
        double abs_diff = fabs(diff);

        if (abs_diff < beta) {
            total_loss += 0.5 * diff * diff / beta;
            result.grad->data[i] = diff / beta;
        } else {
            total_loss += abs_diff - 0.5 * beta;
            result.grad->data[i] = (diff > 0) ? 1.0 : -1.0;
        }
    }

    result.value = total_loss / (double)size;
    Tensor *scale = tensor_full(1, (size_t[]){1}, 1.0 / (double)size, DL_DEVICE_CPU, false);
    Tensor *scaled_grad = tensor_mul_scalar(result.grad, 1.0 / (double)size);
    tensor_free(result.grad);
    result.grad = scaled_grad;
    tensor_free(scale);

    return result;
}

/* ============================================================================
 * FOCAL LOSS
 * ============================================================================ */

DL_LossResult dl_focal_loss(const Tensor *predictions, const Tensor *targets,
                            double alpha, double gamma) {
    DL_LossResult result = {0};

    if (!predictions || !targets) return result;

    size_t batch_size = predictions->shape[0];
    size_t num_classes = predictions->shape[1];

    /* Compute softmax probabilities */
    Tensor *exp_preds = tensor_create(predictions->ndim, predictions->shape,
                                       predictions->device, false);
    double *exp_data = (double *)malloc(sizeof(double) * predictions->size);

    for (size_t b = 0; b < batch_size; b++) {
        double max_val = -1e10;
        for (size_t c = 0; c < num_classes; c++) {
            size_t idx = b * num_classes + c;
            if (predictions->data[idx] > max_val) max_val = predictions->data[idx];
        }

        double sum = 0.0;
        for (size_t c = 0; c < num_classes; c++) {
            size_t idx = b * num_classes + c;
            exp_data[idx] = exp(predictions->data[idx] - max_val);
            sum += exp_data[idx];
        }

        for (size_t c = 0; c < num_classes; c++) {
            size_t idx = b * num_classes + c;
            exp_data[idx] /= sum;
        }
    }

    memcpy(exp_preds->data, exp_data, sizeof(double) * predictions->size);
    free(exp_data);

    /* Compute focal loss */
    double total_loss = 0.0;
    result.grad = tensor_create(predictions->ndim, predictions->shape,
                                predictions->device, false);

    for (size_t b = 0; b < batch_size; b++) {
        int target_class = (int)targets->data[b];
        size_t idx = b * num_classes + target_class;

        double p = exp_preds->data[idx] + 1e-8;
        double p_t = (target_class >= 0) ? p : 1.0 - p;

        double focal_weight = pow(1.0 - p_t, gamma);
        double alpha_t = alpha;

        total_loss -= alpha_t * focal_weight * log(p_t);

        /* Gradient */
        for (size_t c = 0; c < num_classes; c++) {
            size_t grad_idx = b * num_classes + c;
            double target = (c == (size_t)target_class) ? 1.0 : 0.0;
            double p_c = exp_preds->data[grad_idx];

            if (c == (size_t)target_class) {
                result.grad->data[grad_idx] = -alpha_t * focal_weight * (1.0 - p_t) * log(p_t) * gamma * p_c / (1.0 - p_t);
            } else {
                result.grad->data[grad_idx] = alpha_t * focal_weight * gamma * p_c * log(p_t);
            }
        }
    }

    result.value = total_loss / (double)batch_size;

    tensor_free(exp_preds);
    return result;
}

/* ============================================================================
 * HELPERS
 * ============================================================================ */

double dl_softmax_cross_entropy(const Tensor *logits, const Tensor *targets) {
    DL_LossResult loss = dl_cross_entropy_loss(logits, targets, "mean");
    double result = loss.value;
    if (loss.grad) tensor_free(loss.grad);
    return result;
}

double dl_nll_loss(const Tensor *log_probs, const Tensor *targets) {
    size_t batch_size = log_probs->shape[0];
    double total_loss = 0.0;

    for (size_t b = 0; b < batch_size; b++) {
        int target_class = (int)targets->data[b];
        size_t idx = b * log_probs->shape[1] + target_class;
        total_loss -= log_probs->data[idx];
    }

    return total_loss / (double)batch_size;
}

void dl_loss_free(DL_LossResult *result) {
    if (!result) return;
    if (result->grad) {
        tensor_free(result->grad);
        result->grad = NULL;
    }
    result->value = 0;
}
