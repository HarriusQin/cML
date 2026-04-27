/**
 * @file dl_loss.h
 * @brief Loss functions for deep learning
 *
 * Provides common loss functions:
 * - Mean Squared Error (MSE)
 * - Cross-Entropy Loss
 * - Binary Cross-Entropy
 * - Hinge Loss
 * - Smooth L1 (Huber) Loss
 */

#ifndef __DL_LOSS_H__
#define __DL_LOSS_H__

#include "dl_base.h"
#include "dl_tensor.h"

/* ============================================================================
 * LOSS FUNCTION RESULTS
 * ============================================================================ */

/**
 * @brief Result of computing a loss function
 */
typedef struct {
    double value;          /**< Loss value */
    Tensor *grad;          /**< Gradient w.r.t. predictions */
} DL_LossResult;

/* ============================================================================
 * LOSS FUNCTION TYPES
 * ============================================================================ */

typedef enum {
    DL_LOSS_MSE,                 /**< Mean squared error */
    DL_LOSS_CROSS_ENTROPY,       /**< Cross-entropy loss */
    DL_LOSS_BINARY_CROSS_ENTROPY, /**< Binary cross-entropy */
    DL_LOSS_HINGE,               /**< Hinge loss (SVM) */
    DL_LOSS_SMOOTH_L1,           /**< Smooth L1 (Huber) loss */
    DL_LOSS_FOCAL,               /**< Focal loss */
} DL_LossType;

/* ============================================================================
 * LOSS FUNCTION COMPUTATION
 * ============================================================================ */

/**
 * @brief Compute Mean Squared Error loss
 *
 * @param predictions Predicted values [*, n]
 * @param targets Target values [*, n]
 * @return Loss result with value and gradient
 */
DL_LossResult dl_mse_loss(const Tensor *predictions, const Tensor *targets);

/**
 * @brief Compute Cross-Entropy loss
 *
 * @param predictions Logits (before softmax) [*, num_classes]
 * @param targets Target class indices [*,] or one-hot [*, num_classes]
 * @param reduction Reduction type: 'mean', 'sum', or 'none'
 * @return Loss result with value and gradient
 */
DL_LossResult dl_cross_entropy_loss(const Tensor *predictions, const Tensor *targets,
                                    const char *reduction);

/**
 * @brief Compute Binary Cross-Entropy loss
 *
 * @param predictions Predicted probabilities [*, n]
 * @param targets Target values (0 or 1) [*, n]
 * @param reduction Reduction type: 'mean', 'sum', or 'none'
 * @return Loss result with value and gradient
 */
DL_LossResult dl_binary_cross_entropy_loss(const Tensor *predictions, const Tensor *targets,
                                            const char *reduction);

/**
 * @brief Compute Hinge loss (SVM-style)
 *
 * @param predictions Predicted values [*, n]
 * @param targets Target values (-1 or 1) [*, n]
 * @param margin Margin parameter (default: 1.0)
 * @return Loss result with value and gradient
 */
DL_LossResult dl_hinge_loss(const Tensor *predictions, const Tensor *targets, double margin);

/**
 * @brief Compute Smooth L1 (Huber) loss
 *
 * @param predictions Predicted values [*, n]
 * @param targets Target values [*, n]
 * @param beta Threshold for smooth region
 * @return Loss result with value and gradient
 */
DL_LossResult dl_smooth_l1_loss(const Tensor *predictions, const Tensor *targets, double beta);

/**
 * @brief Compute Focal loss
 *
 * @param predictions Predicted probabilities [*, num_classes]
 * @param targets Target class indices [*,] or one-hot [*, num_classes]
 * @param alpha Weighting factor
 * @param gamma Focusing parameter
 * @return Loss result with value and gradient
 */
DL_LossResult dl_focal_loss(const Tensor *predictions, const Tensor *targets,
                            double alpha, double gamma);

/* ============================================================================
 * LOSS FUNCTION HELPERS
 * ============================================================================ */

/**
 * @brief Compute softmax cross-entropy loss with logits
 *
 * Combines softmax and cross-entropy for numerical stability.
 *
 * @param logits Raw logits (before softmax) [*, num_classes]
 * @param targets Target class indices [*,]
 * @return Loss value
 */
double dl_softmax_cross_entropy(const Tensor *logits, const Tensor *targets);

/**
 * @brief NLL (Negative Log Likelihood) loss
 *
 * @param log_probs Log probabilities [*, num_classes]
 * @param targets Target class indices [*,]
 * @return Loss value
 */
double dl_nll_loss(const Tensor *log_probs, const Tensor *targets);

/**
 * @brief Free loss result
 */
void dl_loss_free(DL_LossResult *result);

#endif /* __DL_LOSS_H__ */
