/**
 * @file dl_optimizer.h
 * @brief Optimizers for deep learning
 *
 * Provides gradient-based optimization algorithms:
 * - SGD (Stochastic Gradient Descent)
 * - SGD with Momentum
 * - Nesterov Accelerated Gradient
 * - Adam (Adaptive Moment Estimation)
 * - AdamW (Adam with Weight Decay)
 * - RMSprop
 * - AdaGrad
 */

#ifndef __DL_OPTIMIZER_H__
#define __DL_OPTIMIZER_H__

#include "dl_base.h"
#include "dl_tensor.h"

/* ============================================================================
 * OPTIMIZER TYPES
 * ============================================================================ */

typedef enum {
    DL_OPT_SGD,           /**< Stochastic Gradient Descent */
    DL_OPT_MOMENTUM,      /**< SGD with Momentum */
    DL_OPT_NESTEROV,      /**< Nesterov Accelerated Gradient */
    DL_OPT_ADAM,          /**< Adam */
    DL_OPT_ADAMW,         /**< AdamW (Adam with Weight Decay) */
    DL_OPT_RMSPROP,       /**< RMSprop */
    DL_OPT_ADAGRAD        /**< AdaGrad */
} DL_OptimizerType;

/* ============================================================================
 * OPTIMIZER STATE
 * ============================================================================ */

/**
 * @brief Optimizer state for a single parameter
 */
typedef struct {
    Tensor *exp_avg;      /**< Exponential moving average of gradients (momentum) */
    Tensor *exp_avg_sq;   /**< Exponential moving average of squared gradients */
    Tensor *delta;         /**< Parameter update delta */
    size_t step;          /**< Number of optimization steps */
} DL_OptimizerState;

/**
 * @brief Base optimizer structure
 */
typedef struct DL_Optimizer {
    DL_OptimizerType type;
    double learning_rate;
    double weight_decay;
    double momentum;
    double beta1;          /**< Adam beta1 */
    double beta2;         /**< Adam beta2 */
    double eps;           /**< Numerical stability epsilon */
    double alpha;          /**< Nesterov lookahead */
    double rmsprop_decay;  /**< RMSprop decay rate */

    /* State per parameter */
    DL_OptimizerState *states;
    size_t n_states;

    /* Parameter references */
    Tensor **parameters;
    size_t n_parameters;
} DL_Optimizer;

/* ============================================================================
 * OPTIMIZER FUNCTIONS
 * ============================================================================ */

/**
 * @brief Create an optimizer
 *
 * @param type Optimizer type
 * @param learning_rate Learning rate
 * @return New optimizer or NULL on failure
 */
DL_Optimizer *dl_optimizer_create(DL_OptimizerType type, double learning_rate);

/**
 * @brief Configure optimizer parameters
 */
void dl_optimizer_set_param(DL_Optimizer *opt, const char *name, double value);

/**
 * @brief Register parameters for optimization
 */
void dl_optimizer_register_params(DL_Optimizer *opt, Tensor **params, size_t n_params);

/**
 * @brief Zero gradients of all registered parameters
 */
void dl_optimizer_zero_grad(DL_Optimizer *opt);

/**
 * @brief Perform one optimization step
 *
 * @param opt Optimizer
 * @param gradients Array of gradient tensors (NULL to use stored gradients)
 * @param n_grads Number of gradients
 */
void dl_optimizer_step(DL_Optimizer *opt, Tensor **gradients, size_t n_grads);

/**
 * @brief Free optimizer and release resources
 */
void dl_optimizer_free(DL_Optimizer *opt);

/* ============================================================================
 * SPECIFIC OPTIMIZERS
 * ============================================================================ */

/**
 * @brief SGD optimizer
 */
typedef struct {
    DL_Optimizer base;
} DL_SGDOptimizer;

/**
 * @brief Create SGD optimizer
 */
DL_SGDOptimizer *dl_sgd_create(double learning_rate);

/**
 * @brief Adam optimizer
 */
typedef struct {
    DL_Optimizer base;
    double beta1;
    double beta2;
    double eps;
    size_t t;  /**< Step counter */
} DL_AdamOptimizer;

/**
 * @brief Create Adam optimizer
 */
DL_AdamOptimizer *dl_adam_create(double learning_rate, double beta1, double beta2, double eps);

/**
 * @brief AdamW optimizer
 */
typedef struct {
    DL_Optimizer base;
    double beta1;
    double beta2;
    double eps;
    double weight_decay;
    size_t t;
} DL_AdamWOptimizer;

/**
 * @brief Create AdamW optimizer
 */
DL_AdamWOptimizer *dl_adamw_create(double learning_rate, double weight_decay,
                                   double beta1, double beta2, double eps);

/**
 * @brief Momentum optimizer
 */
typedef struct {
    DL_Optimizer base;
    double momentum;
} DL_MomentumOptimizer;

/**
 * @brief Create Momentum optimizer
 */
DL_MomentumOptimizer *dl_momentum_create(double learning_rate, double momentum);

/**
 * @brief RMSprop optimizer
 */
typedef struct {
    DL_Optimizer base;
    double decay;
    double eps;
} DL_RMSpropOptimizer;

/**
 * @brief Create RMSprop optimizer
 */
DL_RMSpropOptimizer *dl_rmsprop_create(double learning_rate, double decay, double eps);

/* ============================================================================
 * LEARNING RATE SCHEDULING
 * ============================================================================ */

/**
 * @brief Learning rate scheduler types
 */
typedef enum {
    DL_LR_CONSTANT,      /**< Constant learning rate */
    DL_LR_STEP,          /**< Step decay */
    DL_LR_EXPONENTIAL,   /**< Exponential decay */
    DL_LR_COSINE,        /**< Cosine annealing */
    DL_LR_PLATEAU        /**< Reduce on plateau */
} DL_LRSchedulerType;

/**
 * @brief Learning rate scheduler
 */
typedef struct {
    DL_LRSchedulerType type;
    double initial_lr;
    double current_lr;
    double gamma;         /**< Decay factor for step/exp */
    size_t step_size;     /**< Step size for step LR */
    double T_max;         /**< Max iterations for cosine */
    double min_lr;        /**< Minimum LR */
} DL_LRScheduler;

/**
 * @brief Create learning rate scheduler
 */
DL_LRScheduler *dl_lr_scheduler_create(DL_LRSchedulerType type, double initial_lr, double gamma);

/**
 * @brief Step the scheduler
 */
void dl_lr_scheduler_step(DL_LRScheduler *scheduler);

/**
 * @brief Get current learning rate
 */
double dl_lr_scheduler_get_lr(const DL_LRScheduler *scheduler);

/**
 * @brief Free scheduler
 */
void dl_lr_scheduler_free(DL_LRScheduler *scheduler);

#endif /* __DL_OPTIMIZER_H__ */
