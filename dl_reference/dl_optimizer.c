/**
 * @file dl_optimizer.c
 * @brief Optimizer implementations
 */

#include "dl_optimizer.h"
#include "dl_tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * BASE OPTIMIZER
 * ============================================================================ */

DL_Optimizer *dl_optimizer_create(DL_OptimizerType type, double learning_rate) {
    DL_Optimizer *opt = (DL_Optimizer *)calloc(1, sizeof(DL_Optimizer));
    if (!opt) return NULL;

    opt->type = type;
    opt->learning_rate = learning_rate;
    opt->weight_decay = 0.0;
    opt->momentum = 0.0;
    opt->beta1 = 0.9;
    opt->beta2 = 0.999;
    opt->eps = 1e-8;
    opt->alpha = 1.0;
    opt->rmsprop_decay = 0.99;
    opt->states = NULL;
    opt->n_states = 0;
    opt->parameters = NULL;
    opt->n_parameters = 0;

    return opt;
}

void dl_optimizer_set_param(DL_Optimizer *opt, const char *name, double value) {
    if (!opt) return;

    if (strcmp(name, "weight_decay") == 0) {
        opt->weight_decay = value;
    } else if (strcmp(name, "momentum") == 0) {
        opt->momentum = value;
    } else if (strcmp(name, "beta1") == 0) {
        opt->beta1 = value;
    } else if (strcmp(name, "beta2") == 0) {
        opt->beta2 = value;
    } else if (strcmp(name, "eps") == 0) {
        opt->eps = value;
    } else if (strcmp(name, "alpha") == 0) {
        opt->alpha = value;
    } else if (strcmp(name, "rmsprop_decay") == 0) {
        opt->rmsprop_decay = value;
    }
}

void dl_optimizer_register_params(DL_Optimizer *opt, Tensor **params, size_t n_params) {
    if (!opt || !params) return;

    opt->parameters = params;
    opt->n_parameters = n_params;

    /* Allocate state for each parameter */
    opt->states = (DL_OptimizerState *)calloc(n_params, sizeof(DL_OptimizerState));
    for (size_t i = 0; i < n_params; i++) {
        opt->states[i].exp_avg = NULL;
        opt->states[i].exp_avg_sq = NULL;
        opt->states[i].delta = NULL;
        opt->states[i].step = 0;
    }
    opt->n_states = n_params;
}

void dl_optimizer_zero_grad(DL_Optimizer *opt) {
    if (!opt) return;

    for (size_t i = 0; i < opt->n_parameters; i++) {
        Tensor *param = opt->parameters[i];
        if (param && param->grad) {
            tensor_fill((Tensor *)param->grad, 0.0);
        }
    }
}

void dl_optimizer_step(DL_Optimizer *opt, Tensor **gradients, size_t n_grads) {
    if (!opt) return;

    (void)n_grads;  /* Use opt->n_parameters instead */

    Tensor **grads = gradients ? gradients : NULL;

    for (size_t i = 0; i < opt->n_parameters; i++) {
        Tensor *param = opt->parameters[i];
        if (!param) continue;

        Tensor *grad = grads ? grads[i] : (Tensor *)param->grad;
        if (!grad) continue;

        DL_OptimizerState *state = &opt->states[i];

        switch (opt->type) {
            case DL_OPT_SGD: {
                /* Simple SGD: param -= lr * grad */
                Tensor *update = tensor_mul_scalar(tensor_clone(grad), opt->learning_rate);
                for (size_t j = 0; j < param->size; j++) {
                    param->data[j] -= update->data[j];
                }
                tensor_free(update);
                break;
            }

            case DL_OPT_MOMENTUM:
            case DL_OPT_NESTEROV: {
                /* Momentum: v = momentum * v - lr * grad */
                if (!state->exp_avg) {
                    state->exp_avg = tensor_zeros(param->ndim, param->shape, param->device, false);
                }

                Tensor *grad_scaled = tensor_mul_scalar(grad, opt->learning_rate);
                Tensor *momentum_term = tensor_mul_scalar(state->exp_avg, opt->momentum);

                if (opt->type == DL_OPT_NESTEROV) {
                    /* Nesterov: v = momentum * v + grad - lr * grad */
                    Tensor *nesterov_update = tensor_add(momentum_term, grad);
                    tensor_free(state->exp_avg);
                    state->exp_avg = tensor_neg(nesterov_update);
                } else {
                    Tensor *update = tensor_sub(momentum_term, grad_scaled);
                    tensor_free(state->exp_avg);
                    state->exp_avg = tensor_neg(update);
                }

                /* Apply update to parameters */
                for (size_t j = 0; j < param->size; j++) {
                    param->data[j] += state->exp_avg->data[j];
                }

                tensor_free(grad_scaled);
                tensor_free(momentum_term);
                break;
            }

            case DL_OPT_ADAM:
            case DL_OPT_ADAMW: {
                /* Adam: m = beta1 * m + (1 - beta1) * grad */
                /* v = beta2 * v + (1 - beta2) * grad^2 */
                /* param -= lr * m / (sqrt(v) + eps) */

                state->step++;

                if (!state->exp_avg) {
                    state->exp_avg = tensor_zeros(param->ndim, param->shape, param->device, false);
                    state->exp_avg_sq = tensor_zeros(param->ndim, param->shape, param->device, false);
                }

                /* Update biased first moment estimate */
                for (size_t j = 0; j < param->size; j++) {
                    state->exp_avg->data[j] = opt->beta1 * state->exp_avg->data[j] +
                                             (1 - opt->beta1) * grad->data[j];
                }

                /* Update biased second raw moment estimate */
                for (size_t j = 0; j < param->size; j++) {
                    double grad_sq = grad->data[j] * grad->data[j];
                    state->exp_avg_sq->data[j] = opt->beta2 * state->exp_avg_sq->data[j] +
                                                 (1 - opt->beta2) * grad_sq;
                }

                /* Compute bias-corrected estimates */
                double beta1_pow_t = pow(opt->beta1, state->step);
                double beta2_pow_t = pow(opt->beta2, state->step);
                double m_hat_factor = 1.0 / (1.0 - beta1_pow_t);
                double v_hat_factor = 1.0 / (1.0 - beta2_pow_t);

                /* Compute update */
                for (size_t j = 0; j < param->size; j++) {
                    double m_hat = state->exp_avg->data[j] * m_hat_factor;
                    double v_hat = state->exp_avg_sq->data[j] * v_hat_factor;
                    double update = opt->learning_rate * m_hat / (sqrt(v_hat) + opt->eps);

                    if (opt->type == DL_OPT_ADAMW) {
                        /* Weight decay */
                        update += opt->weight_decay * opt->learning_rate * param->data[j];
                    }

                    param->data[j] -= update;
                }
                break;
            }

            case DL_OPT_RMSPROP: {
                /* RMSprop: v = decay * v + (1 - decay) * grad^2 */
                /* param -= lr * grad / sqrt(v + eps) */

                if (!state->exp_avg_sq) {
                    state->exp_avg_sq = tensor_zeros(param->ndim, param->shape, param->device, false);
                }

                /* Update squared gradient average */
                for (size_t j = 0; j < param->size; j++) {
                    double grad_sq = grad->data[j] * grad->data[j];
                    state->exp_avg_sq->data[j] = opt->rmsprop_decay * state->exp_avg_sq->data[j] +
                                                 (1 - opt->rmsprop_decay) * grad_sq;
                }

                /* Compute update */
                for (size_t j = 0; j < param->size; j++) {
                    double update = opt->learning_rate * grad->data[j] /
                                   (sqrt(state->exp_avg_sq->data[j]) + opt->eps);
                    param->data[j] -= update;
                }
                break;
            }

            case DL_OPT_ADAGRAD: {
                /* AdaGrad: v += grad^2 */
                /* param -= lr * grad / sqrt(v + eps) */

                if (!state->exp_avg_sq) {
                    state->exp_avg_sq = tensor_zeros(param->ndim, param->shape, param->device, false);
                }

                /* Accumulate squared gradients */
                for (size_t j = 0; j < param->size; j++) {
                    double grad_sq = grad->data[j] * grad->data[j];
                    state->exp_avg_sq->data[j] += grad_sq;
                }

                /* Compute update */
                for (size_t j = 0; j < param->size; j++) {
                    double update = opt->learning_rate * grad->data[j] /
                                   (sqrt(state->exp_avg_sq->data[j]) + opt->eps);
                    param->data[j] -= update;
                }
                break;
            }
        }

        /* Apply weight decay if specified */
        if (opt->weight_decay > 0 && opt->type != DL_OPT_ADAMW) {
            for (size_t j = 0; j < param->size; j++) {
                param->data[j] -= opt->weight_decay * opt->learning_rate * param->data[j];
            }
        }
    }
}

void dl_optimizer_free(DL_Optimizer *opt) {
    if (!opt) return;

    if (opt->states) {
        for (size_t i = 0; i < opt->n_states; i++) {
            if (opt->states[i].exp_avg) tensor_free(opt->states[i].exp_avg);
            if (opt->states[i].exp_avg_sq) tensor_free(opt->states[i].exp_avg_sq);
            if (opt->states[i].delta) tensor_free(opt->states[i].delta);
        }
        free(opt->states);
    }

    free(opt);
}

/* ============================================================================
 * SPECIFIC OPTIMIZERS
 * ============================================================================ */

DL_SGDOptimizer *dl_sgd_create(double learning_rate) {
    return (DL_SGDOptimizer *)dl_optimizer_create(DL_OPT_SGD, learning_rate);
}

DL_AdamOptimizer *dl_adam_create(double learning_rate, double beta1, double beta2, double eps) {
    DL_AdamOptimizer *adam = (DL_AdamOptimizer *)dl_optimizer_create(DL_OPT_ADAM, learning_rate);
    if (adam) {
        adam->beta1 = beta1;
        adam->beta2 = beta2;
        adam->eps = eps;
        adam->t = 0;
    }
    return adam;
}

DL_AdamWOptimizer *dl_adamw_create(double learning_rate, double weight_decay,
                                    double beta1, double beta2, double eps) {
    DL_AdamWOptimizer *adamw = (DL_AdamWOptimizer *)dl_optimizer_create(DL_OPT_ADAMW, learning_rate);
    if (adamw) {
        adamw->weight_decay = weight_decay;
        adamw->beta1 = beta1;
        adamw->beta2 = beta2;
        adamw->eps = eps;
        adamw->t = 0;
    }
    return adamw;
}

DL_MomentumOptimizer *dl_momentum_create(double learning_rate, double momentum) {
    DL_MomentumOptimizer *mom = (DL_MomentumOptimizer *)dl_optimizer_create(DL_OPT_MOMENTUM, learning_rate);
    if (mom) {
        mom->momentum = momentum;
    }
    return mom;
}

DL_RMSpropOptimizer *dl_rmsprop_create(double learning_rate, double decay, double eps) {
    DL_RMSpropOptimizer *rms = (DL_RMSpropOptimizer *)dl_optimizer_create(DL_OPT_RMSPROP, learning_rate);
    if (rms) {
        rms->decay = decay;
        rms->eps = eps;
    }
    return rms;
}

/* ============================================================================
 * LEARNING RATE SCHEDULERS
 * ============================================================================ */

DL_LRScheduler *dl_lr_scheduler_create(DL_LRSchedulerType type, double initial_lr, double gamma) {
    DL_LRScheduler *scheduler = (DL_LRScheduler *)calloc(1, sizeof(DL_LRScheduler));
    if (!scheduler) return NULL;

    scheduler->type = type;
    scheduler->initial_lr = initial_lr;
    scheduler->current_lr = initial_lr;
    scheduler->gamma = gamma;
    scheduler->step_size = 0;
    scheduler->T_max = 0;
    scheduler->min_lr = 0;

    return scheduler;
}

void dl_lr_scheduler_step(DL_LRScheduler *scheduler) {
    if (!scheduler) return;

    switch (scheduler->type) {
        case DL_LR_CONSTANT:
            /* No change */
            break;

        case DL_LR_STEP:
            /* lr = lr * gamma every step_size iterations */
            if (scheduler->step_size > 0) {
                scheduler->current_lr *= scheduler->gamma;
            }
            break;

        case DL_LR_EXPONENTIAL:
            /* lr = lr * gamma */
            scheduler->current_lr *= scheduler->gamma;
            break;

        case DL_LR_COSINE:
            /* lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * t / T_max)) */
            if (scheduler->T_max > 0) {
                double t = scheduler->step_size;
                scheduler->current_lr = scheduler->min_lr +
                    0.5 * (scheduler->initial_lr - scheduler->min_lr) *
                    (1 + cos(M_PI * t / scheduler->T_max));
            }
            scheduler->step_size++;
            break;

        case DL_LR_PLATEAU:
            /* Reduce when metric stops improving (handled externally) */
            break;
    }
}

double dl_lr_scheduler_get_lr(const DL_LRScheduler *scheduler) {
    return scheduler ? scheduler->current_lr : 0;
}

void dl_lr_scheduler_free(DL_LRScheduler *scheduler) {
    free(scheduler);
}
