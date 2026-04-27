/**
 * @file dl_layers.h
 * @brief Neural network layer abstractions and implementations
 *
 * Provides common neural network layers:
 * - Dense (fully connected)
 * - Activation functions (ReLU, Sigmoid, Tanh, GELU, Softmax)
 * - Dropout
 * - Layer normalization
 * - Batch normalization
 */

#ifndef __DL_LAYERS_H__
#define __DL_LAYERS_H__

#include "dl_base.h"
#include "dl_tensor.h"

/* ============================================================================
 * ACTIVATION FUNCTIONS
 * ============================================================================ */

/**
 * @brief ReLU activation: f(x) = max(0, x)
 */
Tensor *tensor_relu(const Tensor *x);

/**
 * @brief ReLU backward pass
 */
Tensor *tensor_relu_backward(const Tensor *x, const Tensor *grad_output);

/**
 * @brief Sigmoid activation: f(x) = 1 / (1 + exp(-x))
 */
Tensor *tensor_sigmoid(const Tensor *x);

/**
 * @brief Sigmoid backward pass
 */
Tensor *tensor_sigmoid_backward(const Tensor *x, const Tensor *grad_output);

/**
 * @brief Tanh activation: f(x) = tanh(x)
 */
Tensor *tensor_tanh_activation(const Tensor *x);

/**
 * @brief Tanh backward pass
 */
Tensor *tensor_tanh_backward(const Tensor *x, const Tensor *grad_output);

/**
 * @brief GELU activation: f(x) = x * Phi(x) where Phi is CDF of N(0,1)
 *        Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
Tensor *tensor_gelu(const Tensor *x);

/**
 * @brief GELU backward pass
 */
Tensor *tensor_gelu_backward(const Tensor *x, const Tensor *grad_output);

/**
 * @brief Softmax: f(x_i) = exp(x_i) / sum(exp(x_j))
 *
 * @param x Input tensor [*, n_classes]
 * @param axis Axis along which to compute softmax
 * @return Softmax probabilities
 */
Tensor *tensor_softmax(const Tensor *x, int axis);

/**
 * @brief Softmax backward pass
 */
Tensor *tensor_softmax_backward(const Tensor *x, const Tensor *grad_output, int axis);

/**
 * @brief Log softmax: f(x_i) = log(softmax(x)_i)
 */
Tensor *tensor_log_softmax(const Tensor *x, int axis);

/* ============================================================================
 * DROPOUT
 * ============================================================================ */

/**
 * @brief Dropout forward pass
 *
 * @param x Input tensor
 * @param p Dropout probability
 * @param training Whether in training mode
 * @param mask Output mask (can be NULL to ignore)
 * @return Dropped out tensor (same shape as input)
 */
Tensor *tensor_dropout(const Tensor *x, double p, bool training, Tensor **mask);

/**
 * @brief Dropout backward pass
 */
Tensor *tensor_dropout_backward(const Tensor *grad_output, const Tensor *mask, double p);

/* ============================================================================
 * DENSE (FULLY CONNECTED) LAYER
 * ============================================================================ */

/**
 * @brief Dense layer parameters
 */
typedef struct {
    Tensor *weight;   /**< Weight matrix [out_features, in_features] */
    Tensor *bias;     /**< Bias vector [out_features] (NULL if no bias) */
} DenseParams;

/**
 * @brief Create a dense (fully connected) layer
 *
 * @param in_features Input feature size
 * @param out_features Output feature size
 * @param use_bias Whether to use bias term
 * @param requires_grad Whether weights require gradients
 * @return Layer parameters
 */
DenseParams *dense_create(size_t in_features, size_t out_features,
                         bool use_bias, bool requires_grad);

/**
 * @brief Dense layer forward pass
 *
 * @param params Layer parameters
 * @param input Input tensor [*, in_features]
 * @return Output tensor [*, out_features]
 */
Tensor *dense_forward(const DenseParams *params, const Tensor *input);

/**
 * @brief Dense layer backward pass
 *
 * @param params Layer parameters
 * @param input Input tensor
 * @param grad_output Gradient w.r.t. output
 * @param grad_input Output gradient w.r.t. input
 */
void dense_backward(DenseParams *params, const Tensor *input,
                   const Tensor *grad_output, Tensor *grad_input);

/**
 * @brief Free dense layer parameters
 */
void dense_free(DenseParams *params);

/**
 * @brief Initialize dense weights with Xavier/Glorot initialization
 */
void dense_init_xavier(DenseParams *params);

/**
 * @brief Initialize dense weights with Kaiming/He initialization
 */
void dense_init_kaiming(DenseParams *params);

/* ============================================================================
 * LAYER NORMALIZATION
 * ============================================================================ */

/**
 * @brief Layer normalization parameters
 */
typedef struct {
    Tensor *gamma;  /**< Scale parameter */
    Tensor *beta;   /**< Shift parameter */
    double eps;      /**< Epsilon for numerical stability */
} LayerNormParams;

/**
 * @brief Create layer normalization parameters
 */
LayerNormParams *layer_norm_create(size_t normalized_shape, double eps, bool requires_grad);

/**
 * @brief Layer normalization forward
 */
Tensor *layer_norm_forward(const LayerNormParams *params, const Tensor *x);

/**
 * @brief Layer normalization backward
 */
void layer_norm_backward(LayerNormParams *params, const Tensor *x,
                        const Tensor *grad_output, Tensor *grad_input);

/**
 * @brief Free layer normalization parameters
 */
void layer_norm_free(LayerNormParams *params);

/* ============================================================================
 * BATCH NORMALIZATION
 * ============================================================================ */

/**
 * @brief Batch normalization parameters
 */
typedef struct {
    Tensor *gamma;       /**< Scale parameter */
    Tensor *beta;        /**< Shift parameter */
    Tensor *running_mean; /**< Running mean */
    Tensor *running_var;  /**< Running variance */
    double momentum;       /**< Momentum for running statistics */
    double eps;           /**< Epsilon */
} BatchNormParams;

/**
 * @brief Create batch normalization parameters
 */
BatchNormParams *batch_norm_create(size_t num_features, double momentum, double eps,
                                   bool requires_grad);

/**
 * @brief Batch normalization forward (training mode)
 */
Tensor *batch_norm_forward_train(BatchNormParams *params, const Tensor *input,
                                 bool training);

/**
 * @brief Batch normalization forward (inference mode)
 */
Tensor *batch_norm_forward_eval(const BatchNormParams *params, const Tensor *input);

/**
 * @brief Batch normalization backward
 */
void batch_norm_backward(BatchNormParams *params, const Tensor *input,
                        const Tensor *grad_output, Tensor *grad_input);

/**
 * @brief Free batch normalization parameters
 */
void batch_norm_free(BatchNormParams *params);

/* ============================================================================
 * CONV2D LAYER (2D Convolution)
 * ============================================================================ */

/**
 * @brief Conv2D parameters
 */
typedef struct {
    Tensor *weight;           /**< Filter weights [out_channels, in_channels, kh, kw] */
    Tensor *bias;             /**< Bias [out_channels] (NULL if no bias) */
    size_t in_channels;
    size_t out_channels;
    size_t kernel_h, kernel_w;
    size_t stride_h, stride_w;
    size_t padding_h, padding_w;
} Conv2DParams;

/**
 * @brief Create Conv2D parameters
 */
Conv2DParams *conv2d_create(size_t in_channels, size_t out_channels,
                            size_t kernel_h, size_t kernel_w,
                            size_t stride_h, size_t stride_w,
                            size_t padding_h, size_t padding_w,
                            bool use_bias, bool requires_grad);

/**
 * @brief Conv2D forward pass
 *
 * Input: [batch, in_channels, height, width]
 * Output: [batch, out_channels, out_height, out_width]
 */
Tensor *conv2d_forward(const Conv2DParams *params, const Tensor *input);

/**
 * @brief Conv2D backward pass
 */
void conv2d_backward(Conv2DParams *params, const Tensor *input,
                    const Tensor *grad_output, Tensor *grad_input);

/**
 * @brief Free Conv2D parameters
 */
void conv2d_free(Conv2DParams *params);

/* ============================================================================
 * POOLING LAYERS
 * ============================================================================ */

/**
 * @brief Max pooling parameters (no learnable params)
 */
typedef struct {
    size_t kernel_h, kernel_w;
    size_t stride_h, stride_w;
} MaxPool2DParams;

/**
 * @brief Create max pooling parameters
 */
MaxPool2DParams *maxpool2d_create(size_t kernel_h, size_t kernel_w,
                                  size_t stride_h, size_t stride_w);

/**
 * @brief Max pooling forward
 */
Tensor *maxpool2d_forward(const MaxPool2DParams *params, const Tensor *input);

/**
 * @brief Max pooling backward
 */
Tensor *maxpool2d_backward(const MaxPool2DParams *params, const Tensor *input,
                           const Tensor *grad_output);

/**
 * @brief Average pooling parameters
 */
typedef struct {
    size_t kernel_h, kernel_w;
    size_t stride_h, stride_w;
} AvgPool2DParams;

/**
 * @brief Create average pooling parameters
 */
AvgPool2DParams *avgpool2d_create(size_t kernel_h, size_t kernel_w,
                                 size_t stride_h, size_t stride_w);

/**
 * @brief Average pooling forward
 */
Tensor *avgpool2d_forward(const AvgPool2DParams *params, const Tensor *input);

/**
 * @brief Average pooling backward
 */
Tensor *avgpool2d_backward(const AvgPool2DParams *params, const Tensor *input,
                          const Tensor *grad_output);

/**
 * @brief Free pooling parameters
 */
void maxpool2d_free(MaxPool2DParams *params);
void avgpool2d_free(AvgPool2DParams *params);

#endif /* __DL_LAYERS_H__ */
