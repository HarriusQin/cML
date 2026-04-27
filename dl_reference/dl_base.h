/**
 * @file dl_base.h
 * @brief Deep Learning Library - Base Definitions
 *
 * Core type definitions and macros for the cDL deep learning library.
 * This is the foundational header that all other dl_* headers depend on.
 *
 * Design principles:
 * - Tensor-first: All computations are tensor operations
 * - Autograd: Dynamic computational graph with automatic differentiation
 * - Modular: Layers, optimizers, and losses are decoupled
 * - GPU-ready: Abstract backend design for CUDA/OpenCL support (future)
 * - C-style: Pure C implementation following cML conventions
 */

#ifndef __DL_BASE_H__
#define __DL_BASE_H__

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ============================================================================
 * DEVICE TYPES
 * ============================================================================ */

/**
 * @brief Computing device type
 */
typedef enum {
    DL_DEVICE_CPU = 0,   /**< CPU (default) */
    DL_DEVICE_CUDA,      /**< NVIDIA GPU (future) */
    DL_DEVICE_OPENCL      /**< OpenCL (future) */
} DL_Device_t;

/* ============================================================================
 * TENSOR DEFINITIONS
 * ============================================================================ */

/**
 * @brief N-dimensional array structure
 *
 * Represents a tensor (ndim array) of double precision values.
 * Uses row-major (C-style) memory layout.
 *
 * Memory layout for shape [B, C, H, W]:
 *   element at (b, c, h, w) -> data[b*C*H*W + c*H*W + h*W + w]
 *
 * For 2D matrix [rows, cols]:
 *   element at (i, j) -> data[i*cols + j]
 */
typedef struct {
    double *data;         /**< Data buffer (CPU or GPU memory) */
    size_t *shape;        /**< Dimensions [dim0, dim1, ...] */
    size_t *stride;       /**< Strides for each dimension */
    size_t ndim;          /**< Number of dimensions */
    size_t size;          /**< Total number of elements */
    DL_Device_t device;  /**< Device where data resides */
    bool requires_grad;   /**< Whether gradients needed */
    void *grad;           /**< Gradient tensor (NULL if no gradient) */
    void *ctx;            /**< Backend-specific context */
} Tensor;

/**
 * @brief Tensor node for computation graph
 *
 * Wraps a Tensor with autograd metadata for tracking operations.
 */
typedef struct TensorNode {
    Tensor *data;          /**< The tensor data */
    Tensor *grad;         /**< Gradient tensor */
    bool is_leaf;         /**< True if created by user (not by operation) */
    int version;          /**< Version counter for in-place ops */
    void *creator_ctx;    /**< Operation that created this tensor */
} TensorNode;

/* ============================================================================
 * COMPUTATION GRAPH NODE
 * ============================================================================ */

/**
 * @brief Operation types in computation graph
 */
typedef enum {
    /* Element-wise operations */
    OP_ADD,               /**< Element-wise addition */
    OP_SUB,               /**< Element-wise subtraction */
    OP_MUL,               /**< Element-wise multiplication */
    OP_DIV,               /**< Element-wise division */
    OP_POW,               /**< Element-wise power */
    OP_NEG,               /**< Negation */

    /* Matrix operations */
    OP_MATMUL,            /**< Matrix multiplication */
    OP_TRANSPOSE,         /**< Transpose */
    OP_RESHAPE,           /**< Reshape */

    /* Activation functions */
    OP_RELU,              /**< ReLU activation */
    OP_SIGMOID,           /**< Sigmoid activation */
    OP_TANH,              /**< Tanh activation */
    OP_GELU,              /**< GELU activation */
    OP_SOFTMAX,           /**< Softmax activation */
    OP_LOG_SOFTMAX,       /**< Log-softmax activation */

    /* Neural network layers */
    OP_DENSE,             /**< Dense (linear) layer */
    OP_CONV2D,            /**< 2D Convolution */
    OP_MAXPOOL2D,         /**< 2D Max Pooling */
    OP_AVGPOOL2D,         /**< 2D Average Pooling */
    OP_LAYER_NORM,        /**< Layer Normalization */
    OP_BATCH_NORM,        /**< Batch Normalization */
    OP_DROPOUT,           /**< Dropout */

    /* Attention */
    OP_SCALED_DOT_ATTENTION,  /**< Scaled dot-product attention */
    OP_MULTIHEAD_ATTENTION,    /**< Multi-head attention */

    /* RNN variants */
    OP_RNN_CELL,          /**< RNN cell */
    OP_LSTM_CELL,         /**< LSTM cell */
    OP_GRU_CELL,          /**< GRU cell */

    /* Loss functions */
    OP_MSE_LOSS,          /**< Mean squared error */
    OP_CROSS_ENTROPY_LOSS,/**< Cross-entropy loss */

    /* Other */
    OP_SUM,               /**< Sum all elements */
    OP_MEAN,              /**< Mean of elements */
    OP_CLONE,             /**< Clone tensor (copy) */
} OpType;

/**
 * @brief Operation node in computation graph
 *
 * Represents a single operation in the autograd computation graph.
 */
typedef struct OpNode {
    OpType type;              /**< Operation type */
    Tensor **inputs;          /**< Array of input tensors */
    size_t n_inputs;          /**< Number of inputs */
    Tensor *output;           /**< Output tensor */
    void *params;             /**< Operation-specific parameters */
    void (*backward)(struct OpNode *);  /**< Backward pass function */
    bool saved;               /**< Whether forward values saved for backward */
    void *saved_ctx;          /**< Saved values for backward pass */
} OpNode;

/* ============================================================================
 * LAYER DEFINITIONS
 * ============================================================================ */

/**
 * @brief Layer types
 */
typedef enum {
    LAYER_DENSE,           /**< Fully connected layer */
    LAYER_CONV2D,          /**< 2D Convolution layer */
    LAYER_MAXPOOL2D,       /**< 2D Max pooling layer */
    LAYER_AVGPOOL2D,       /**< 2D Average pooling layer */
    LAYER_LSTM,             /**< LSTM layer */
    LAYER_GRU,              /**< GRU layer */
    LAYER_EMBEDDING,        /**< Embedding layer */
    LAYER_LAYER_NORM,       /**< Layer normalization */
    LAYER_BATCH_NORM,       /**< Batch normalization */
    LAYER_DROPOUT,          /**< Dropout layer */
    LAYER_ATTENTION,        /**< Self-attention layer */
    LAYER_TRANSFORMER_ENCODER,  /**< Transformer encoder block */
    LAYER_TRANSFORMER_DECODER,  /**< Transformer decoder block */
    LAYER_LINEAR,           /**< Alias for LAYER_DENSE */
    LAYER_RELU,             /**< ReLU activation layer */
    LAYER_SIGMOID,          /**< Sigmoid activation layer */
    LAYER_TANH,             /**< Tanh activation layer */
    LAYER_SOFTMAX,          /**< Softmax activation layer */
} LayerType;

/**
 * @brief Forward declaration of Layer
 */
typedef struct Layer Layer;

/**
 * @brief Layer virtual function table
 */
typedef struct LayerVTable {
    Tensor *(*forward)(Layer *layer, Tensor *input);
    Tensor *(*backward)(Layer *layer, Tensor *grad_output);
    Tensor **(*parameters)(Layer *layer, size_t *n_params);
    void (*free_layer)(Layer *layer);
} LayerVTable;

/**
 * @brief Base layer structure
 *
 * All neural network layers inherit from this structure.
 */
typedef struct Layer {
    LayerVTable *vtable;      /**< Virtual function table */
    char *name;               /**< Layer name for identification */
    Tensor **params;          /**< Learnable parameters */
    size_t n_params;          /**< Number of parameters */
    void *state;              /**< Layer state (e.g., running stats for BN) */
    LayerType type;           /**< Layer type */
    bool trainable;           /**< Whether layer has learnable params */
    bool training;            /**< Whether in training mode */
} Layer;

/* ============================================================================
 * MODEL DEFINITIONS
 * ============================================================================ */

/**
 * @brief Model types
 */
typedef enum {
    MODEL_MLP,              /**< Multi-layer perceptron */
    MODEL_CNN,              /**< Convolutional neural network */
    MODEL_RNN,              /**< Recurrent neural network */
    MODEL_LSTM,             /**< LSTM network */
    MODEL_GRU,              /**< GRU network */
    MODEL_TRANSFORMER,       /**< Transformer */
    MODEL_GAN,              /**< Generative adversarial network */
    MODEL_AUTOENCODER,       /**< Autoencoder */
    MODEL_VAE,              /**< Variational autoencoder */
    MODEL_RESNET,           /**< Residual network */
    MODEL_VIT,              /**< Vision transformer */
    MODEL_CUSTOM            /**< Custom model */
} ModelType;

/**
 * @brief Model virtual function table
 */
typedef struct Model Model;

typedef struct ModelVTable {
    void (*forward)(Model *model, Tensor *input, Tensor *output);
    void (*backward)(Model *model, Tensor *grad_output);
    void (*train_mode)(Model *model);
    void (*eval_mode)(Model *model);
    Tensor **(*parameters)(Model *model, size_t *n_params);
    void (*save)(Model *model, const char *path);
    void (*load)(Model *model, const char *path);
    void (*free)(Model *model);
} ModelVTable;

/**
 * @brief Neural network model
 */
typedef struct Model {
    ModelVTable *vtable;    /**< Virtual function table */
    ModelType type;          /**< Model type */
    char *name;              /**< Model name */
    Layer **layers;          /**< Array of layers */
    size_t n_layers;         /**< Number of layers */
    bool training;            /**< Training mode flag */
} Model;

/* ============================================================================
 * OPTIMIZER DEFINITIONS
 * ============================================================================ */

/**
 * @brief Optimizer types
 */
typedef enum {
    OPT_SGD,                 /**< Stochastic gradient descent */
    OPT_MOMENTUM,            /**< SGD with momentum */
    OPT_NESTEROV,           /**< Nesterov accelerated gradient */
    OPT_ADAM,                /**< Adam optimizer */
    OPT_ADAMW,               /**< Adam with weight decay */
    OPT_RMSPROP,             /**< RMSprop */
    OPT_ADAGRAD              /**< Adagrad */
} OptimizerType;

/**
 * @brief Forward declaration of Optimizer
 */
typedef struct Optimizer Optimizer;

/**
 * @brief Optimizer virtual function table
 */
typedef struct OptimizerVTable {
    void (*step)(Optimizer *opt, Tensor **params, size_t n_params);
    void (*zero_grad)(Optimizer *opt, Tensor **params, size_t n_params);
    void (*set_lr)(Optimizer *opt, double lr);
    void (*free)(Optimizer *opt);
} OptimizerVTable;

/**
 * @brief Base optimizer structure
 */
typedef struct Optimizer {
    OptimizerVTable *vtable; /**< Virtual function table */
    OptimizerType type;       /**< Optimizer type */
    double learning_rate;     /**< Learning rate */
    double weight_decay;      /**< Weight decay (L2 regularization) */
} Optimizer;

/* ============================================================================
 * LOSS FUNCTION DEFINITIONS
 * ============================================================================ */

/**
 * @brief Loss function types
 */
typedef enum {
    LOSS_MSE,                /**< Mean squared error */
    LOSS_CROSS_ENTROPY,      /**< Cross-entropy loss */
    LOSS_BINARY_CROSS_ENTROPY, /**< Binary cross-entropy */
    LOSS_HINGE,              /**< Hinge loss (SVM) */
    LOSS_SMOOTH_L1,          /**< Smooth L1 (Huber) loss */
    LOSS_FOCAL,              /**< Focal loss */
    LOSS_CONTRASTIVE,        /**< Contrastive loss (SimCLR) */
    LOSS_MARGIN,             /**< Margin ranking loss */
    LOSS_WASSERSTEIN         /**< Wasserstein / EMD loss */
} LossType;

/**
 * @brief Loss function result structure
 */
typedef struct {
    double loss;             /**< Loss value */
    Tensor *grad;            /**< Gradient w.r.t. predictions */
} LossResult;

/**
 * @brief Loss function virtual table
 */
typedef struct LossFunction {
    LossResult (*forward)(struct LossFunction *loss, Tensor *pred, Tensor *target);
    LossType type;           /**< Loss type */
    void *params;            /**< Loss-specific parameters */
} LossFunction;

/* ============================================================================
 * AUTOGRAD CONTEXT
 * ============================================================================ */

/**
 * @brief Global autograd context
 *
 * Controls whether gradient computation is enabled.
 */
typedef struct {
    bool enabled;            /**< Whether autograd is enabled */
    bool in_grad_scope;      /**< Whether inside grad() scope */
    int version_counter;     /**< Version counter for in-place ops */
} DLContext;

extern DLContext dl_ctx;

/**
 * @brief Enable gradient computation
 */
static inline void dl_enable_grad(void) {
    dl_ctx.enabled = true;
}

/**
 * @brief Disable gradient computation
 */
static inline void dl_disable_grad(void) {
    dl_ctx.enabled = false;
}

/**
 * @brief Check if gradient computation is enabled
 */
static inline bool dl_grad_enabled(void) {
    return dl_ctx.enabled;
}

/* ============================================================================
 * MEMORY MANAGEMENT MACROS
 * ============================================================================ */

/**
 * @brief Allocate and zero-initialize
 */
#define DL_CALLOC(type, n) (type *)calloc(n, sizeof(type))

/**
 * @brief Allocate
 */
#define DL_MALLOC(type, n) (type *)malloc((n) * sizeof(type))

/**
 * @brief Reallocate
 */
#define DL_REALLOC(type, ptr, n) (type *)realloc(ptr, (n) * sizeof(type))

/* ============================================================================
 * TENSOR SHAPE MACROS
 * ============================================================================ */

/**
 * @brief Get tensor dimension safely
 */
#define TENSOR_DIM(t, i) (((i) < (t)->ndim) ? (t)->shape[i] : 1)

/**
 * @brief Get total size from shape array
 */
#define SHAPE_SIZE(shape, ndim) \
    ({ size_t _s = 1; for (size_t _i = 0; _i < (ndim); _i++) _s *= (shape)[_i]; _s; })

/* ============================================================================
 * ERROR HANDLING
 * ============================================================================ */

#define DL_ERROR(msg, ...) fprintf(stderr, "[DL ERROR] " msg "\n", ##__VA_ARGS__)
#define DL_WARN(msg, ...) fprintf(stderr, "[DL WARN] " msg "\n", ##__VA_ARGS__)
#define DL_INFO(msg, ...) fprintf(stdout, "[DL INFO] " msg "\n", ##__VA_ARGS__)

/* ============================================================================
 * MATH CONSTANTS
 * ============================================================================ */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_E
#define M_E 2.71828182845904523536
#endif

#define DL_EPS 1e-8
#define DL_FLOAT_MAX 1e10
#define DL_FLOAT_MIN -1e10

/* ============================================================================
 * VERSION INFO
 * ============================================================================ */

#define DL_VERSION_MAJOR 0
#define DL_VERSION_MINOR 1
#define DL_VERSION_PATCH 0

#endif /* __DL_BASE_H__ */
