/**
 * @file dl_model.h
 * @brief Neural network models
 *
 * Provides common neural network models:
 * - MLP (Multi-Layer Perceptron)
 * - CNN (Convolutional Neural Network)
 * - Transformer (Encoder/Decoder)
 * - LSTM/GRU (Recurrent)
 */

#ifndef __DL_MODEL_H__
#define __DL_MODEL_H__

#include "dl_base.h"
#include "dl_tensor.h"
#include "dl_layers.h"
#include "dl_loss.h"
#include "dl_optimizer.h"

/* ============================================================================
 * MODEL TYPES
 * ============================================================================ */

typedef enum {
    DL_MODEL_MLP,           /**< Multi-Layer Perceptron */
    DL_MODEL_CNN,           /**< Convolutional Neural Network */
    DL_MODEL_TRANSFORMER,   /**< Transformer */
    DL_MODEL_LSTM,         /**< LSTM */
    DL_MODEL_GRU,           /**< GRU */
    DL_MODEL_RESNET,        /**< ResNet */
    DL_MODEL_CUSTOM         /**< Custom model */
} DL_ModelType;

/* ============================================================================
 * MODEL BASE
 * ============================================================================ */

/**
 * @brief Forward pass function type
 */
typedef Tensor *(*DL_ForwardFn)(void *model, const Tensor *input);

/**
 * @brief Backward pass function type
 */
typedef void (*DL_BackwardFn)(void *model, const Tensor *grad_output);

/**
 * @brief Model configuration
 */
typedef struct {
    DL_ModelType type;
    const char *name;
    void *params;
} DL_ModelConfig;

/**
 * @brief Model state
 */
typedef struct {
    bool training;
    void *layers;
    void *optim_state;
} DL_ModelState;

/* ============================================================================
 * MLP (MULTI-LAYER PERCEPTRON)
 * ============================================================================ */

/**
 * @brief MLP activation type
 */
typedef enum {
    DL_ACTIVATION_RELU,
    DL_ACTIVATION_SIGMOID,
    DL_ACTIVATION_TANH,
    DL_ACTIVATION_GELU
} DL_ActivationType;

/**
 * @brief MLP layer configuration
 */
typedef struct {
    size_t input_size;
    size_t *hidden_sizes;
    size_t n_hidden;
    size_t output_size;
    bool use_bias;
    DL_ActivationType activation;
} DL_MLPConfig;

/**
 * @brief MLP model
 */
typedef struct {
    DL_ModelType type;
    const char *name;
    bool training;

    DenseParams **layers;
    size_t n_layers;

    DL_ActivationType activation;
    bool use_dropout;
    double dropout_p;

    size_t input_size;
    size_t output_size;
} DL_MLP;

/**
 * @brief Create an MLP model
 */
DL_MLP *dl_mlp_create(const DL_MLPConfig *config);

/**
 * @brief MLP forward pass
 */
Tensor *dl_mlp_forward(DL_MLP *mlp, const Tensor *input);

/**
 * @brief MLP backward pass
 */
void dl_mlp_backward(DL_MLP *mlp, const Tensor *grad_output);

/**
 * @brief Get MLP parameters
 */
Tensor **dl_mlp_parameters(DL_MLP *mlp, size_t *n_params);

/**
 * @brief Free MLP model
 */
void dl_mlp_free(DL_MLP *mlp);

/* ============================================================================
 * CNN (CONVOLUTIONAL NEURAL NETWORK)
 * ============================================================================ */

/**
 * @brief CNN layer configuration
 */
typedef struct {
    size_t in_channels;
    size_t out_channels;
    size_t kernel_size;
    size_t stride;
    size_t padding;
    bool use_bias;
} DL_Conv2DConfig;

/**
 * @brief Simple CNN model
 */
typedef struct {
    DL_ModelType type;
    const char *name;
    bool training;

    Conv2DParams **conv_layers;
    size_t n_conv_layers;

    DenseParams **fc_layers;
    size_t n_fc_layers;

    MaxPool2DParams **pool_layers;
    size_t n_pool_layers;

    size_t num_classes;
} DL_CNN;

/**
 * @brief Create a CNN model
 */
DL_CNN *dl_cnn_create(size_t in_channels, size_t num_classes);

/**
 * @brief CNN forward pass
 */
Tensor *dl_cnn_forward(DL_CNN *cnn, const Tensor *input);

/**
 * @brief CNN backward pass
 */
void dl_cnn_backward(DL_CNN *cnn, const Tensor *grad_output);

/**
 * @brief Get CNN parameters
 */
Tensor **dl_cnn_parameters(DL_CNN *cnn, size_t *n_params);

/**
 * @brief Free CNN model
 */
void dl_cnn_free(DL_CNN *cnn);

/* ============================================================================
 * TRANSFORMER
 * ============================================================================ */

/**
 * @brief Transformer encoder layer
 */
typedef struct {
    /* Multi-head self-attention */
    DenseParams *wq;   /**< Query projection */
    DenseParams *wk;   /**< Key projection */
    DenseParams *wv;   /**< Value projection */
    DenseParams *wo;   /**< Output projection */

    /* Feed-forward network */
    DenseParams *ff_w1;
    DenseParams *ff_b1;
    DenseParams *ff_w2;
    DenseParams *ff_b2;

    /* Layer norms */
    LayerNormParams *ln1;
    LayerNormParams *ln2;

    size_t d_model;
    size_t n_heads;
    size_t d_ff;
    double dropout_p;
} DL_TransformerEncoderLayer;

/**
 * @brief Transformer decoder layer
 */
typedef struct {
    /* Self-attention */
    DenseParams *wq;
    DenseParams *wk;
    DenseParams *wv;
    DenseParams *wo;

    /* Cross-attention */
    DenseParams *q2;
    DenseParams *k2;
    DenseParams *v2;
    DenseParams *wo2;

    /* Feed-forward */
    DenseParams *ff_w1;
    DenseParams *ff_b1;
    DenseParams *ff_w2;
    DenseParams *ff_b2;

    LayerNormParams *ln1;
    LayerNormParams *ln2;
    LayerNormParams *ln3;

    size_t d_model;
    size_t n_heads;
    size_t d_ff;
} DL_TransformerDecoderLayer;

/**
 * @brief Transformer model
 */
typedef struct {
    DL_ModelType type;
    const char *name;
    bool training;

    /* Embeddings */
    Tensor *token_embedding;  /**< [vocab_size, d_model] */
    Tensor *pos_embedding;    /**< [max_seq_len, d_model] */

    /* Encoder layers */
    DL_TransformerEncoderLayer *enc_layers;
    size_t n_encoder_layers;

    /* Decoder layers */
    DL_TransformerDecoderLayer *dec_layers;
    size_t n_decoder_layers;

    /* Output projection */
    DenseParams *output_projection;

    LayerNormParams *final_ln;

    size_t vocab_size;
    size_t d_model;
    size_t n_heads;
    size_t d_ff;
    size_t max_seq_len;
    double dropout_p;
} DL_Transformer;

/**
 * @brief Create a Transformer model
 */
DL_Transformer *dl_transformer_create(size_t vocab_size, size_t d_model,
                                      size_t n_heads, size_t n_encoder_layers,
                                      size_t n_decoder_layers, size_t d_ff,
                                      size_t max_seq_len, double dropout_p);

/**
 * @brief Transformer forward pass
 */
Tensor *dl_transformer_forward(DL_Transformer *transformer, const Tensor *input,
                               const Tensor *target, bool training);

/**
 * @brief Get Transformer parameters
 */
Tensor **dl_transformer_parameters(DL_Transformer *tr, size_t *n_params);

/**
 * @brief Free Transformer model
 */
void dl_transformer_free(DL_Transformer *transformer);

/* ============================================================================
 * TRAINING UTILITIES
 * ============================================================================ */

/**
 * @brief Training step result
 */
typedef struct {
    double loss;
    double accuracy;
} DL_TrainingResult;

/**
 * @brief Train a model for one epoch
 */
DL_TrainingResult dl_train_epoch(DL_MLP *model, const Tensor *X, const Tensor *y,
                                DL_Optimizer *optimizer, double (*loss_fn)(Tensor *, Tensor *));

/**
 * @brief Evaluate model on test data
 */
DL_TrainingResult dl_evaluate(DL_MLP *model, const Tensor *X, const Tensor *y,
                              double (*loss_fn)(Tensor *, Tensor *));

/**
 * @brief Predict using model
 */
Tensor *dl_predict(DL_MLP *model, const Tensor *input);

#endif /* __DL_MODEL_H__ */
