# API Reference

Complete API reference for the cML machine learning library.

## Table of Contents

1. [Data Structures](#data-structures)
2. [Common Utilities](#common-utilities)
3. [Tensor Operations](#tensor-operations)
4. [OpenCL Tensor](#opencl-tensor)
5. [Linear Models](#linear-models)
6. [Ensemble Methods](#ensemble-methods)
7. [Deep Learning](#deep-learning)

---

## Data Structures

### TensorDType

```c
typedef enum {
    TENSOR_DTYPE_F32,        // float32
    TENSOR_DTYPE_F16,         // float16
    TENSOR_DTYPE_INT8,        // int8_t
    TENSOR_DTYPE_INT32,       // int32_t
    TENSOR_DTYPE_UINT8,       // uint8_t
} TensorDType;
```

### TensorLayout

```c
typedef enum {
    TENSOR_LAYOUT_NCHW,       // Batch, Channel, Height, Width
    TENSOR_LAYOUT_NHWC,       // Batch, Height, Width, Channel
    TENSOR_LAYOUT_CHWN,       // Channel, Height, Width, Batch
    TENSOR_LAYOUT_ANY          // Strides provided
} TensorLayout;
```

### Tensor

```c
typedef struct {
    void* data;               // Raw data buffer
    size_t size;             // Total number of elements
    TensorDType dtype;       // Data type
    TensorLayout layout;     // Memory layout
    uint8_t ndim;            // Number of dimensions
    size_t shape[4];        // Max 4 dimensions
    size_t strides[4];        // Stride for each dimension
    bool owner;             // Whether we own the data
} Tensor;
```

### Conv2DParams

```c
typedef struct {
    size_t stride_h, stride_w;
    size_t pad_h, pad_w;
    size_t dilation_h, dilation_w;
} Conv2DParams;
```

---

## Common Utilities

### FeatureSample

```c
typedef struct {
    size_t idx;  // Sample index
    double val;  // Feature value
} FeatureSample;
```

### ML_Model_t

```c
typedef struct ML_Model_t {
    const char* name;
    const ML_Model_Config_t config;
    ML_Weights_t state;
    ML_MethodTable methods;
} ML_Model_t;

typedef struct ML_MethodTable {
    int (*fit)(const ML_Model_Config_t*, ML_Weights_t*, ...);
    int (*predict)(const ML_Weights_t*, ...);
    int (*predict_proba)(const ML_Weights_t*, ...);
    void (*free)(ML_Weights_t*);
} ML_MethodTable;
```

### ML_ScalingParams

```c
typedef enum {
    SCALING_STANDARD,    // (x - mean) / std
    SCALING_MINMAX,      // (x - min) / (max - min)
    SCALING_MAXABS,      // x / max(|x|)
    SCALING_ROBUST       // (x - median) / IQR
} ScalingType;
```

### Dataset Functions

```c
// Create dataset from column-oriented data
dataset* dataset_create(size_t n_samples, size_t n_features,
                        double** features, double* labels);

// Free dataset
void dataset_free(dataset* ds);

// Split indices into train/test
int train_test_split(const size_t* indices, size_t n_samples,
                      size_t** train_out, size_t** test_out,
                      size_t* train_size_out, size_t* test_size_out,
                      double test_ratio, unsigned int seed);

// Feature scaling
ML_ScalingParams* ml_fit_scaling(const dataset* ds, const size_t* feat_idx,
                                  size_t n_features, const size_t* sample_idx,
                                  size_t n_samples, ScalingType type);

dataset* ml_transform_features(const ML_ScalingParams* scaler,
                               const dataset* ds, const size_t* feat_idx,
                               size_t n_features, const size_t* sample_idx,
                               size_t n_samples);
```

---

## Tensor Operations

### Tensor Creation

```c
// Create tensor with shape
Tensor* tensor_create(TensorDType dtype, TensorLayout layout,
                      size_t* shape, uint8_t ndim);

// Create from existing data (no copy)
Tensor* tensor_wrap(void* data, TensorDType dtype, TensorLayout layout,
                    size_t* shape, uint8_t ndim);

// Clone tensor
Tensor* tensor_clone(const Tensor* t);

// Free tensor
void tensor_free(Tensor* t);
```

### Fill Operations

```c
void tensor_fill_f32(Tensor* t, float val);
void tensor_fill_randn(Tensor* t, float mean, float std_dev);
void tensor_fill_xavier(Tensor* t);
```

### Element-wise Operations

```c
Tensor* tensor_add(const Tensor* a, const Tensor* b);
Tensor* tensor_sub(const Tensor* a, const Tensor* b);
Tensor* tensor_mul(const Tensor* a, const Tensor* b);
Tensor* tensor_div(const Tensor* a, const Tensor* b);
void tensor_scale(Tensor* t, float scalar);
void tensor_relu(Tensor* t);
void tensor_sigmoid(Tensor* t);
void tensor_tanh(Tensor* t);
void tensor_softmax(Tensor* t, size_t axis);
```

### Reduction Operations

```c
Tensor* tensor_sum(const Tensor* t, size_t axis);
Tensor* tensor_mean(const Tensor* t, size_t axis);
Tensor* tensor_max(const Tensor* t, size_t axis);
```

### Matrix Operations

```c
Tensor* tensor_matmul(const Tensor* a, const Tensor* b);
void tensor_gemm(float* C, const float* A, const float* B,
                size_t M, size_t N, size_t K,
                float alpha, float beta);
```

### Shape Operations

```c
Tensor* tensor_reshape(const Tensor* t, size_t* new_shape, uint8_t ndim);
Tensor* tensor_transpose(const Tensor* t, uint8_t axis0, uint8_t axis1);
Tensor* tensor_slice(const Tensor* t, size_t dim, size_t start, size_t end);
```

### Pooling and Convolution

```c
Tensor* tensor_maxpool2d(const Tensor* input, size_t pool_h, size_t pool_w,
                        size_t stride_h, size_t stride_w);

Tensor* tensor_avgpool2d(const Tensor* input, size_t pool_h, size_t pool_w,
                        size_t stride_h, size_t stride_w);

Tensor* tensor_conv2d(const Tensor* input, const Tensor* weight,
                     const Conv2DParams* params);
```

### Quantization

```c
Tensor* tensor_quantize_affine(const Tensor* t, TensorDType dtype);
Tensor* tensor_dequantize(const Tensor* t);
```

### Accessing Elements

```c
float tensor_get_f32(const Tensor* t, size_t* indices);
void tensor_set_f32(Tensor* t, size_t* indices, float val);
```

---

## OpenCL Tensor

### CLTensorDType

```c
typedef enum {
    CL_TENSOR_DTYPE_F32,
    CL_TENSOR_DTYPE_F16,
    CL_TENSOR_DTYPE_INT8,
    CL_TENSOR_DTYPE_INT32,
} CLTensorDType;
```

### CLTensorLayout

```c
typedef enum {
    CL_TENSOR_LAYOUT_NCHW,
    CL_TENSOR_LAYOUT_NHWC,
    CL_TENSOR_LAYOUT_CHWN,
} CLTensorLayout;
```

### CLTensor

```c
typedef struct {
    cl_mem buffer;              // OpenCL memory buffer
    cl_context context;         // OpenCL context
    cl_command_queue queue;     // Command queue
    CLTensorLayout layout;
    CLTensorDType dtype;
    uint8_t ndim;
    size_t* shape;
    size_t* strides;
    size_t size;
    size_t nbytes;
    cl_uint dev_id;
} CLTensor;
```

### CLOpenCL

```c
typedef struct {
    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
    cl_uint dev_count;
    cl_platform_id platform;
    cl_kernel_cache_t kernel_cache;  // Pre-compiled kernels
} CLOpenCL;
```

### Context Functions

```c
bool cl_init(CLOpenCL* cl, cl_device_type device_type);
void cl_release(CLOpenCL* cl);
void cl_print_device_info(cl_device_id device);
```

### Tensor Functions

```c
CLTensor* cl_tensor_create(CLOpenCL* cl, CLTensorDType dtype,
                            CLTensorLayout layout,
                            size_t* shape, uint8_t ndim);

CLTensor* cl_tensor_create_from_host(CLOpenCL* cl, CLTensorDType dtype,
                                      CLTensorLayout layout,
                                      size_t* shape, uint8_t ndim,
                                      const void* host_data);

void cl_tensor_free(CLTensor* t);
void cl_tensor_upload(CLTensor* t, const void* host_data);
void cl_tensor_download(const CLTensor* t, void* host_data);
```

### GPU Operations

```c
CLTensor* cl_tensor_relu(CLOpenCL* cl, const CLTensor* input);
CLTensor* cl_tensor_matmul(CLOpenCL* cl, const CLTensor* a, const CLTensor* b);
void cl_tensor_fill_f32(CLOpenCL* cl, CLTensor* t, float val);
CLTensor* cl_tensor_softmax(CLOpenCL* cl, const CLTensor* t, size_t axis);
```

---

## Linear Models

### MLR (Multiple Linear Regression)

```c
typedef struct MLR_State {
    Matrix* weights;    // [n_features, 1]
    double* bias;
} MLR_State;

int mlr_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
            const dataset* ds, const size_t* feature_indices,
            size_t n_features, size_t target_index,
            const size_t* sample_indices, size_t n_samples);

int mlr_predict(const ML_Weights_t* state, const dataset* ds,
                const size_t* feature_indices, size_t n_features,
                const size_t* sample_indices, size_t n_samples,
                void* output);
```

### Ridge Regression

```c
typedef struct Ridge_State {
    Matrix* weights;
    double* bias;
    double alpha;  // Regularization strength
} Ridge_State;

int ridge_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
              const dataset* ds, const size_t* feature_indices,
              size_t n_features, size_t target_index,
              const size_t* sample_indices, size_t n_samples,
              double alpha);

int ridge_predict(const ML_Weights_t* state, const dataset* ds,
                  const size_t* feature_indices, size_t n_features,
                  const size_t* sample_indices, size_t n_samples,
                  void* output);
```

### Softmax Regression

```c
typedef struct Softmax_State {
    Matrix* weights;     // [n_classes, n_features]
    double* bias;       // [n_classes]
    double learning_rate;
    size_t n_iterations;
} Softmax_State;

int softmax_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
                const dataset* ds, const size_t* feature_indices,
                size_t n_features, size_t target_index,
                const size_t* sample_indices, size_t n_samples,
                double lr, size_t n_iter);

int softmax_predict(const ML_Weights_t* state, const dataset* ds,
                    const size_t* feature_indices, size_t n_features,
                    const size_t* sample_indices, size_t n_samples,
                    int* predictions);
```

### Polynomial Regression

```c
typedef struct PolyReg_State {
    Matrix* weights;    // [degree]
    double* bias;
    size_t degree;
} PolyReg_State;

int polynomial_features(const double* x, size_t n_samples,
                         size_t degree, Matrix* result);

int poly_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
              const double* x, const double* y,
              size_t n_samples, size_t degree);

int poly_predict(const ML_Weights_t* state,
                 const double* x, size_t n_samples,
                 double* output);
```

### Weighted Least Squares

```c
typedef struct WLS_State {
    Matrix* weights;
    double* bias;
} WLS_State;

int wls_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
            const dataset* ds, const size_t* feature_indices,
            size_t n_features, size_t target_index,
            const size_t* sample_indices, size_t n_samples,
            const double* weights);

int wls_predict(const ML_Weights_t* state, const dataset* ds,
                const size_t* feature_indices, size_t n_features,
                const size_t* sample_indices, size_t n_samples,
                void* output);
```

---

## Ensemble Methods

### Decision Tree

```c
typedef struct DecisionTreeNode {
    size_t feature_index;
    double threshold;
    bool is_leaf;
    double value;           // Leaf prediction
    double impurity;       // Gini/entropy at node
    struct DecisionTreeNode *left, *right;
} DecisionTreeNode;

typedef struct {
    DecisionTreeNode* root;
    size_t max_depth;
    size_t min_samples_split;
    size_t n_classes;
} DecisionTree;

DecisionTree* dt_create(size_t max_depth, size_t min_samples_split);
void dt_free(DecisionTree* tree);

int dt_fit(DecisionTree* tree, const dataset* ds,
           const size_t* feature_indices, size_t n_features,
           const size_t* sample_indices, size_t n_samples,
           size_t target_index);

int dt_predict(const DecisionTree* tree, const dataset* ds,
               const size_t* feature_indices, size_t n_features,
               const size_t* sample_indices, size_t n_samples,
               int* predictions);
```

### Random Forest

```c
typedef struct {
    int n_estimators;
    DecisionTree** trees;
    double* feature_importance;
} RandomForest_State;

typedef struct {
    int n_estimators;
    int max_depth;
    size_t min_samples_split;
    size_t n_features_subset;
} RandomForest_Config_t;

ML_Model_t create_random_forest_model(void);

int RandomForest_fit(const ML_Model_Config_t* config, RandomForest_Weights_t* state,
                      const dataset* ds, size_t* feat_idx, size_t n_features,
                      size_t label_idx, size_t* indices, size_t n_samples);

int RandomForest_predict(const RandomForest_Weights_t* state, const dataset* ds,
                          size_t* feat_idx, size_t n_features,
                          size_t* indices, size_t n_samples, int* predictions);
```

### AdaBoost

```c
typedef struct {
    DecisionTree** stumps;     // Weak learners (decision stumps)
    double* alphas;             // Weight of each stump
    int n_stumps;
} AdaBoost_State;

typedef struct {
    int n_estimators;
    double learning_rate;
} AdaBoost_Config_t;

ML_Model_t create_adaboost_model(void);

int AdaBoost_fit(const ML_Model_Config_t* config, AdaBoost_State* state,
                 const dataset* ds, size_t* feat_idx, size_t n_features,
                 size_t label_idx, size_t* indices, size_t n_samples);

int AdaBoost_predict(const AdaBoost_State* state, const dataset* ds,
                     size_t* feat_idx, size_t n_features,
                     size_t* indices, size_t n_samples, int* predictions);
```

### XGBoost

```c
typedef struct XGBoost_State {
    XGBoost_Tree** trees;
    size_t n_trees;
    double learning_rate;
    double reg_lambda;
    double reg_alpha;
    size_t max_depth;
} XGBoost_State;

int XGBoost_fit(const ML_Model_Config_t* config, XGBoost_State* state,
                const dataset* ds, size_t* feat_idx, size_t n_features,
                size_t label_idx, size_t* indices, size_t n_samples);

int XGBoost_predict(const XGBoost_State* state, const dataset* ds,
                    size_t* feat_idx, size_t n_features,
                    size_t* indices, size_t n_samples, int* predictions);

int XGBoost_predict_proba(const XGBoost_State* state, const dataset* ds,
                          size_t* feat_idx, size_t n_features,
                          size_t* indices, size_t n_samples,
                          size_t n_classes, void* probabilities);
```

### CatBoost

```c
typedef struct CatBoost_State {
    CatBoost_Tree** trees;
    size_t n_trees;
    double learning_rate;
    double l2_leaf_reg;
    size_t max_depth;
} CatBoost_State;

int CatBoost_fit(const ML_Model_Config_t* config, CatBoost_State* state,
                 const dataset* ds, size_t* feat_idx, size_t n_features,
                 size_t target_idx, size_t* indices, size_t n_samples);

int CatBoost_predict(const CatBoost_State* state, const dataset* ds,
                     size_t* feat_idx, size_t n_features,
                     size_t* indices, size_t n_samples, int* predictions);
```

---

## Deep Learning

### MLP (Multi-Layer Perceptron)

```c
typedef struct {
    Tensor* weights;
    Tensor* bias;
    Tensor* grad_w;
    Tensor* grad_b;
    Tensor* velocity_w;
    Tensor* velocity_b;
    size_t input_dim;
    size_t output_dim;
    ActivationType activation;
} FCLayer;

typedef struct {
    FCLayer** layers;
    size_t num_layers;
    size_t input_dim;
    size_t output_dim;
} MLP;

typedef struct {
    MLP* mlp;
    float lr;
    float momentum;
    float weight_decay;
} SGDOptimizer;

MLP* mlp_create(size_t input_dim, size_t hidden_dim, size_t output_dim,
                 size_t num_layers);
SGDOptimizer* sgd_create(MLP* mlp, float lr, float momentum, float weight_decay);
float mlp_train_step(MLP* mlp, SGDOptimizer* opt, Tensor* X, Tensor* y);
Tensor* mlp_predict(MLP* mlp, Tensor* input);
float mlp_accuracy(MLP* mlp, Tensor* input, Tensor* labels);
void mlp_free(MLP* mlp);
void sgd_free(SGDOptimizer* opt);
```

### OpenCL MLP

```c
typedef struct {
    CLTensor* weight;
    CLTensor* bias;
    CLTensor* grad_w;
    CLTensor* grad_b;
    CLTensor* input_cache;
    CLTensor* relu_mask;
} CLFCLayer;

typedef struct {
    CLTensor** velocity_w;
    CLTensor** velocity_b;
    size_t num_layers;
    CLSGDConfig config;
} CLSGDOptimizer;

CLOpenCLMLP* cl_mlp_create(CLOpenCL* cl, size_t input_dim, size_t hidden_dim,
                            size_t output_dim, size_t num_layers);
CLSGDOptimizer* cl_sgd_create(CLOpenCL* cl, CLOpenCLMLP* mlp,
                               float lr, float momentum, float weight_decay);
float cl_mlp_train_step(CLOpenCL* cl, cl_kernel_cache_t* cache,
                         CLOpenCLMLP* mlp, CLSGDOptimizer* opt,
                         const CLTensor* input, const int* targets);
CLTensor* cl_mlp_predict(CLOpenCL* cl, cl_kernel_cache_t* cache,
                         CLOpenCLMLP* mlp, const CLTensor* input);
float cl_mlp_accuracy(CLOpenCL* cl, cl_kernel_cache_t* cache,
                       CLOpenCLMLP* mlp, const CLTensor* input, const int* targets);
void cl_mlp_free(CLOpenCL* cl, CLOpenCLMLP* mlp);
void cl_sgd_free(CLSGDOptimizer* opt);
```

### LeNet-5

```c
typedef struct {
    ConvLayer* conv1;   // C1: 1→6 channels, 5×5
    ConvLayer* conv2;   // C3: 6→16 channels, 5×5
    ConvLayer* conv3;  // C5: 16→120 channels, 5×5
    FCLayer* fc1;      // F6: 120→84
    FCLayer* fc2;      // Output: 84→10
    bool training;
} LeNet5;

LeNet5* lenet5_create(void);
void lenet5_free(LeNet5* model);
Tensor* lenet5_forward(LeNet5* model, const Tensor* input);
float lenet5_train_step(LeNet5* model, float lr,
                          const Tensor* input, const Tensor* targets);
Tensor* lenet5_predict(LeNet5* model, const Tensor* input);
float lenet5_accuracy(LeNet5* model, const Tensor* input, const Tensor* targets);
```

### LSTM

```c
typedef struct {
    Tensor* W_f, *W_i, *W_c, *W_o;  // Input weights
    Tensor* R_f, *R_i, *R_c, *R_o;  // Recurrent weights
    Tensor* b_f, *b_i, *b_c, *b_o; // Biases
    // ... gradients and caches
} LSTMLayer;

typedef struct {
    LSTMLayer* layer;
    size_t input_size;
    size_t hidden_size;
    size_t num_layers;
    bool training;
    float clip_threshold;
} LSTM;

LSTM* lstm_create(size_t input_size, size_t hidden_size, size_t num_layers);
void lstm_free(LSTM* model);
Tensor* lstm_forward(LSTM* model, const Tensor* input);
void lstm_backward(LSTM* model, const Tensor* grad_output);
float lstm_train_step(LSTM* model, float lr,
                       const Tensor* input, const Tensor* targets);
Tensor* lstm_predict(LSTM* model, const Tensor* input);
```

### RNN

```c
typedef struct {
    Tensor* W_ih;   // Input-to-hidden
    Tensor* W_hh;   // Hidden-to-hidden
    Tensor* b;       // Bias
    Tensor* grad_W_ih;
    Tensor* grad_W_hh;
    Tensor* grad_b;
    Tensor** h_cache;
    Tensor** x_cache;
} RNNLayer;

typedef struct {
    RNNLayer* layer;
    size_t input_size;
    size_t hidden_size;
    size_t num_layers;
    bool training;
} RNN;

RNN* rnn_create(size_t input_size, size_t hidden_size, size_t num_layers);
void rnn_free(RNN* model);
Tensor* rnn_forward(RNN* model, const Tensor* input);
void rnn_backward(RNN* model, const Tensor* grad_output);
float rnn_train_step(RNN* model, float lr,
                      const Tensor* input, const Tensor* targets);
Tensor* rnn_predict(RNN* model, const Tensor* input);
```

---

## File Formats

### IDX (MNIST)

```c
typedef struct {
    IDX_TYPE type;
    uint8_t n_dims;
    uint32_t* dims;
    void* idx_data;
    size_t size;
} cIDX;

cIDX* idx_load(const char* path);
void free_idx(cIDX* idx);
```

### CSV

```c
typedef struct csv_t {
    csv_row* rows;
    size_t size;
} csv_t;

csv_t* csv_load(const char* filepath);
csv_t* parse_csv(const char* input);
void free_csv_data(csv_t* csv);
```
