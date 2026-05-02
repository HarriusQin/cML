# API 参考

cML 机器学习库的完整 API 参考。

## 目录

1. [数据结构](#数据结构)
2. [通用工具](#通用工具)
3. [张量运算](#张量运算)
4. [OpenCL 张量](#opencl-张量)
5. [线性模型](#线性模型)
6. [集成学习](#集成学习)
7. [深度学习](#深度学习)

---

## 数据结构

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
    TENSOR_LAYOUT_ANY          // 手动指定 strides
} TensorLayout;
```

### Tensor

```c
typedef struct {
    void* data;               // 原始数据缓冲区
    size_t size;             // 元素总数
    TensorDType dtype;       // 数据类型
    TensorLayout layout;     // 内存布局
    uint8_t ndim;            // 维度数
    size_t shape[4];        // 最大 4 维
    size_t strides[4];        // 每维步长
    bool owner;             // 是否拥有数据
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

## 通用工具

### FeatureSample

```c
typedef struct {
    size_t idx;  // 样本索引
    double val;  // 特征值
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

### 数据集函数

```c
dataset* dataset_create(size_t n_samples, size_t n_features,
                        double** features, double* labels);
void dataset_free(dataset* ds);
dataset* csv_to_dataset(csv_t* csv, const char** label_cols, size_t num_labels);
void train_test_split(const dataset* ds, float test_ratio, unsigned int seed,
                      Dataset_Split_t* split);
ML_ScalingParams_t* ml_fit_scaling(const dataset* ds, size_t* feat_idx,
                                    size_t n_features, size_t* indices,
                                    size_t n_samples, ScalingType type);
```

---

## 张量运算

### 张量创建

```c
Tensor* tensor_create(TensorDType dtype, TensorLayout layout,
                      size_t* shape, uint8_t ndim);
Tensor* tensor_wrap(void* data, TensorDType dtype, TensorLayout layout,
                    size_t* shape, uint8_t ndim);
Tensor* tensor_clone(const Tensor* t);
void tensor_free(Tensor* t);
```

### 填充运算

```c
void tensor_fill_f32(Tensor* t, float val);
void tensor_fill_randn(Tensor* t, float mean, float std_dev);
void tensor_fill_xavier(Tensor* t);
```

### 逐元素运算

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

### 归约运算

```c
Tensor* tensor_sum(const Tensor* t, size_t axis);
Tensor* tensor_mean(const Tensor* t, size_t axis);
Tensor* tensor_max(const Tensor* t, size_t axis);
```

### 矩阵运算

```c
Tensor* tensor_matmul(const Tensor* a, const Tensor* b);
void tensor_gemm(float* C, const float* A, const float* B,
                size_t M, size_t N, size_t K,
                float alpha, float beta);
```

### 形状运算

```c
Tensor* tensor_reshape(const Tensor* t, size_t* new_shape, uint8_t ndim);
Tensor* tensor_transpose(const Tensor* t, uint8_t axis0, uint8_t axis1);
Tensor* tensor_slice(const Tensor* t, size_t dim, size_t start, size_t end);
```

### 池化和卷积

```c
Tensor* tensor_maxpool2d(const Tensor* input, size_t pool_h, size_t pool_w,
                        size_t stride_h, size_t stride_w);
Tensor* tensor_avgpool2d(const Tensor* input, size_t pool_h, size_t pool_w,
                        size_t stride_h, size_t stride_w);
Tensor* tensor_conv2d(const Tensor* input, const Tensor* weight,
                     const Conv2DParams* params);
```

---

## OpenCL 张量

### 上下文函数

```c
bool cl_init(CLOpenCL* cl, cl_device_type device_type);
void cl_release(CLOpenCL* cl);
void cl_print_device_info(cl_device_id device);
```

### 张量函数

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

### GPU 运算

```c
CLTensor* cl_tensor_relu(CLOpenCL* cl, const CLTensor* input);
CLTensor* cl_tensor_matmul(CLOpenCL* cl, const CLTensor* a, const CLTensor* b);
void cl_tensor_fill_f32(CLOpenCL* cl, CLTensor* t, float val);
CLTensor* cl_tensor_softmax(CLOpenCL* cl, const CLTensor* t, size_t axis);
```

---

## 线性模型

### 多元线性回归 (MLR)

```c
typedef struct {
    Matrix* weights;    // [n_features, 1]
    double* bias;
} MLR_State;

int mlr_fit(const ML_Model_Config_t* config, ML_Weights_t* state, ...);
int mlr_predict(const ML_Weights_t* state, const dataset* ds, ...);
```

### 岭回归

```c
typedef struct {
    double* weights;          // [n_features + 1]
    double alpha;             // 正则化参数
} Ridge_Weights_t;

ML_Model_t create_ridge_regression_model(double alpha);
int Ridge_fit(const ML_Model_Config_t* config, Ridge_Weights_t* state, ...);
```

### Softmax 回归

```c
typedef struct {
    double** weights;         // [K][n_features]
    double* bias;             // [K]
    size_t n_classes;
    double learning_rate;
} SoftmaxReg_Weights_t;

ML_Model_t create_softmax_model(void);
ML_Model_t create_softmax_model_gd(void);
int softmax_predict_proba(const SoftmaxReg_Weights_t* state, ...);
```

### 多项式回归

```c
typedef struct {
    Matrix* weights;    // [degree]
    double* bias;
    size_t degree;
} PolyReg_State;

int polynomial_features(const double* x, size_t n_samples, size_t degree, Matrix* result);
int poly_fit(const ML_Model_Config_t* config, ML_Weights_t* state, ...);
int poly_predict(const ML_Weights_t* state, const double* x, size_t n_samples, double* output);
```

### 加权最小二乘

```c
typedef struct {
    Matrix* weights;
    double* bias;
} WLS_State;

int wls_fit(const ML_Model_Config_t* config, ML_Weights_t* state,
            const dataset* ds, const size_t* feature_indices,
            size_t n_features, size_t target_index,
            const size_t* sample_indices, size_t n_samples,
            const double* weights);
int wls_predict(const ML_Weights_t* state, ...);
```

---

## 集成学习

### 决策树

```c
typedef struct DecisionTreeNode {
    size_t feature_index;
    double threshold;
    bool is_leaf;
    double value;
    struct DecisionTreeNode *left, *right;
} DecisionTreeNode;

DecisionTree* dt_create(size_t max_depth, size_t min_samples_split);
void dt_free(DecisionTree* tree);
int dt_fit(DecisionTree* tree, const dataset* ds, ...);
int dt_predict(const DecisionTree* tree, ...);
```

### 随机森林

```c
typedef struct {
    int n_estimators;
    DecisionTree** trees;
    double* feature_importance;
} RandomForest_Weights_t;

ML_Model_t create_random_forest_model(void);
int RandomForest_fit(const ML_Model_Config_t* config, RandomForest_Weights_t* state, ...);
int RandomForest_predict(const RandomForest_Weights_t* state, ...);
```

### AdaBoost

```c
typedef struct {
    DecisionTree** stumps;
    double* alphas;
    int n_stumps;
} AdaBoost_State;

ML_Model_t create_adaboost_model(void);
int AdaBoost_fit(const ML_Model_Config_t* config, AdaBoost_State* state, ...);
int AdaBoost_predict(const AdaBoost_State* state, ...);
```

### XGBoost

```c
typedef struct XGBoost_State {
    XGBoost_Tree** trees;
    size_t n_trees;
    double learning_rate;
    double reg_lambda;
    double reg_alpha;
} XGBoost_State;

int XGBoost_fit(const ML_Model_Config_t* config, XGBoost_State* state, ...);
int XGBoost_predict(const XGBoost_State* state, ...);
int XGBoost_predict_proba(const XGBoost_State* state, ...);
```

### CatBoost

```c
typedef struct CatBoost_State {
    CatBoost_Tree** trees;
    size_t n_trees;
    double learning_rate;
    double l2_leaf_reg;
} CatBoost_State;

int CatBoost_fit(const ML_Model_Config_t* config, CatBoost_State* state, ...);
int CatBoost_predict(const CatBoost_State* state, ...);
```

---

## 深度学习

### MLP（多层感知机）

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

MLP* mlp_create(size_t input_dim, size_t hidden_dim, size_t output_dim, size_t num_layers);
SGDOptimizer* sgd_create(MLP* mlp, float lr, float momentum, float weight_decay);
float mlp_train_step(MLP* mlp, SGDOptimizer* opt, Tensor* X, Tensor* y);
Tensor* mlp_predict(MLP* mlp, Tensor* input);
float mlp_accuracy(MLP* mlp, Tensor* input, Tensor* labels);
void mlp_free(MLP* mlp);
void sgd_free(SGDOptimizer* opt);
```

### OpenCL MLP

```c
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
```

### LeNet-5

```c
typedef struct {
    ConvLayer* conv1;
    ConvLayer* conv2;
    ConvLayer* conv3;
    FCLayer* fc1;
    FCLayer* fc2;
    bool training;
} LeNet5;

LeNet5* lenet5_create(void);
void lenet5_free(LeNet5* model);
Tensor* lenet5_forward(LeNet5* model, const Tensor* input);
float lenet5_train_step(LeNet5* model, float lr, const Tensor* input, const Tensor* targets);
Tensor* lenet5_predict(LeNet5* model, const Tensor* input);
float lenet5_accuracy(LeNet5* model, const Tensor* input, const Tensor* targets);
```

### LSTM

```c
typedef struct {
    Tensor* W_f, *W_i, *W_c, *W_o;  // 输入权重
    Tensor* R_f, *R_i, *R_c, *R_o;  // 循环权重
    Tensor* b_f, *b_i, *b_c, *b_o; // 偏置
    // ... 梯度和缓存 ...
} LSTMLayer;

LSTM* lstm_create(size_t input_size, size_t hidden_size, size_t num_layers);
void lstm_free(LSTM* model);
Tensor* lstm_forward(LSTM* model, const Tensor* input);
void lstm_backward(LSTM* model, const Tensor* grad_output);
float lstm_train_step(LSTM* model, float lr, const Tensor* input, const Tensor* targets);
Tensor* lstm_predict(LSTM* model, const Tensor* input);
```

### RNN

```c
RNN* rnn_create(size_t input_size, size_t hidden_size, size_t num_layers);
void rnn_free(RNN* model);
Tensor* rnn_forward(RNN* model, const Tensor* input);
void rnn_backward(RNN* model, const Tensor* grad_output);
float rnn_train_step(RNN* model, float lr, const Tensor* input, const Tensor* targets);
Tensor* rnn_predict(RNN* model, const Tensor* input);
```

---

## 文件格式

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
