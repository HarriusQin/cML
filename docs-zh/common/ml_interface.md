# 统一 ML 接口

所有模型都实现 `ML_Model_t` 接口以保持一致的用法。

## ML_Model_t 结构

```c
typedef struct ML_Model_t {
    void* config;                    // 模型特定配置
    size_t config_size;

    struct {
        int (*fit)(ML_Model_Config_t* config, ML_Weights_t* state,
                   const dataset* ds, size_t* feat_idx, size_t n_features,
                   size_t label_idx, size_t* indices, size_t n_samples);

        int (*predict)(const ML_Weights_t* state, const dataset* ds,
                       size_t* feat_idx, size_t n_features,
                       size_t* indices, size_t n_samples, int* predictions);

        int (*predict_proba)(const ML_Weights_t* state, const dataset* ds,
                             size_t* feat_idx, size_t n_features,
                             size_t* indices, size_t n_samples,
                             float* probabilities);

        int (*get_coefficients)(const ML_Weights_t* state, double* coef);

        int (*serialize)(const ML_Weights_t* state, uint8_t* buffer, size_t size);
        int (*deserialize)(ML_Weights_t* state, const uint8_t* buffer, size_t size);

        void (*free_state)(ML_Weights_t* state);
    } methods;

    ML_Weights_t state;              // 已训练模型状态
} ML_Model_t;
```

## 模型创建函数

每个模型提供一个创建函数：

```c
ML_Model_t create_softmax_model(void);        // Softmax 回归
ML_Model_t create_softmax_model_gd(void);      // 梯度下降 Softmax
ML_Model_t create_linear_regression_model(void);
ML_Model_t create_ridge_regression_model(double alpha);
ML_Model_t create_decision_tree_model(void);
ML_Model_t create_random_forest_model(void);
ML_Model_t create_adaboost_model(void);
// 等等
```

## 通用训练模式

```c
// 创建模型
ML_Model_t model = create_softmax_model();

// 配置（如需要）
SoftmaxReg_Config config = {
    .learning_rate = 0.1,
    .max_iter = 500,
    .tolerance = 1e-5,
    .verbose = 1
};
model.config.params = &config;
model.config.size = sizeof(config);

// 训练
int ret = model.methods.fit(&model.config, &model.state,
                            train_ds, feat_idx, n_features, label_idx,
                            train_idx, train_size);
if (ret != 0) {
    fprintf(stderr, "训练失败\n");
    return 1;
}

// 预测
int predictions[100];
model.methods.predict(&model.state, test_ds, feat_idx, n_features,
                     test_idx, 100, predictions);

// 释放
model.methods.free_state(&model.state);
```

## 准确率评估

```c
double compute_accuracy(const ML_Weights_t* state, const dataset* ds,
                      size_t* feat_idx, size_t n_features,
                      size_t* indices, size_t n_samples);
```

## 交叉验证

```c
typedef struct {
    double accuracy;
    double precision;
    double recall;
    double f1_score;
    double auc;
    double log_loss;
} CV_Result_t;

CV_Result_t kfold_cross_validate(ML_Model_t* model, const dataset* ds,
                                  size_t label_idx, int k);
```

## 模型序列化

```c
// 序列化到缓冲区
uint8_t buffer[1024];
size_t size = model.methods.serialize(&model.state, buffer, sizeof(buffer));

// 反序列化
model.methods.deserialize(&model.state, buffer, size);
```

## 说明

- 所有模型使用相同接口以保持一致性
- 配置是模型特定的（通过 config.params 传递）
- 使用后必须释放状态以防止内存泄漏
