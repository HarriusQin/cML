# Unified ML Interface

All models implement the `ML_Model_t` interface for consistent usage.

## ML_Model_t Structure

```c
typedef struct ML_Model_t {
    void* config;                    // Model-specific configuration
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

    ML_Weights_t state;              // Trained model state
} ML_Model_t;
```

## Model Creation Functions

Each model provides a creation function:

```c
ML_Model_t create_softmax_model(void);        // Softmax regression
ML_Model_t create_softmax_model_gd(void);      // Softmax with gradient descent
ML_Model_t create_linear_regression_model(void);
ML_Model_t create_ridge_regression_model(double alpha);
ML_Model_t create_decision_tree_model(void);
ML_Model_t create_random_forest_model(void);
ML_Model_t create_adaboost_model(void);
// etc.
```

## Generic Training Pattern

```c
// Create model
ML_Model_t model = create_softmax_model();

// Configure (if needed)
SoftmaxReg_Config config = {
    .learning_rate = 0.1,
    .max_iter = 500,
    .tolerance = 1e-5,
    .verbose = 1
};
model.config.params = &config;
model.config.size = sizeof(config);

// Fit
int ret = model.methods.fit(&model.config, &model.state,
                            train_ds, feat_idx, n_features, label_idx,
                            train_idx, train_size);
if (ret != 0) {
    fprintf(stderr, "Training failed\n");
    return 1;
}

// Predict
int predictions[100];
model.methods.predict(&model.state, test_ds, feat_idx, n_features,
                     test_idx, 100, predictions);

// Free
model.methods.free_state(&model.state);
```

## Accuracy Evaluation

```c
double compute_accuracy(const ML_Weights_t* state, const dataset* ds,
                      size_t* feat_idx, size_t n_features,
                      size_t* indices, size_t n_samples);
```

## Cross-Validation

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

## Model Serialization

```c
// Serialize to buffer
uint8_t buffer[1024];
size_t size = model.methods.serialize(&model.state, buffer, sizeof(buffer));

// Deserialize
model.methods.deserialize(&model.state, buffer, size);
```

## Notes

- All models use the same interface for consistency
- Configuration is model-specific (passed via config.params)
- State must be freed after use to prevent memory leaks
