# Deep Learning Models Guide

This guide documents the deep learning models implemented in cML.

## Table of Contents

1. [LeNet-5](#lenet-5)
2. [RNN (Recurrent Neural Network)](#rnn)
3. [LSTM (Long Short-Term Memory)](#lstm)
4. [Transformer](#transformer)

---

## LeNet-5

LeNet-5 is a classic convolutional neural network designed for handwritten digit recognition (MNIST).

### Architecture

```
Input: [N, 1, 32, 32]
    │
    ▼
C1: Conv(1→6, 5×5) → [N, 6, 28, 28] → tanh
    │
    ▼
S2: AvgPool(2×2, stride 2) → [N, 6, 14, 14]
    │
    ▼
C3: Conv(6→16, 5×5) → [N, 16, 10, 10] → tanh
    │
    ▼
S4: AvgPool(2×2, stride 2) → [N, 16, 5, 5]
    │
    ▼
C5: Conv(16→120, 5×5) → [N, 120, 1, 1] → tanh
    │
    ▼
Flatten → [N, 120]
    │
    ▼
F6: FC(120→84) → [N, 84] → tanh
    │
    ▼
Output: FC(84→10) → [N, 10]
```

### Usage

```c
#include "lenet5.h"

// Create model
LeNet5* model = lenet5_create();

// Forward pass
Tensor* input = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW,
                               (size_t[]){batch, 1, 32, 32}, 4);
// ... fill input data ...
Tensor* output = lenet5_forward(model, input);

// Predict (includes softmax)
Tensor* pred = lenet5_predict(model, input);

// Training step
float loss = lenet5_train_step(model, 0.01f, input, targets);

// Free resources
lenet5_free(model);
```

### Key Functions

| Function | Description |
|----------|-------------|
| `lenet5_create()` | Create LeNet-5 model |
| `lenet5_forward(model, input)` | Forward pass |
| `lenet5_predict(model, input)` | Predict with softmax |
| `lenet5_train_step(model, lr, input, targets)` | Single training step |
| `lenet5_free(model)` | Free model |

### Compilation

```bash
gcc test_lenet5.c tensor.h -o test_lenet5 -lm
./test_lenet5
```

---

## RNN

Simple Elman RNN for sequence processing.

### Architecture

```
h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
```

### Input/Output

- **Input**: `[batch, seq_len, input_size]`
- **Output**: `[batch, seq_len, hidden_size]`

### Usage

```c
#include "rnn.h"

// Create model: input_size=10, hidden_size=32, num_layers=1
RNN* model = rnn_create(10, 32, 1);

// Input: [batch=4, seq_len=8, input_size=10]
size_t shape[] = {4, 8, 10};
Tensor* input = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 3);
// ... fill input data ...

// Forward pass
Tensor* output = rnn_forward(model, input);

// Training step
Tensor* targets = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW,
                               (size_t[]){4, 8, 32}, 3);
float loss = rnn_train_step(model, 0.001f, input, targets);

// Free
rnn_free(model);
```

### Key Functions

| Function | Description |
|----------|-------------|
| `rnn_create(input_size, hidden_size, num_layers)` | Create RNN |
| `rnn_forward(model, input)` | Forward pass |
| `rnn_backward(model, grad_output)` | Backward pass (BPTT) |
| `rnn_train_step(model, lr, input, targets)` | Training step |
| `rnn_free(model)` | Free model |

### Compilation

```bash
gcc test_rnn.c tensor.h -o test_rnn -lm
./test_rnn
```

---

## LSTM

Long Short-Term Memory network for learning long-range dependencies.

### Architecture

```
f_t = sigmoid(W_f @ x_t + W_h @ h_{t-1} + b_f)  // forget gate
i_t = sigmoid(W_i @ x_t + W_h @ h_{t-1} + b_i)  // input gate
c_t = f_t * c_{t-1} + i_t * tanh(W_c @ x_t + W_h @ h_{t-1} + b_c)  // cell
o_t = sigmoid(W_o @ x_t + W_h @ h_{t-1} + b_o)  // output gate
h_t = o_t * tanh(c_t)
```

### Input/Output

- **Input**: `[batch, seq_len, input_size]`
- **Output**: `[batch, seq_len, hidden_size]`

### Usage

```c
#include "lstm.h"

// Create model: input_size=10, hidden_size=32
LSTM* model = lstm_create(10, 32, 1);

// Input: [batch=4, seq_len=8, input_size=10]
size_t shape[] = {4, 8, 10};
Tensor* input = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 3);
// ... fill input data ...

// Forward pass
Tensor* output = lstm_forward(model, input);

// Training step
float loss = lstm_train_step(model, 0.001f, input, targets);

// Free
lstm_free(model);
```

### Key Functions

| Function | Description |
|----------|-------------|
| `lstm_create(input_size, hidden_size, num_layers)` | Create LSTM |
| `lstm_forward(model, input)` | Forward pass |
| `lstm_backward(model, grad_output)` | Backward pass (BPTT) |
| `lstm_train_step(model, lr, input, targets)` | Training step |
| `lstm_free(model)` | Free model |

### Features

- Gradient clipping to prevent exploding gradients
- Cell state and hidden state caching for BPTT
- Four-gate architecture (forget, input, cell candidate, output)

### Compilation

```bash
gcc test_lstm.c tensor.h -o test_lstm -lm
./test_lstm
```

---

## Transformer

Transformer architecture with multi-head self-attention.

### Architecture

```
Encoder:
  x → Embedding → PosEncoding → [EncoderLayer × N] → LayerNorm → Output

EncoderLayer:
  x → MultiHeadAttention → Residual → LayerNorm → FFN → Residual → LayerNorm

Decoder:
  y → Embedding → PosEncoding → [DecoderLayer × N] → LayerNorm → Output

DecoderLayer:
  y → MaskedMHAttention → Residual → LayerNorm →
      CrossAttention → Residual → LayerNorm →
      FFN → Residual → LayerNorm
```

### Input/Output

- **Input (encoder)**: `[batch, seq_len]` (token IDs)
- **Output**: `[batch, seq_len, d_model]`

### Configuration

```c
typedef struct {
    size_t vocab_size;      // vocabulary size
    size_t d_model;          // embedding dimension
    size_t n_heads;          // number of attention heads
    size_t n_encoder_layers;
    size_t n_decoder_layers;
    size_t d_ff;             // feed-forward dimension
    size_t max_seq_len;      // maximum sequence length
    float dropout_p;
} TransformerConfig;
```

### Usage

```c
#include "transformer.h"

// Configure
TransformerConfig config = {
    .vocab_size = 100,
    .d_model = 64,
    .n_heads = 4,
    .n_encoder_layers = 2,
    .n_decoder_layers = 2,
    .d_ff = 256,
    .max_seq_len = 50,
    .dropout_p = 0.1f
};

// Create model
Transformer* model = transformer_create(config);

// Input: [batch=2, seq_len=8] (token IDs)
size_t shape[] = {2, 8};
Tensor* input = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 2);
// ... fill with token IDs (0 to vocab_size-1) ...

// Forward pass
Tensor* output = transformer_forward(model, input);

// Free
transformer_free(model);
```

### Key Components

#### Multi-Head Attention (MHAttention)

```
Q = x @ W_q.T
K = x @ W_k.T
V = x @ W_v.T

Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V

output = Attention(Q, K, V) @ W_o.T
```

#### Feed-Forward Network (FFN)

```
FFN(x) = ReLU(x @ W1.T + b1) @ W2.T + b2
```

#### Layer Normalization

```
LN(x) = gamma * (x - mean) / sqrt(var + eps) + beta
```

### Key Functions

| Function | Description |
|----------|-------------|
| `transformer_create(config)` | Create Transformer |
| `transformer_forward(model, input)` | Forward pass |
| `transformer_free(model)` | Free model |
| `mha_create(d_model, n_heads)` | Create multi-head attention |
| `ffn_create(d_model, d_ff)` | Create feed-forward network |
| `layer_norm_create(d_model, eps)` | Create layer normalization |

### Compilation

```bash
gcc test_transformer.c tensor.h -o test_transformer -lm
./test_transformer
```

---

## New Layers Added to tensor.h

### Batch Normalization

```c
// Training mode
void tensor_batch_norm_train(const Tensor* x, Tensor* y,
                             const Tensor* gamma, const Tensor* beta,
                             Tensor* mean, Tensor* var,
                             Tensor* running_mean, Tensor* running_var,
                             float momentum, float eps);

// Inference mode
void tensor_batch_norm_inference(const Tensor* x, Tensor* y,
                                 const Tensor* gamma, const Tensor* beta,
                                 const Tensor* mean, const Tensor* var,
                                 float eps);
```

### Dropout

```c
void tensor_dropout_forward(const Tensor* x, Tensor* y, Tensor* mask,
                            float p, bool training);

void tensor_dropout_backward(const Tensor* grad_output, Tensor* grad_input,
                             const Tensor* mask, float p);
```

### Layer Normalization

```c
void tensor_layer_norm_forward(const Tensor* x, Tensor* y,
                               const Tensor* gamma, const Tensor* beta,
                               float eps);
```

### Upsample (Nearest Neighbor)

```c
// Input: [N, C, H, W], Output: [N, C, H*scale_h, W*scale_w]
Tensor* tensor_upsample_nearest_2d(const Tensor* x, size_t scale_h, size_t scale_w);
```

---

## Testing

All models include test files:

```bash
# Compile and run all tests
gcc test_lenet5.c tensor.h -o test_lenet5 -lm && ./test_lenet5
gcc test_rnn.c tensor.h -o test_rnn -lm && ./test_rnn
gcc test_lstm.c tensor.h -o test_lstm -lm && ./test_lstm
gcc test_transformer.c tensor.h -o test_transformer -lm && ./test_transformer
```

### Test with Address Sanitizer

```bash
gcc -fsanitize=address test_lenet5.c tensor.h -o test_lenet5_asan -lm
./test_lenet5_asan
```

---

## Files

| File | Description |
|------|-------------|
| `lenet5.h` | LeNet-5 CNN implementation |
| `rnn.h` | RNN implementation |
| `lstm.h` | LSTM implementation |
| `transformer.h` | Transformer implementation |
| `test_lenet5.c` | LeNet-5 tests |
| `test_rnn.c` | RNN tests |
| `test_lstm.c` | LSTM tests |
| `test_transformer.c` | Transformer tests |
