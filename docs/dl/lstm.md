# Long Short-Term Memory (LSTM)

Recurrent network with gated cell state for long-range dependencies.

## Overview

LSTM (Hochreiter & Schmidhuber, 1997) addresses the vanishing gradient problem in vanilla RNNs through gated memory cells that can selectively remember or forget information over long sequences.

## Architecture

```
Input: [batch, seq_len, input_size]
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  For each timestep t:                                   │
│                                                         │
│  f_t = σ(W_f @ x_t + R_f @ h_{t-1} + b_f)  // forget    │
│  i_t = σ(W_i @ x_t + R_i @ h_{t-1} + b_i)  // input     │
│  c_t = f_t * c_{t-1} + i_t * tanh(W_c @ x_t + R_c @ h_{t-1} + b_c) │
│  o_t = σ(W_o @ x_t + R_o @ h_{t-1} + b_o)  // output    │
│  h_t = o_t * tanh(c_t)                                  │
└─────────────────────────────────────────────────────────┘
         │
         ▼
Output: [batch, seq_len, hidden_size]
```

## Gates Explained

### Forget Gate ($f_t$)
Decides what information to discard from the cell state:
- $f_t = \sigma(\mathbf{W}_f \mathbf{x}_t + \mathbf{R}_f \mathbf{h}_{t-1} + \mathbf{b}_f)$
- Output near 1: keep the information
- Output near 0: forget the information

### Input Gate ($i_t$)
Decides what new information to store:
- $i_t = \sigma(\mathbf{W}_i \mathbf{x}_t + \mathbf{R}_i \mathbf{h}_{t-1} + \mathbf{b}_i)$

### Cell Candidate ($\tilde{c}_t$)
New candidate values:
- $\tilde{c}_t = \tanh(\mathbf{W}_c \mathbf{x}_t + \mathbf{R}_c \mathbf{h}_{t-1} + \mathbf{b}_c)$

### Output Gate ($o_t$)
Decides what to output:
- $o_t = \sigma(\mathbf{W}_o \mathbf{x}_t + \mathbf{R}_o \mathbf{h}_{t-1} + \mathbf{b}_o)$
- $\mathbf{h}_t = o_t \odot \tanh(c_t)$

## Data Structures

### LSTMLayer

```c
typedef struct {
    // Input weights [hidden_size, input_size]
    Tensor* W_f, *W_i, *W_c, *W_o;

    // Recurrent weights [hidden_size, hidden_size]
    Tensor* R_f, *R_i, *R_c, *R_o;

    // Biases [hidden_size]
    Tensor* b_f, *b_i, *b_c, *b_o;

    // Gradients
    Tensor *grad_W_f, *grad_W_i, *grad_W_c, *grad_W_o;
    Tensor *grad_R_f, *grad_R_i, *grad_R_c, *grad_R_o;
    Tensor *grad_b_f, *grad_b_i, *grad_b_c, *grad_b_o;

    // Caches for BPTT (backpropagation through time)
    Tensor** h_cache;      // Hidden states [seq_len]
    Tensor** c_cache;      // Cell states [seq_len]
    Tensor** x_cache;      // Inputs [seq_len]
    Tensor** f_cache;      // Forget gates [seq_len]
    Tensor** i_cache;      // Input gates [seq_len]
    Tensor** c_tilde_cache; // Cell candidates [seq_len]
    Tensor** o_cache;      // Output gates [seq_len]
    size_t seq_len;
} LSTMLayer;
```

### LSTM Model

```c
typedef struct {
    LSTMLayer* layer;
    size_t input_size;
    size_t hidden_size;
    size_t num_layers;
    bool training;
    float clip_threshold;  // Gradient clipping
} LSTM;
```

## Functions

```c
// Create/destroy
LSTM* lstm_create(size_t input_size, size_t hidden_size, size_t num_layers);
void lstm_free(LSTM* model);

// Forward/backward
Tensor* lstm_forward(LSTM* model, const Tensor* input);
void lstm_backward(LSTM* model, const Tensor* grad_output);

// Training
float lstm_train_step(LSTM* model, float lr,
                       const Tensor* input, const Tensor* targets);

// Prediction
Tensor* lstm_predict(LSTM* model, const Tensor* input);
```

## Forward Pass

```c
static Tensor* lstm_forward(LSTM* model, const Tensor* input) {
    // input: [batch, seq_len, input_size]
    size_t batch = input->shape[0];
    size_t seq_len = input->shape[1];
    size_t hidden_size = model->hidden_size;

    LSTMLayer* layer = model->layer;

    // Allocate output
    size_t out_shape[] = {batch, seq_len, hidden_size};
    Tensor* output = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW,
                                   out_shape, 3);

    // Initial states
    float* h_prev = calloc(batch * hidden_size, sizeof(float));
    float* c_prev = calloc(batch * hidden_size, sizeof(float));

    for (size_t t = 0; t < seq_len; t++) {
        float* x_t = &in_data[t * batch * input_size];
        float* h_t = &out_data[t * batch * hidden_size];

        // Compute all gates
        float* gates = malloc(4 * hidden_size * sizeof(float));
        float* f_t = gates;
        float* i_t = gates + hidden_size;
        float* c_tilde = gates + 2 * hidden_size;
        float* o_t = gates + 3 * hidden_size;

        for (size_t b = 0; b < batch; b++) {
            for (size_t h = 0; h < hidden_size; h++) {
                // f_t = sigmoid(W_f @ x + R_f @ h_prev + b_f)
                f_t[h] = sigmoidf(sum_f);
                // i_t = sigmoid(W_i @ x + R_i @ h_prev + b_i)
                i_t[h] = sigmoidf(sum_i);
                // c_tilde = tanh(W_c @ x + R_c @ h_prev + b_c)
                c_tilde[h] = tanhf(sum_c);
                // o_t = sigmoid(W_o @ x + R_o @ h_prev + b_o)
                o_t[h] = sigmoidf(sum_o);
            }
        }

        // Update cell and hidden state
        for (size_t b = 0; b < batch; b++) {
            for (size_t h = 0; h < hidden_size; h++) {
                c_t[b*hidden_size + h] = f_t[h] * c_prev[b*hidden_size + h]
                                        + i_t[h] * c_tilde[h];
                h_t[b*hidden_size + h] = o_t[h] * tanhf(c_t[b*hidden_size + h]);
            }
        }

        memcpy(c_prev, c_t, batch * hidden_size * sizeof(float));
        memcpy(h_prev, h_t, batch * hidden_size * sizeof(float));
    }

    return output;
}
```

## Backward Pass (BPTT)

```c
static void lstm_backward(LSTM* model, const Tensor* grad_output) {
    // Gradient buffers
    float* dh_next = calloc(batch * hidden_size, sizeof(float));
    float* dc_next = calloc(batch * hidden_size, sizeof(float));

    // Backward through time (reverse order)
    for (int t = seq_len - 1; t >= 0; t--) {
        // Get cached values
        float* x_t = layer->x_cache[t]->data;
        float* h_t = layer->h_cache[t]->data;
        float* c_t = layer->c_cache[t]->data;
        float* c_prev = (t > 0) ? layer->c_cache[t-1]->data : zeros;

        // dh_t = grad_output + dh_next
        float* dh = dh_next + grad_out[t * batch * hidden_size];

        // dc_t = dh_t * o_t * (1 - tanh^2(c_t)) + dc_next * f_t
        float* dc = dc_next;

        // Accumulate gradients for all gate parameters
        // (simplified - proper implementation accumulates over batch)
    }

    free(dh_next);
    free(dc_next);
}
```

## Gradient Flow

The LSTM's key innovation is uninterrupted gradient flow through the cell state:

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$

Since $f_t \in [0, 1]$, the gradient is scaled but not exponentially damped like vanilla RNNs where:

$$\frac{\partial h_t}{\partial h_{t-1}} = \tanh'(\mathbf{W}_h \mathbf{h}_{t-1} + \ldots) \cdot \mathbf{W}_h$$

## Example

```c
#define LSTM_IMPLEMENTATION
#include "lstm.h"

// Create LSTM: input=100, hidden=256, 1 layer
LSTM* model = lstm_create(100, 256, 1);

// Input: batch=32, seq_len=50, input_size=100
size_t input_shape[] = {32, 50, 100};
Tensor* input = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW,
                              input_shape, 3);
// Fill with data...

// Training
for (size_t epoch = 0; epoch < 20; epoch++) {
    float loss = lstm_train_step(model, 0.001f, input, targets);
    printf("Epoch %zu: loss=%.4f\n", epoch, loss);
}

// Prediction
Tensor* output = lstm_predict(model, input);

lstm_free(model);
```

## Variants

### Peephole Connections

Allow gates to directly observe the cell state:

```c
// f_t = σ(W_f @ x_t + R_f @ h_{t-1} + p_f ⊙ c_{t-1} + b_f)
```

### GRU (Gated Recurrent Unit)

Simplified variant with fewer gates:

```
z_t = σ(W_z @ x_t + R_z @ h_{t-1})      // Update gate
r_t = σ(W_r @ x_t + R_r @ h_{t-1})    // Reset gate
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ tanh(W_h @ x_t + R_h @ (r_t ⊙ h_{t-1}))
```

## Notes

- Gradient clipping prevents exploding gradients (threshold: 5.0)
- Xavier initialization for input weights, orthogonal init for recurrent
- Sigmoid used for gates, tanh for cell state and output
- Cell state uses hadamard product (*), not matrix multiplication
- For多层 LSTM, stack multiple LSTMLayer objects
