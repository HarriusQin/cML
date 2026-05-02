# Recurrent Neural Network (RNN)

Simple Elman RNN for sequential data processing.

## Overview

The vanilla RNN (Elman network) processes sequences by maintaining hidden state that captures context from previous timesteps. Unlike LSTMs, it has a simple recurrence that makes it prone to vanishing gradients.

## Architecture

```
Input: [batch, seq_len, input_size]
         │
         ▼
┌────────────────────────────────────────────┐
│  h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)  │
│                                            │
│  where h_{-1} = 0 (initial hidden state)   │
└────────────────────────────────────────────┘
         │
         ▼
Output: [batch, seq_len, hidden_size]
```

## Forward Pass

For each timestep:
$$\mathbf{h}_t = \tanh(\mathbf{W}_{ih}\mathbf{x}_t + \mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{b})$$

The hidden state $\mathbf{h}_t$ is passed both to the output and to the next timestep.

## Data Structures

### RNNLayer

```c
typedef struct {
    Tensor* W_ih;   // [hidden_size, input_size] input-to-hidden
    Tensor* W_hh;   // [hidden_size, hidden_size] hidden-to-hidden
    Tensor* b;      // [hidden_size] bias
    Tensor* grad_W_ih;
    Tensor* grad_W_hh;
    Tensor* grad_b;

    // BPTT caches
    Tensor** h_cache;   // Hidden states per timestep
    Tensor** x_cache;   // Inputs per timestep
    size_t seq_len;
} RNNLayer;
```

### RNN Model

```c
typedef struct {
    RNNLayer* layer;
    size_t input_size;
    size_t hidden_size;
    size_t num_layers;
    bool training;
} RNN;
```

## Functions

```c
// Create/destroy
RNN* rnn_create(size_t input_size, size_t hidden_size, size_t num_layers);
void rnn_free(RNN* model);

// Forward/backward
Tensor* rnn_forward(RNN* model, const Tensor* input);
void rnn_backward(RNN* model, const Tensor* grad_output);

// Training
float rnn_train_step(RNN* model, float lr,
                      const Tensor* input, const Tensor* targets);

// Prediction
Tensor* rnn_predict(RNN* model, const Tensor* input);
```

## Forward Pass

```c
static Tensor* rnn_forward(RNN* model, const Tensor* input) {
    // input: [batch, seq_len, input_size]
    size_t batch = input->shape[0];
    size_t seq_len = input->shape[1];
    size_t hidden_size = model->hidden_size;

    RNNLayer* layer = model->layer;
    layer->seq_len = seq_len;

    // Allocate caches
    layer->h_cache = malloc(sizeof(Tensor*) * seq_len);
    layer->x_cache = malloc(sizeof(Tensor*) * seq_len);

    // Output tensor
    size_t out_shape[] = {batch, seq_len, hidden_size};
    Tensor* output = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW,
                                   out_shape, 3);

    // h_{-1} = zeros
    float* h_prev = calloc(hidden_size, sizeof(float));

    for (size_t t = 0; t < seq_len; t++) {
        float* x_t = &in_data[t * batch * input_size];
        float* h_t = &out_data[t * batch * hidden_size];

        // Cache input and hidden state
        layer->x_cache[t] = tensor_clone(x_t);
        layer->h_cache[t] = tensor_clone(h_t);

        // h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
        for (size_t b = 0; b < batch; b++) {
            for (size_t h = 0; h < hidden_size; h++) {
                float sum = b[h];

                // x_t @ W_ih.T
                for (size_t i = 0; i < input_size; i++) {
                    sum += x_t[b * input_size + i] * w_ih[h * input_size + i];
                }

                // h_prev @ W_hh.T
                for (size_t hh = 0; hh < hidden_size; hh++) {
                    sum += h_prev[hh] * w_hh[h * hidden_size + hh];
                }

                h_t[b * hidden_size + h] = tanhf(sum);
            }
        }

        memcpy(h_prev, h_t, batch * hidden_size * sizeof(float));
    }

    free(h_prev);
    return output;
}
```

## Backward Pass (BPTT)

Backpropagation Through Time unrolls the network and computes gradients:

```c
static void rnn_backward(RNN* model, const Tensor* grad_output) {
    size_t batch = grad_output->shape[0];
    size_t seq_len = grad_output->shape[1];
    size_t hidden_size = model->hidden_size;
    size_t input_size = model->input_size;

    float* dh_next = calloc(hidden_size, sizeof(float));

    // Backward through time (reverse order)
    for (int t = seq_len - 1; t >= 0; t--) {
        float* dh = dh_next + grad_out[t * batch * hidden_size];
        float* h_t = layer->h_cache[t]->data;

        // Add gradient from output at this timestep
        for (size_t h = 0; h < hidden_size; h++) {
            float tanh_deriv = 1.0f - h_t[h] * h_t[h];
            dh[h] = dh[h] + grad_out[t * batch * hidden_size + h] * tanh_deriv;
        }

        // dL/dW_ih = sum over batch (dh_t).T @ x_t
        float* x_t = layer->x_cache[t]->data;
        for (size_t h = 0; h < hidden_size; h++) {
            for (size_t i = 0; i < input_size; i++) {
                float sum = 0.0f;
                for (size_t b = 0; b < batch; b++)
                    sum += dh[b * hidden_size + h] * x_t[b * input_size + i];
                grad_W_ih[h * input_size + i] += sum / batch;
            }
        }

        // dL/dW_hh = sum over batch (dh_t).T @ h_{t-1}
        float* h_prev = (t > 0) ? layer->h_cache[t-1]->data : zeros;
        for (size_t h = 0; h < hidden_size; h++) {
            for (size_t hh = 0; hh < hidden_size; hh++) {
                float sum = 0.0f;
                for (size_t b = 0; b < batch; b++)
                    sum += dh[b * hidden_size + h] * h_prev[b * hidden_size + hh];
                grad_W_hh[h * hidden_size + hh] += sum / batch;
            }
        }

        // dh_next = dh_t @ W_hh (for previous timestep)
        memset(dh_next, 0, batch * hidden_size * sizeof(float));
        for (size_t hh = 0; hh < hidden_size; hh++) {
            float sum = 0.0f;
            for (size_t h = 0; h < hidden_size; h++)
                sum += dh[h] * w_hh[h * hidden_size + hh];
            dh_next[hh] = sum;
        }
    }

    free(dh_next);
}
```

## Vanishing Gradient Problem

In vanilla RNNs, gradients are multiplied by $\tanh'(\mathbf{z}_t) \cdot \mathbf{W}_{hh}$ at each step:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}} = \sum_{t=0}^{T-1} \left( \prod_{k=t+1}^{T-1} \tanh'(\mathbf{z}_k) \mathbf{W}_{hh} \right) \frac{\partial \mathcal{L}}{\partial \mathbf{h}_t}$$

Since $\|\tanh'(\mathbf{z})\| \leq 1$, and $\mathbf{W}_{hh}$ typically has eigenvalues $< 1$, the product decays exponentially.

## Comparison with LSTM

| Aspect | RNN | LSTM |
|--------|-----|------|
| Memory | Single hidden state | Cell state + gates |
| Gradient flow | May vanish | Uninterrupted via cell state |
| Gates | None | Forget, input, output |
| Parameters | Fewer | 4× more |
| Training difficulty | Harder | Easier |
| Sequence length | Short (< 50) | Long (< 1000) |

## Example

```c
#define RNN_IMPLEMENTATION
#include "rnn.h"

// Create RNN: input=50, hidden=128, 1 layer
RNN* model = rnn_create(50, 128, 1);

// Input: batch=16, seq_len=30, input_size=50
size_t input_shape[] = {16, 30, 50};
Tensor* input = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW,
                              input_shape, 3);
// Fill with data...

// Training loop with MSE loss
for (size_t epoch = 0; epoch < 50; epoch++) {
    float loss = rnn_train_step(model, 0.001f, input, targets);
    printf("Epoch %zu: loss=%.4f\n", epoch, loss);
}

// Prediction
Tensor* output = rnn_predict(model, input);

rnn_free(model);
```

## Notes

- Gradient clipping prevents exploding gradients
- Xavier initialization: $\text{std} = \sqrt{2/(n_{in} + n_{out})}$
- tanh activation keeps values in $[-1, 1]$
- For深层 RNNs, stack multiple RNNLayer objects
- Truncated BPTT often used for long sequences (ignore long-range dependencies)
