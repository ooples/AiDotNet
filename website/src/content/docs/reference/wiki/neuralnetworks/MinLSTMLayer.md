---
title: "MinLSTMLayer<T>"
description: "Implements the minLSTM layer from \"Were RNNs All We Needed?\" (Feng et al., 2024)."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the minLSTM layer from "Were RNNs All We Needed?" (Feng et al., 2024).

## For Beginners

minLSTM is a drastically simplified LSTM that trains much faster.

Traditional LSTMs have a chicken-and-egg problem: to compute the gates at step t, you need
the hidden state from step t-1, which means you must process the sequence one step at a time.
minLSTM solves this by making gates depend only on the current input:

- Standard LSTM gates: "What should I remember?" depends on what I currently remember (h_{t-1})
- minLSTM gates: "What should I remember?" depends only on what I'm currently seeing (x_t)

The normalization trick (f' + i' = 1) makes the update a weighted average:
new_state = (weight_forget * old_state) + (weight_input * new_candidate)
where weight_forget + weight_input = 1

This simple change means the entire sequence can be processed in parallel during training,
just like a Transformer, while still maintaining efficient O(1)-per-step inference.

## How It Works

minLSTM is a minimal LSTM variant that removes all hidden-state dependencies from the gates,
enabling fully parallelizable training via a parallel prefix scan. In a standard LSTM, the forget
gate f_t, input gate i_t, and output gate o_t all depend on the previous hidden state h_{t-1}.
minLSTM eliminates every one of these dependencies:

**Architecture:**

**Gate Normalization:** The normalization f' = f/(f+i), i' = i/(f+i) is the key insight
that makes minLSTM mathematically equivalent to a linear recurrence. Because f' + i' = 1,
the cell update is a convex combination of the previous state and the new candidate. This
is directly analogous to minGRU's formulation where z and (1-z) play the same role.
This convex structure allows the recurrence to be computed via a parallel scan operation,
transforming training complexity from O(T) sequential to O(log T) parallel depth.

**Comparison with minGRU:**

- minGRU uses a single gate z: h_t = (1-z)*h_{t-1} + z*tilde_h
- minLSTM uses two gates (f, i) that are normalized: c_t = f'*c_{t-1} + i'*c_tilde
- Both achieve input-only gating for parallel training
- minLSTM's two separate gates provide more expressive control over the forget-vs-input

trade-off, whereas minGRU ties them together through a single z

- Both outperform traditional LSTMs and GRUs on many sequence modeling benchmarks while

being significantly faster to train due to parallel scan

**Reference:** Feng et al., "Were RNNs All We Needed?", 2024.
https://arxiv.org/abs/2410.01201

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MinLSTMLayer(Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new minLSTM layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpandedDimension` | Gets the expanded internal dimension. |
| `ModelDimension` | Gets the model dimension (input/output width). |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateOnesLike(Tensor<>)` | Creates a tensor of ones with the same shape as the template tensor. |
| `Forward(Tensor<>)` |  |
| `GetForgetGateWeights` | Gets the forget gate weights for external inspection. |
| `GetInputGateWeights` | Gets the input gate weights for external inspection. |
| `GetOutputProjectionWeights` | Gets the output projection weights for external inspection. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeTensor2D(Tensor<>)` | Initializes a 2D tensor using Xavier/Glorot initialization. |
| `MinLSTMRecurrenceForward(Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Implements the minLSTM recurrence: c_t = f'_t * c_{t-1} + i'_t * c_tilde_t. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

