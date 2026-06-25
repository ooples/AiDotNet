---
title: "ExtendedLSTMLayer<T>"
description: "Implements the Extended LSTM (xLSTM) layer from Hochreiter et al., 2024."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the Extended LSTM (xLSTM) layer from Hochreiter et al., 2024.

## For Beginners

xLSTM is a modernized version of the classic LSTM (1997).

The original LSTM was the dominant sequence model for years, but was overtaken by Transformers.
xLSTM brings it back by fixing key limitations:

1. **Exponential gating**: Instead of sigmoid (0 to 1), gates use exp() which can amplify

important signals, not just dampen them.

2. **Matrix memory**: Instead of a vector cell, mLSTM uses a matrix. This is like having

a lookup table that maps keys to values, similar to attention but stored as a running sum.

The result: an LSTM that matches Transformer performance at scale while maintaining the
efficient O(1) per-step inference of RNNs.

## How It Works

xLSTM modernizes the classic LSTM architecture with two key innovations:

1. **sLSTM (scalar LSTM)**: Enhanced gating with exponential activation functions and

a new memory mixing mechanism. Uses scalar (diagonal) memory cells.

2. **mLSTM (matrix LSTM)**: Replaces the scalar memory cell with a matrix-valued memory,

connecting LSTMs to modern linear attention/state space models.

This layer implements the mLSTM variant, which is the more impactful innovation:

The connection to linear attention: if f_t = 1 and i_t = 1, the matrix cell C_t accumulates
k*v outer products exactly like the state matrix in linear attention. The gates allow
selective forgetting and input scaling, which is what makes xLSTM competitive.

**Reference:** Beck et al., "xLSTM: Extended Long Short-Term Memory", 2024.
https://arxiv.org/abs/2405.04517

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExtendedLSTMLayer(Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new Extended LSTM (xLSTM) layer using the mLSTM (matrix memory) variant. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HeadDimension` | Gets the dimension per head. |
| `ModelDimension` | Gets the model dimension. |
| `NumHeads` | Gets the number of heads for the matrix memory. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` |  |
| `GetForgetGateWeights` | Gets the forget gate weights for external inspection. |
| `GetOutputProjectionWeights` | Gets the output projection weights for external inspection. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

