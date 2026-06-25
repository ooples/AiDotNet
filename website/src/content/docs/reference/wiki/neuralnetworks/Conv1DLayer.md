---
title: "Conv1DLayer<T>"
description: "1D convolutional layer for sequence / waveform data, with optional dilation."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

1D convolutional layer for sequence / waveform data, with optional
dilation. Operates on rank-3 input `[B, C_in, T]` and produces
rank-3 output `[B, C_out, T_out]`, where
`T_out = (T + 2*padding - dilation*(kernelSize - 1) - 1) / stride + 1`
per the standard 1D convolution formula (PyTorch `nn.Conv1d`
convention).

## How It Works

Implemented by delegating to `Engine.Conv2D` with the time axis
expanded to a degenerate 2D layout — input `[B, C, T]` is
reshaped to `[B, C, 1, T]`, kernel shape is
`[C_out, C_in, 1, kernelSize]`, dilation is `(1, dilation)`,
and padding is `(0, padding)`. This avoids duplicating the conv
kernel inside the layer and keeps the tape autodiff path identical to
every other Conv layer in the codebase. The degenerate height axis is
reshaped away on return.

Used by `DiffWaveModel` for the
dilated convolution stack from Kong et al. 2020 "DiffWave" §3 — kernel
size 3, dilation `2^(i % dilation_cycle)`. Also valid as a 1×1
channel mixer (`kernelSize=1`) — the same shape used by DiffWave
for the input/skip/output projections.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Conv1DLayer(Int32,Int32,Int32,Int32,Int32,Nullable<Int32>,IActivationFunction<>,IInitializationStrategy<>)` | Eager-init constructor — pre-allocates kernel and bias tensors at construction time when the input channel count is known up-front (DiffWave / WaveNet style architectures with fixed per-block channel counts). |
| `Conv1DLayer(Int32,Int32,Int32,Int32,Nullable<Int32>,IActivationFunction<>,IInitializationStrategy<>)` | Initializes a new 1D convolutional layer with lazy input-channel resolution. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Live parameter count. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` |  |
| `GetMetadata` | Returns layer-specific metadata for serialization. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `OnFirstForward(Tensor<>)` |  |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

