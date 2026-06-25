---
title: "RRDBLayer<T>"
description: "Residual in Residual Dense Block (RRDB) - the core building block of ESRGAN and Real-ESRGAN generators."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Residual in Residual Dense Block (RRDB) - the core building block of ESRGAN and Real-ESRGAN generators.

## For Beginners

RRDB is like a "super block" that contains 3 smaller blocks (RDBs).

The key insight is **residual-in-residual** learning:

- Each RDB has its own residual connection (local)
- The entire RRDB also has a residual connection (global)

This nested residual structure helps:

- Very deep networks train more easily
- Gradients flow better during backpropagation
- The network can learn fine details without losing coarse features

Real-ESRGAN typically uses 23 RRDB blocks, each containing 3 RDBs,
for a total of 69 residual dense blocks!

## How It Works

RRDB combines 3 Residual Dense Blocks with a global residual connection.
This is the architecture from the ESRGAN paper (Wang et al., 2018) that enables
training very deep networks for high-quality image super-resolution.

The architecture is:

**Reference:** Wang et al., "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks",
ECCV 2018 Workshops. https://arxiv.org/abs/1809.00219

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RRDBLayer(Int32,Int32,Double)` | Initializes a new RRDB layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GrowthChannels` | Gets the growth channels used in each RDB. |
| `NumFeatures` | Gets the number of feature channels. |
| `ParameterCount` |  |
| `ResidualScale` | Gets the global residual scaling factor. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddResidual(Tensor<>,Tensor<>,Double)` | Adds residual with scaling: output = a * scale + b. |
| `AddTensors(Tensor<>,Tensor<>)` | Adds two tensors element-wise. |
| `Forward(Tensor<>)` |  |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass on GPU tensors. |
| `GetParameters` |  |
| `OnFirstForward(Tensor<>)` |  |
| `ResetState` |  |
| `ScaleGradient(Tensor<>,Double)` | Scales a tensor by a factor. |
| `ScaleNode(ComputationNode<>,Double,String)` | Scales a computation node by a scalar value using element-wise multiplication. |
| `SetParameters(Vector<>)` |  |
| `UpdateParameters()` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_gpuLastInput` | GPU cached input tensor for backward pass. |
| `_growthChannels` | Growth channels for each RDB. |
| `_lastInput` | Cached input for backpropagation. |
| `_numFeatures` | Number of feature channels. |
| `_rdb3Output` | Cached output from RDB3 for backpropagation. |
| `_rdbBlocks` | The 3 Residual Dense Blocks that make up this RRDB. |
| `_residualScale` | Global residual scaling factor. |

