---
title: "MEGALayer<T>"
description: "Implements the MEGA (Moving Average Equipped Gated Attention) layer from Ma et al., 2023."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the MEGA (Moving Average Equipped Gated Attention) layer from Ma et al., 2023.

## For Beginners

MEGA is like having two complementary systems working together:

1. The EMA is like a set of "smoothing filters" applied to the input sequence. Imagine running

your finger along a line in a graph to smooth out bumps -- each dimension has its own smoothing
strength. Some dimensions smooth heavily (capturing broad trends), others barely smooth at all
(preserving sharp details).

2. The attention mechanism then looks at these smoothed representations and decides which positions

are important for each output. Because the input has already been smoothed, the attention can
focus on content similarity rather than having to also learn positional patterns.

Together, EMA handles "where to look" (position-aware) and attention handles "what's relevant"
(content-aware), dividing the labor efficiently.

## How It Works

MEGA combines an exponential moving average (EMA) with gated single-head attention to capture
both position-aware local smoothing and content-based global mixing. It achieves strong results
on sequence modeling benchmarks while being significantly more efficient than multi-head attention.

The architecture:

The EMA acts as a learned positional prior: it smooths the input with different decay rates
per dimension, so some dimensions capture very local context (fast decay) while others retain
long-range information (slow decay). The attention mechanism then operates on these position-aware
representations, making it easier to learn both local and global dependencies.

**Reference:** Ma et al., "MEGA: Moving Average Equipped Gated Attention", ICLR 2023.
https://arxiv.org/abs/2209.10655

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MEGALayer(Int32,Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new MEGA (Moving Average Equipped Gated Attention) layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmaDimension` | Gets the EMA dimension. |
| `HeadDimension` | Gets the dimension per head. |
| `ModelDimension` | Gets the model dimension. |
| `NumHeads` | Gets the number of attention heads. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CausalAttentionForward(Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Multi-head causal softmax attention. |
| `EmaBackward(Tensor<>,Int32,Int32)` | Backward pass through the EMA recurrence. |
| `EmaForward(Tensor<>,Int32,Int32)` | Multi-dimensional EMA forward pass. |
| `Forward(Tensor<>)` |  |
| `GetEmaAlphaLogit` | Gets the EMA alpha logit parameters for external inspection. |
| `GetOutputProjectionWeights` | Gets the output projection weights for external inspection. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetQueryWeights` | Gets the query weights for external inspection. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

