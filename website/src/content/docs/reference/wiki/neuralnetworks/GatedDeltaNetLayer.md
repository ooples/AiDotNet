---
title: "GatedDeltaNetLayer<T>"
description: "Implements the GatedDeltaNet layer from NVIDIA, ICLR 2025."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the GatedDeltaNet layer from NVIDIA, ICLR 2025.

## For Beginners

GatedDeltaNet is one of the best sub-quadratic architectures as of 2025.

Think of the state matrix S as a "lookup table" that maps keys to values:

- Linear attention: "Just add key-value pairs to the table" -> entries pile up, old ones never corrected
- Delta rule: "Before adding, check if this key already has a value. Only write the correction."

This is like the difference between:

- Memorizing every flashcard answer independently (linear attention)
- Checking what you already know, then only memorizing what's new or different (delta rule)

The gating mechanism (alpha, beta) lets the model control:

- How much to forget old entries (alpha)
- How strongly to write new corrections (beta)

Combined with a short convolution for local context, this simple recipe matches Transformers
while being much more efficient for long sequences.

## How It Works

GatedDeltaNet combines the delta rule for fast weight updates with gated output, achieving
state-of-the-art performance among sub-quadratic architectures. It matches Transformer quality
on many benchmarks while maintaining linear O(n) complexity.

The architecture:

The delta rule update is key: instead of blindly accumulating K*V outer products (like linear
attention), it computes the error (V - S*K) first and updates accordingly. This is exactly the
delta rule from neural network learning theory, applied to the fast weight matrix at each step.

**Reference:** Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule", ICLR 2025.
https://arxiv.org/abs/2412.06464

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GatedDeltaNetLayer(Int32,Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new GatedDeltaNet layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ConvKernelSize` | Gets the convolution kernel size. |
| `HeadDimension` | Gets the dimension per head. |
| `ModelDimension` | Gets the model dimension. |
| `NumHeads` | Gets the number of heads. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DeltaRuleForward(Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Delta rule forward: fast weight update with gated forgetting. |
| `DepthwiseConv1DForward(Tensor<>,Int32,Int32)` | Depthwise causal Conv1D forward. |
| `Forward(Tensor<>)` |  |
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

