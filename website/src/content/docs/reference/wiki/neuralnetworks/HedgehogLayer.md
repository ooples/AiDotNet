---
title: "HedgehogLayer<T>"
description: "Implements the Hedgehog layer from \"The Hedgehog and the Porcupine: Expressive Linear Attentions with Softmax Mimicry\" (Zhang et al., 2024, ICLR 2024)."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the Hedgehog layer from "The Hedgehog and the Porcupine: Expressive Linear Attentions
with Softmax Mimicry" (Zhang et al., 2024, ICLR 2024).

## For Beginners

Hedgehog makes linear attention much better by learning HOW to pay attention.

In standard linear attention, there's a mathematical trick (feature map) that converts the expensive
softmax operation into a cheaper form. But the standard tricks (like adding 1 and clipping negatives)
are crude approximations. It's like trying to draw a circle using only straight lines -- you can
approximate it, but it's never quite right.

Hedgehog says: "Instead of using a fixed approximation, let's LEARN the best feature map from data."
It uses a small neural network (just two layers) that is trained alongside the main model. This learned
feature map can capture much more nuanced attention patterns than any fixed formula.

The result: Hedgehog gets close to the quality of full softmax attention while keeping the speed
advantage of linear attention (O(n) instead of O(n^2)).

Think of it this way:

- Standard linear attention: "I'll use this simple formula to decide what's important"
- Hedgehog: "I'll learn the BEST formula for deciding what's important from the data itself"

## How It Works

Hedgehog replaces the fixed feature map in linear attention with a small trainable MLP that learns
to approximate softmax attention from data. Standard linear attention uses simple feature maps like
ELU(x)+1 or polynomial expansions, which poorly approximate softmax and lead to degraded quality.
Hedgehog instead trains the feature map end-to-end, achieving much better softmax approximation.

The architecture:

The key insight is that softmax attention can be decomposed as a kernel: softmax(QK^T) = phi(Q) phi(K)^T
for some (unknown) feature map phi. Rather than using a fixed approximation, Hedgehog learns phi directly.
The MLP is small (typically 64 hidden units) so the overhead is minimal, but the quality improvement
over fixed feature maps is substantial -- closing most of the gap to full softmax attention.

**Reference:** Zhang et al., "The Hedgehog and the Porcupine: Expressive Linear Attentions with Softmax Mimicry", ICLR 2024.
https://arxiv.org/abs/2402.04347

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HedgehogLayer(Int32,Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new Hedgehog layer with trainable feature maps for linear attention. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureMapHiddenDim` | Gets the hidden dimension of the feature map MLP. |
| `HeadDimension` | Gets the dimension per head. |
| `ModelDimension` | Gets the model dimension. |
| `NumHeads` | Gets the number of attention heads. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyFeatureMap([],Int32,[],[],[])` | Applies the trainable feature map MLP: phi(x) = W2 * GELU(W1 * x + b1) + b2 |
| `Forward(Tensor<>)` |  |
| `GetFeatureMapW1` | Gets the feature map W1 weights for external inspection. |
| `GetFeatureMapW2` | Gets the feature map W2 weights for external inspection. |
| `GetOutputProjectionWeights` | Gets the output projection weights for external inspection. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetQueryWeights` | Gets the query weights for external inspection. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `LinearAttentionForward(Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Causal linear attention with learned feature maps. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

