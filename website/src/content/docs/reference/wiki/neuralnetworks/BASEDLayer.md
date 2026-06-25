---
title: "BASEDLayer<T>"
description: "Implements the BASED (Bidirectional Attention with Sliding-window and Expanded features) layer from \"Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff\" (Arora et al., 2024)."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the BASED (Bidirectional Attention with Sliding-window and Expanded features) layer from
"Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff" (Arora et al., 2024).

## For Beginners

BASED combines two ways of paying attention to solve a key tradeoff:

Imagine you're reading a long book and need to answer questions about it:

- Linear attention is like having a summary of everything you've read so far. It's fast to update

and gives you the gist, but you might miss exact details (like a specific name on page 42).

- Sliding window attention is like being able to flip back a few pages to check exact wording.

It's precise but only covers recent pages.

BASED combines both: the summary (linear attention) gives global context, and the "flip back"
(sliding window) handles precise recall of recent information. Together, they match the quality
of full Transformer attention while being much more efficient for long sequences.

The Taylor expansion feature map phi(x) = [1, x, x*x/sqrt(2)] is a mathematical trick that
approximates the softmax kernel exp(q*k) using polynomial terms. The constant 1 captures baseline
attention, x captures first-order similarity, and x*x/sqrt(2) captures quadratic interactions.

## How It Works

BASED is a hybrid architecture that combines two attention mechanisms to get the best of both worlds:

The key insight is that linear attention is fast (O(n)) but struggles with recall-intensive tasks
because the feature map approximation loses precision. Adding a small sliding window (w=64 or 128)
cheaply fixes this: the window handles precise local recall while linear attention covers global context.
The total complexity is O(n * (d + w)) which is still linear in sequence length for fixed window size.

**Reference:** Arora et al., "Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff", 2024.
https://arxiv.org/abs/2402.18668

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BASEDLayer(Int32,Int32,Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new BASED layer that combines linear attention with sliding window attention. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureExpansion` | Gets the feature expansion factor for the Taylor feature map. |
| `HeadDimension` | Gets the dimension per head. |
| `ModelDimension` | Gets the model dimension. |
| `NumHeads` | Gets the number of attention heads. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SupportsTraining` |  |
| `WindowSize` | Gets the sliding window size for local attention. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyFeatureMap([],Int32,[])` | Applies the Taylor expansion feature map: phi(x) = [x, x * x / sqrt(2)] (scaled by learned parameters). |
| `CombineAttentionOutputs(Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Combines linear attention and window attention outputs using learned per-head mixing gate. |
| `Forward(Tensor<>)` |  |
| `GetFeatureMapScale` | Gets the feature map scale parameters for external inspection. |
| `GetLinearQueryWeights` | Gets the linear attention query weights for external inspection. |
| `GetOutputProjectionWeights` | Gets the output projection weights for external inspection. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `GetWindowQueryWeights` | Gets the window attention query weights for external inspection. |
| `LinearAttentionForward(Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Linear attention forward pass using the causal linear attention recurrence with Taylor feature map. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `SlidingWindowAttentionForward(Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Sliding window causal attention: standard softmax attention restricted to a local window. |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

