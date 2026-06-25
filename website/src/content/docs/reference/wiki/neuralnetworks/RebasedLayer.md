---
title: "RebasedLayer<T>"
description: "Implements the ReBased linear attention layer from \"Linearizing Large Language Models\" (Bick et al., 2024)."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the ReBased linear attention layer from "Linearizing Large Language Models" (Bick et al., 2024).

## For Beginners

ReBased is a way to make attention much faster for long sequences.

Standard attention compares every token with every other token, which takes O(n^2) time.
ReBased replaces this with a clever trick:

- Instead of comparing all pairs, it maintains a running "summary" matrix S
- At each step, it updates the summary with the current key-value pair
- To compute output, it just multiplies the summary by the query

The "squared ReLU" feature map is what makes this work well:

- ReLU(x) = max(0, x) keeps only positive values
- Squaring these positive values (ReLU(x)^2) creates quadratic interactions
- This is a much better approximation of what softmax attention does internally
- Normalizing by the L2 norm keeps values from exploding

Think of it like summarizing a book: instead of re-reading the whole book for each question
(softmax attention), you maintain a running summary (the state matrix S) and just look up
the answer from the summary (multiply by query). The squared ReLU trick makes the summary
much more informative than simpler approaches.

## How It Works

ReBased replaces standard softmax attention with a linear attention mechanism that uses improved
polynomial feature maps. The key idea is to use squared ReLU features to approximate softmax attention,
achieving sub-quadratic complexity while maintaining competitive quality.

The architecture:

The squared ReLU kernel is central: compared to first-order maps like phi(x) = ELU(x) + 1,
the quadratic term ReLU(x)^2 creates richer feature interactions that more closely approximate
the exponential kernel exp(q*k) used in softmax attention. The L2 normalization prevents the
squared features from growing too large, stabilizing training.

**Reference:** Bick et al., "Linearizing Large Language Models", 2024.
https://arxiv.org/abs/2402.10644

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RebasedLayer(Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new ReBased linear attention layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HeadDimension` | Gets the dimension per head. |
| `ModelDimension` | Gets the model dimension. |
| `NumHeads` | Gets the number of attention heads. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplySquaredReluFeatureMap([],[])` | Applies the squared ReLU feature map with L2 normalization: phi(x) = ReLU(x)^2 / \|\|ReLU(x)^2\|\|. |
| `Forward(Tensor<>)` |  |
| `GetOutputProjectionWeights` | Gets the output projection weights for external inspection. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetQueryWeights` | Gets the query weights for external inspection. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `LinearAttentionBackward(Tensor<>,Int32,Int32)` | Backward pass for causal linear attention with squared ReLU feature maps. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `SquaredReluLinearAttentionForward(Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Causal linear attention with squared ReLU feature maps. |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

