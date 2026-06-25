---
title: "LogLinearAttentionLayer<T>"
description: "Implements the Log-Linear Attention layer from Zhang et al., 2025."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the Log-Linear Attention layer from Zhang et al., 2025.

## For Beginners

Think of this like a hierarchical note-taking system:

- Level 0 (seconds): Detailed notes from the last few moments
- Level 1 (minutes): Summaries of recent detailed notes
- Level 2 (hours): High-level summaries of minute-level notes
- Level 3 (days): Very compressed overviews

When you need to answer a question (query), you check all levels:
recent details for recent questions, older summaries for historical questions.

The total storage is O(L * d^2) where L = log(n) levels, instead of O(n * d^2) for
standard linear attention. This makes it much more memory-efficient for long sequences
while retaining more information than a single fixed-size state.

## How It Works

Standard linear attention maintains a hidden state that grows linearly with sequence length:
at each step, a new key-value outer product is added to the state matrix S. Over long sequences,
the accumulated state becomes noisy and the effective capacity is diluted, which is why linear
attention often underperforms softmax attention on long contexts.

Log-Linear Attention solves this by organizing the state into a **hierarchical multi-level structure**
where the total state size grows only logarithmically with sequence length. The key idea is periodic
compression: instead of keeping every update, lower-level states are periodically compressed and
promoted to higher levels.

The hierarchy works as follows:

Compression at each level uses a learned linear projection to reduce the accumulated state,
preserving the most important information while discarding noise. This is analogous to how
human memory works: recent events are stored in detail (Level 0), while older events are
remembered as compressed summaries (higher levels).

**Reference:** Zhang et al., "Log-Linear Attention", 2025.
https://arxiv.org/abs/2506.04761

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LogLinearAttentionLayer(Int32,Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new Log-Linear Attention layer with hierarchical state compression. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HeadDimension` | Gets the dimension per head. |
| `ModelDimension` | Gets the model dimension. |
| `NumHeads` | Gets the number of heads. |
| `NumLevels` | Gets the number of hierarchy levels. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` |  |
| `GetLevelMixWeights` | Gets the level mixing weights for external inspection. |
| `GetOutputProjectionWeights` | Gets the output projection weights for external inspection. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetQueryWeights` | Gets the query weights for external inspection. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `LogLinearForward(Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Log-linear forward: hierarchical state accumulation with periodic compression. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

