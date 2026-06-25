---
title: "PaTHAttentionLayer<T>"
description: "Implements the PaTH Attention (Positional-aware Transformer via Householder) layer from Mao et al., 2025 (arXiv:2505.16381)."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the PaTH Attention (Positional-aware Transformer via Householder) layer
from Mao et al., 2025 (arXiv:2505.16381).

## For Beginners

PaTH Attention is a new way to tell the model WHERE each token
is in the sequence, using geometry instead of adding numbers.

Traditional approaches add position information:

- "I am token 5" gets added as a number pattern to the token's embedding
- This can interfere with the token's meaning

PaTH instead REFLECTS the token's query/key vectors using a mirror unique to each position:

- Position 1 has mirror A, position 2 has mirror B, etc.
- Each mirror "rotates" the query/key differently based on position
- The attention score naturally captures both WHAT the token means AND WHERE it is

Think of it like a hall of mirrors:

- Each position has its own unique mirror (Householder reflection)
- When you look at a token through its position's mirror, you see a unique view
- Two tokens at different positions produce different reflections
- The similarity between reflections captures both content and position

This is mathematically cleaner because reflections preserve vector lengths (they just
change direction), whereas adding position numbers changes lengths and can distort meaning.

## How It Works

PaTH Attention replaces traditional positional encodings (sinusoidal, rotary, etc.) with
Householder reflections applied to queries and keys. Each position in the sequence has a
learned reflection vector p, and the corresponding Householder transform H = I - 2*p*p^T/||p||^2
is applied to Q and K before computing attention. This embeds positional information directly
into the geometry of the attention space rather than adding it as a bias.

The architecture:

The key insight is that Householder reflections are orthogonal transformations. Unlike additive
positional encodings that can distort the magnitude of embeddings, Householder reflections
preserve norms while encoding position. The attention score between two positions depends on
BOTH the content (Q, K) and the relative position (H_i vs H_j), but in a multiplicative way
that is more expressive than simple additive bias.

**Reference:** Mao et al., "PaTH Attention: Positional-aware Transformer via Householder", 2025.
https://arxiv.org/abs/2505.16381

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PaTHAttentionLayer(Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new PaTH Attention layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HeadDimension` | Gets the dimension per head. |
| `ModelDimension` | Gets the model dimension. |
| `NumHeads` | Gets the number of attention heads. |
| `ParameterCount` |  |
| `SequenceLength` | Gets the sequence length. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AccumulateHouseholderGradient([],[],Int32,Int32)` | Accumulates gradient of the Householder vector from one input vector path. |
| `ApplyHouseholderReflection([],Int32,Int32,[])` | Applies Householder reflection H = I - 2*p*p^T/\|\|p\|\|^2 to a vector x. |
| `ComputeAttention(Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Standard scaled dot-product attention using reflected Q and K. |
| `Forward(Tensor<>)` |  |
| `GetHouseholderVectors` | Gets the Householder vectors for external inspection. |
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

