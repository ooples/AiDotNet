---
title: "MultiLatentAttentionLayer<T>"
description: "Implements the Multi-Latent Attention (MLA) layer from DeepSeek-V2 (Aixin Liu et al., 2024)."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the Multi-Latent Attention (MLA) layer from DeepSeek-V2 (Aixin Liu et al., 2024).

## For Beginners

MLA is a memory-efficient attention mechanism used in DeepSeek-V2.

Standard attention is like keeping a complete notebook for every word you've read:

- You write the full Key (what this word is about) and Value (what it says) for every word
- When answering a question (Query), you look up all Key-Value pairs
- For a 100K word document, that's a LOT of notebook pages

MLA is like keeping compressed sticky notes instead:

- For each word, you write just a short summary (the latent c_t)
- When you need the full Key or Value, you expand the summary on the fly
- You still get nearly the same answer quality, but use far less paper

The "Multi-Latent" name comes from having multiple attention heads, each with its own
latent compression. This is what enables DeepSeek-V2 to handle very long contexts efficiently.

## How It Works

Multi-Latent Attention compresses the KV cache via low-rank factorization, dramatically reducing
memory usage during inference. Instead of caching full-dimensional K and V for every token, MLA
caches a much smaller latent vector c_t, from which K and V are reconstructed on the fly.

The architecture:

The key insight is KV cache compression: during inference, you only need to store c_t (latentDim per token)
instead of K and V (2 * modelDim per token). When latentDim = modelDim/4, this yields an 8x reduction
in KV cache memory, which is the primary bottleneck for long-context LLM serving.

**Reference:** Aixin Liu et al., "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts
Language Model", 2024. https://arxiv.org/abs/2405.04434

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultiLatentAttentionLayer(Int32,Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new Multi-Latent Attention (MLA) layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HeadDimension` | Gets the dimension per head. |
| `LatentDimension` | Gets the latent dimension for KV cache compression. |
| `ModelDimension` | Gets the model dimension. |
| `NumHeads` | Gets the number of attention heads. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CausalMultiHeadAttention(Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Causal multi-head attention: softmax(Q*K^T / sqrt(d_k)) * V with causal mask. |
| `Forward(Tensor<>)` |  |
| `GetCompressionWeights` | Gets the compression weights for external inspection. |
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

