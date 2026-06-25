---
title: "MixtureOfMambaLayer<T>"
description: "Implements the Mixture-of-Mamba layer from Jiang et al., 2025 (arXiv:2501.16295)."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the Mixture-of-Mamba layer from Jiang et al., 2025 (arXiv:2501.16295).

## For Beginners

Mixture-of-Mamba is like having a team of specialized assistants
instead of one generalist.

Imagine you're processing a long document:

- You have 8 assistants (experts), each good at different things:
- Expert 1 might be great at tracking names and entities
- Expert 2 might excel at following numerical patterns
- Expert 3 might specialize in temporal/causal relationships
- etc.
- For each word, a "router" decides which 2 assistants (top-K=2) should handle it
- Each chosen assistant processes the word using their own specialized memory (SSM state)
- The final answer combines the two assistants' outputs, weighted by confidence

This is more efficient than one giant assistant because:

- Each expert can specialize in different patterns (divide and conquer)
- Only 2 out of 8 experts run per token (sparse computation saves resources)
- Different tokens automatically get routed to the most relevant experts

The "Mamba" part refers to the selective scan mechanism each expert uses, which is a
very efficient way to process sequences with learned, input-dependent state transitions.

## How It Works

Mixture-of-Mamba combines Mamba's selective state space model (SSM) with Mixture of Experts (MoE)
sparsity. Instead of a single monolithic SSM, the layer maintains multiple "expert" SSM blocks,
each with its own A, B, C, D parameters. A learned router selects the top-K experts for each
token, allowing different tokens to be processed by different specialized SSM pathways.

The architecture:

The key insight is that different types of sequential patterns benefit from different SSM
dynamics. By routing tokens to specialized experts, the model can learn distinct temporal
patterns without interference. For example, one expert might specialize in short-range
dependencies (fast-decaying A), while another handles long-range dependencies (slow-decaying A).
The sparse routing means only top-K experts are active per token, maintaining efficiency.

**Reference:** Jiang et al., "Mixture-of-Mamba: Enhancing Multi-Modal State-Space Models with Mixture of Experts", 2025.
https://arxiv.org/abs/2501.16295

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MixtureOfMambaLayer(Int32,Int32,Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new Mixture-of-Mamba layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelDimension` | Gets the model dimension. |
| `NumExperts` | Gets the number of experts. |
| `ParameterCount` |  |
| `StateDimension` | Gets the SSM state dimension per expert. |
| `SupportsTraining` |  |
| `TopK` | Gets the number of active experts per token (top-K). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeTopKSoftmax(Tensor<>,Tensor<>,Int32[0:,0:],Int32)` | Computes softmax over experts and selects top-K for each token. |
| `Forward(Tensor<>)` |  |
| `GetExpertA` | Gets the expert A (state transition) parameters for external inspection. |
| `GetOutputProjectionWeights` | Gets the output projection weights for external inspection. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetRouterWeights` | Gets the router weights for external inspection. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

