---
title: "HGRN2Layer<T>"
description: "Implements the HGRN2 layer from \"HGRN2: Gated Linear RNNs with State Expansion\" (Qin et al., 2024)."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the HGRN2 layer from "HGRN2: Gated Linear RNNs with State Expansion" (Qin et al., 2024).

## For Beginners

HGRN2 is a sequence model that processes tokens one at a time while
maintaining a "memory matrix" for each attention head.

Think of each head's state matrix as a small notebook:

- At each step, the model writes a new "entry" (the outer product k*v^T) into the notebook
- The forget gate g_t controls how much the old notes fade: g=1 means perfect memory, g=0 means

forget everything from before

- To produce output, the model "looks up" information by multiplying the notebook by a query vector

Compared to a standard Transformer:

- Transformers re-read ALL previous tokens at every step (O(n^2) cost)
- HGRN2 compresses all history into a fixed-size matrix (O(n) cost, constant memory)
- The matrix state is much richer than a simple vector (like LSTM/GRU), letting HGRN2

remember more complex patterns

HGRN2 achieves competitive performance with Transformers on language modeling benchmarks while
being significantly more efficient for long sequences.

## How It Works

HGRN2 extends HGRN (Hierarchical Gated Recurrent Network) with "state expansion", bridging the gap
between element-wise gated recurrences (vector state) and linear attention (matrix state). Instead of
maintaining a hidden vector h_t as in HGRN, HGRN2 maintains a hidden matrix S_t of shape
[head_dim x head_dim] per head, enabling richer state representations.

The architecture:

The key insight of "state expansion" is that using an outer product k*v^T to build the state matrix
gives each head a rank-1 update per step. Over time the state accumulates a low-rank approximation
of the key-value associations, similar to how linear attention accumulates K^T V. The crucial
difference from linear attention is the forget gate g_t, which prevents unbounded state growth and
allows the model to selectively discard old information.

This bridges two extremes:

- HGRN (vector state): S_t is a vector, updated element-wise. Capacity limited by head_dim.
- Linear attention (matrix state): S_t = S_{t-1} + k_t * v_t^T, no forgetting. Unbounded growth.
- HGRN2 (gated matrix state): S_t = g_t * S_{t-1} + k_t * v_t^T. Best of both worlds.

**Reference:** Qin et al., "HGRN2: Gated Linear RNNs with State Expansion", 2024.
https://arxiv.org/abs/2404.07904

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HGRN2Layer(Int32,Int32,Int32,Double,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new HGRN2 layer with state expansion. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HeadDimension` | Gets the dimension per head (d_model / numHeads). |
| `ModelDimension` | Gets the model dimension (d_model). |
| `NumHeads` | Gets the number of attention heads. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` |  |
| `GetOutputProjectionWeights` | Gets the output projection weights for external inspection. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetQueryWeights` | Gets the query weights for external inspection. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `OuterProductRecurrenceForward(Tensor<>,Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Outer-product gated recurrence forward pass. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

