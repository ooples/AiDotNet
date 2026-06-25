---
title: "HGRNLayer<T>"
description: "Implements the Hierarchically Gated Recurrent Neural Network (HGRN) layer from NeurIPS 2023."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the Hierarchically Gated Recurrent Neural Network (HGRN) layer from NeurIPS 2023.

## For Beginners

HGRN is like a stack of simple "memory filters" that work at different speeds.

Imagine you're listening to music:

- One filter (lower layer, high forget bias) remembers the overall melody for a long time

-- it barely forgets anything, holding onto the big picture

- Another filter (upper layer, low forget bias) tracks the current beat and rhythm

-- it quickly forgets old beats to focus on what's happening right now

- Together, they understand both the long melody AND the short rhythm simultaneously

Each filter is extremely simple: at every step, it just does:
new_memory = (forget_amount * old_memory) + (input_amount * new_input)

This is much simpler than Transformers (which compare every position to every other position)
or even Mamba (which maintains matrix-valued states). Yet by stacking these simple filters
with different forgetting speeds, HGRN achieves competitive performance with linear O(n)
complexity -- it processes a sequence of length n in time proportional to n, not n^2.

## How It Works

HGRN uses hierarchical gating to achieve multi-scale temporal processing with linear O(n) complexity.
Each layer performs a simple element-wise gated recurrence:

The "hierarchical" aspect is the key insight: when stacking multiple HGRN layers, each layer
uses a different forget gate bias. Lower layers use higher forget gate bias values (sigmoid output
closer to 1), creating slow-decaying memory that captures long-range dependencies. Upper layers
use lower bias values (sigmoid output closer to 0), creating fast-decaying memory for short-range
local patterns. This naturally creates a multi-scale temporal hierarchy where different layers
specialize in different time scales, similar to how wavelets decompose signals at multiple
resolutions.

Unlike GatedDeltaNet or Mamba which maintain matrix-valued states, HGRN maintains only a
vector-valued hidden state h_t (same dimension as the input). This makes it extremely
memory-efficient and fast, while the hierarchical gating compensates for the simpler state
by distributing temporal modeling across layers.

**Reference:** Qin et al., "Hierarchically Gated Recurrent Neural Network for Sequence Modeling", NeurIPS 2023.
https://arxiv.org/abs/2311.04823

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HGRNLayer(Int32,Int32,Double,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new HGRN (Hierarchically Gated Recurrent Neural Network) layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ForgetBias` | Gets the forget gate bias value used for hierarchical gating. |
| `ModelDimension` | Gets the model dimension (input/output width). |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` |  |
| `GatedRecurrenceForward(Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Implements the element-wise gated recurrence: h_t = f_t * h_{t-1} + i_t * x_t. |
| `GetForgetGateWeights` | Gets the forget gate weights for external inspection. |
| `GetOutputProjectionWeights` | Gets the output projection weights for external inspection. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

