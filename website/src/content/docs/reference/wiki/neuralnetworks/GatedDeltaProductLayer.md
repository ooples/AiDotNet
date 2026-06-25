---
title: "GatedDeltaProductLayer<T>"
description: "Implements the Gated DeltaProduct layer from \"DeltaProduct: Increasing the Expressivity of DeltaNet Through Products of Householders\" (Siems et al., 2025)."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the Gated DeltaProduct layer from "DeltaProduct: Increasing the Expressivity of DeltaNet
Through Products of Householders" (Siems et al., 2025).

## For Beginners

Gated DeltaProduct is the fully-featured version that combines
everything from both GatedDeltaNet and DeltaProduct.

Think of managing a library:

- Alpha (forget gate): "Remove 10% of books from each shelf" (uniform fading)
- Householder product: "Rearrange remaining books by topic instead of author" (rotation)
- Beta (write gate): "How many new books to add to the collection" (write strength)
- Output gate: "Which shelves to show to the current visitor" (output filtering)

Without gating (plain DeltaProduct): you can rearrange and add books, but can't thin out
the collection. Without Householder (plain GatedDeltaNet): you can thin out and add, but
can't rearrange. Gated DeltaProduct does all four operations at each step.

This makes it the most expressive variant in the DeltaNet family, at the cost of slightly
more compute per step.

## How It Works

Gated DeltaProduct combines the Householder product state transitions of DeltaProduct with
output gating (similar to how GatedDeltaNet adds gating to DeltaNet). The forget gate (alpha)
provides an additional scalar decay before the Householder rotation, and the output gate
controls information flow to the final output.

The architecture:

The combination of gating and Householder products provides maximum expressivity:

- Alpha gate: Controls how much old state to retain (like GatedDeltaNet)
- Householder product: ROTATES the retained state (like DeltaProduct)
- Beta gate: Controls how strongly to write new information
- Output gate: Controls what to expose from the state

This means the model can: fade old information (alpha), rearrange what remains (Householder),
write new information selectively (beta), and filter what to output (gate).

**Reference:** Siems et al., "DeltaProduct: Increasing the Expressivity of DeltaNet Through
Products of Householders", 2025. https://arxiv.org/abs/2502.10297

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GatedDeltaProductLayer(Int32,Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new Gated DeltaProduct layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HeadDimension` | Gets the dimension per head. |
| `ModelDimension` | Gets the model dimension. |
| `NumHeads` | Gets the number of heads. |
| `NumHouseholders` | Gets the number of Householder reflections per timestep. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AccumulateHouseholderWeightGradients(Tensor<>,Tensor<>,Int32,Int32)` | Accumulates Householder weight gradients from per-position gradients. |
| `ComputeHouseholderInputGradient(Tensor<>,Int32,Int32)` | Computes input gradient contribution from Householder vectors. |
| `ComputeHouseholderVectors(Tensor<>,Int32,Int32)` | Computes Householder vectors from input for each timestep. |
| `Forward(Tensor<>)` |  |
| `GatedDeltaProductRecurrence(Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Gated DeltaProduct recurrence: S_t = alpha_t * H_t * S_{t-1} + beta_t * v_t * k_t^T O_t = S_t * q_t |
| `GetHouseholderWeights` | Gets the Householder projection weights for external inspection. |
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

