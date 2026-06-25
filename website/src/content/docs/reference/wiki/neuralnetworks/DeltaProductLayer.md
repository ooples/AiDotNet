---
title: "DeltaProductLayer<T>"
description: "Implements the DeltaProduct layer from \"DeltaProduct: Increasing the Expressivity of DeltaNet Through Products of Householders\" (Siems et al., 2025)."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the DeltaProduct layer from "DeltaProduct: Increasing the Expressivity of DeltaNet
Through Products of Householders" (Siems et al., 2025).

## For Beginners

DeltaProduct improves DeltaNet by adding "rotations" to memory updates.

Think of the state matrix as a whiteboard of notes:

- DeltaNet: Before writing new notes, you can only FADE the old notes (scalar alpha)
- DeltaProduct: Before writing, you can REARRANGE the old notes (rotate/reflect them)

A Householder reflection is like flipping the whiteboard across a mirror:

- One reflection can flip everything across one axis
- Two reflections can rotate everything by any angle
- M reflections can do any rearrangement that preserves the "length" of your notes

This means DeltaProduct can:

- Move old information to make room for new information (rotation)
- Flip the organization of information (reflection)
- All while preserving the total amount of stored information (orthogonality)

The result is a more expressive model that better manages what it remembers and forgets.

## How It Works

DeltaProduct extends DeltaNet by replacing the scalar forget gate with a product of Householder
reflections for state transitions. A Householder reflection H = I - 2*v*v^T/||v||^2 is an
orthogonal transformation that reflects vectors across the hyperplane perpendicular to v.
By composing multiple Householder reflections, DeltaProduct can represent any orthogonal
transformation of the state, making the state transition far more expressive than a scalar decay.

The architecture:

The key insight: in standard DeltaNet, the state transition is S_t = alpha * S_{t-1} + ...,
where alpha is just a scalar decay. This limits how the state can evolve -- old information
can only fade uniformly. With Householder products, the state can be ROTATED and REFLECTED
before the new write, preserving information while restructuring it. Since any orthogonal
matrix can be decomposed into Householder reflections, M reflections can express any rotation
in the head dimension space when M >= headDim.

**Reference:** Siems et al., "DeltaProduct: Increasing the Expressivity of DeltaNet Through
Products of Householders", 2025. https://arxiv.org/abs/2502.10297

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeltaProductLayer(Int32,Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new DeltaProduct layer. |

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
| `ApplyHouseholderProduct(Tensor<>,Tensor<>,Int32,Int32,Int32)` | Applies the product of M Householder reflections to a matrix. |
| `ComputeHouseholderInputGradient(Tensor<>,Int32,Int32)` | Computes input gradient contribution from Householder vectors. |
| `ComputeHouseholderVectors(Tensor<>,Int32,Int32)` | Computes Householder vectors from input for each timestep. |
| `DeltaProductRecurrence(Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | DeltaProduct recurrence: state update with Householder product transitions. |
| `Forward(Tensor<>)` |  |
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

