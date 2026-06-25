---
title: "DeltaNetLayer<T>"
description: "Implements the DeltaNet layer from \"Linear Transformers with Learnable Kernel Functions\" (Yang et al., 2024)."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the DeltaNet layer from "Linear Transformers with Learnable Kernel Functions" (Yang et al., 2024).

## For Beginners

DeltaNet is a simpler, foundational variant of GatedDeltaNet.

Think of the state matrix S as a "lookup table" that maps keys to values:

- Linear attention: "Just add every key-value pair to the table" -> entries pile up, old ones never corrected
- Delta rule: "Before adding, check what S already predicts for this key. Only write the correction."

This is like the difference between:

- Writing every flashcard answer on top of the previous one (linear attention -> messy)
- Erasing only the wrong part and writing the correction (delta rule -> clean)

The beta parameter controls how much of the correction to actually apply:

- beta near 0: "I trust the existing memory, barely update"
- beta near 1: "Fully overwrite whatever was stored for this key"

Because there is no forget gate (alpha) or output gate, this model is simpler and faster than
GatedDeltaNet, but may underperform on tasks that require selective forgetting or output gating.

## How It Works

DeltaNet applies the delta rule to linear attention, replacing the naive accumulation of key-value
outer products with an error-corrective update. This produces a linear-complexity recurrent model
that is significantly more expressive than standard linear attention.

The architecture:

This is the "ungated" version of GatedDeltaNet: there is no alpha forget gate, no output gate,
and no short convolution. The state is purely additive (S_{t-1} carries forward with weight 1),
and the only learned control is beta which modulates how strongly corrections are written.

The delta rule update is key: instead of blindly accumulating K*V outer products (like linear
attention), it computes the error (V - S*K) first and updates accordingly. This is exactly the
Widrow-Hoff / delta rule from neural network learning theory, applied to a fast weight matrix
at each timestep.

**Reference:** Yang et al., "Linear Transformers with Learnable Kernel Functions", 2024.
https://arxiv.org/abs/2406.06484

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeltaNetLayer(Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new DeltaNet layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HeadDimension` | Gets the dimension per head (modelDimension / numHeads). |
| `ModelDimension` | Gets the model dimension (d_model). |
| `NumHeads` | Gets the number of attention heads. |
| `ParameterCount` | Gets the total number of trainable parameters in this layer. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateOnesLike(Tensor<>)` | Creates a tensor of ones with the same shape as the template tensor. |
| `DeltaRuleForward(Tensor<>,Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Delta rule forward: error-corrective fast weight update without gating. |
| `Forward(Tensor<>)` |  |
| `GetAllTensors` | Returns all trainable parameter tensors in a consistent order for serialization. |
| `GetOutputProjectionWeights` | Gets the output projection weights for external inspection or analysis. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetQueryWeights` | Gets the query weights for external inspection or analysis. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeParameters` | Initializes all trainable parameters using Xavier/Glorot initialization for weight matrices and appropriate constants for biases. |
| `InitializeTensor2D(Tensor<>)` | Applies Xavier/Glorot uniform initialization to a 2D weight tensor. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

