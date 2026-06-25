---
title: "HyperbolicLinearLayer<T>"
description: "Represents a fully connected layer operating in hyperbolic (Poincare ball) space."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a fully connected layer operating in hyperbolic (Poincare ball) space.

## For Beginners

This layer works in hyperbolic space instead of flat Euclidean space.

Benefits of hyperbolic layers:

- Naturally represents hierarchical data (trees, graphs, taxonomies)
- Can embed large hierarchies with low distortion
- Fewer dimensions needed for complex hierarchical structures

The layer uses the Poincare ball model where all points are inside a unit ball.
Points near the center are "higher" in the hierarchy, points near the edge are "lower".

## How It Works

A hyperbolic linear layer performs linear transformations in hyperbolic space using
Mobius operations. This is particularly useful for learning hierarchical representations
where tree-like structures need to be embedded.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HyperbolicLinearLayer(Int32,Int32,Double,IActivationFunction<>)` | Initializes a new instance of the HyperbolicLinearLayer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputFeatures` | Gets the number of input features. |
| `OutputFeatures` | Gets the number of output features. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | Gets whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyActivationToComputationNode(ComputationNode<>)` | Applies the activation function to a computation node. |
| `ClearGradients` | Resets the internal state of the layer. |
| `CreateOriginVector(Int32)` | Creates a vector at the origin of hyperbolic space. |
| `DistanceFromOriginGrad(Double[],Double,Double)` | Gradient of Poincaré distance from origin: d(0,y) = (2/√c)·arctanh(√c·\|\|y\|\|) ∂d/∂y_i = 2·y_i / (\|\|y\|\|·(1 - c·\|\|y\|\|²)) |
| `ExpMapFromOrigin(Double[],Double,Double)` | Exponential map from origin: exp_0(v) = tanh(√c·\|\|v\|\|)/(√c·\|\|v\|\|) · v |
| `ExpMapFromOriginGrad(Double[],Double[],Double,Double)` | Gradient of exp_0(v) w.r.t. |
| `Forward(Tensor<>)` | Performs the forward pass through the layer. |
| `GetMetadata` |  |
| `GetParameterGradients` |  |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeParameters` | Initializes weights using Xavier/Glorot initialization adapted for hyperbolic space. |
| `MobiusAddDouble(Double[],Double[],Double)` | Möbius addition: x ⊕ y = ((1+2c⟨x,y⟩+c\|\|y\|\|²)x + (1-c\|\|x\|\|²)y) / D where D = 1+2c⟨x,y⟩+c²\|\|x\|\|²\|\|y\|\|² |
| `MobiusAddGrad(Double[],Double[],Double[],Double)` | Gradient of Möbius addition z = x ⊕ y w.r.t. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` | Sets all trainable parameters from a single vector. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_biases` | Scalar bias per output feature — added to the Poincaré ball coordinates after Möbius matrix-vector multiplication. |
| `_biasesGradient` | Gradient for biases, stored during backward pass. |
| `_curvature` | The curvature of the hyperbolic space (negative for hyperbolic). |
| `_lastInput` | Stored input from forward pass for backpropagation. |
| `_lastOutput` | Stored pre-activation output for gradient computation. |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |
| `_weights` | Weight matrix stored in tangent space at the origin. |
| `_weightsGradient` | Gradient for weights, stored during backward pass. |
| `_weightsTCache` | Cached W^T — invalidated when weights change. |

