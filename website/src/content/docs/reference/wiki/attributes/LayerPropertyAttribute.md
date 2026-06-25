---
title: "LayerPropertyAttribute"
description: "Declares architectural properties of a neural network layer for cataloging and test generation."
section: "API Reference"
---

`Attributes` · `AiDotNet.Attributes`

Declares architectural properties of a neural network layer for cataloging and test generation.

## Properties

| Property | Summary |
|:-----|:--------|
| `ApiShape` | The Forward method signature shape this layer uses. |
| `ChangesShape` | Whether the layer changes the shape of its input. |
| `Cost` | Relative computational cost. |
| `ExpectedInputRank` | Expected input rank (number of dimensions). |
| `HasTrainingMode` | Whether the layer behaves differently during training vs inference (e.g., Dropout, BatchNorm). |
| `IsStateful` | Whether the layer is stateful across forward passes (RNN hidden state, running mean). |
| `IsTrainable` | Whether the layer has trainable parameters (weights/biases). |
| `NormalizesInput` | Whether the layer normalizes input (LayerNorm, BatchNorm, etc.) so uniform-value inputs produce identical outputs regardless of the actual value. |
| `ProducesNonFiniteOutput` | Whether the layer legitimately produces ±Infinity values in its Forward output, by design rather than by numerical instability. |
| `SupportsBackpropagation` | Whether the layer supports backpropagation gradient computation. |
| `SupportsInPlace` | Whether the layer supports in-place operation (output can alias input). |
| `TestConstructorArgs` | Constructor arguments as a comma-separated string for auto-generated tests. |
| `TestInputShape` | The input tensor shape to use for auto-generated tests, as a comma-separated string. |
| `TestSetupCode` | C# code to call on the layer after construction to set up graph data, adjacency matrices, or other domain-specific initialization required before Forward. |
| `TrainsViaCustomLoss` | Whether the layer trains via a custom loss rather than gradients flowing from its Forward output. |
| `UsesSurrogateGradient` | Whether the layer uses surrogate gradients (e.g., spiking neural networks use a smooth approximation for the non-differentiable Heaviside step function). |

