---
title: "SquaredHingeLoss<T>"
description: "Implements the Squared Hinge Loss function for binary classification problems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LossFunctions`

Implements the Squared Hinge Loss function for binary classification problems.

## For Beginners

Squared Hinge Loss is a variation of the Hinge Loss used in Support Vector Machines (SVMs)
that applies a squared penalty to incorrectly classified examples.

The formula is: max(0, 1 - y*f(x))²
Where:

- y is the true label (usually -1 or 1)
- f(x) is the model's prediction

Key properties:

- It heavily penalizes predictions that are incorrect or not confident enough
- The quadratic nature creates a smoother loss surface compared to regular Hinge Loss
- It has a continuous derivative everywhere, which can make optimization easier
- It's zero when predictions are correct and confident (y*f(x) = 1)

Squared Hinge Loss is particularly useful for:

- Binary classification problems
- Support Vector Machines
- Any situation where smoother gradients are beneficial for optimization

Compared to regular Hinge Loss, it penalizes violations more severely due to the squaring operation.

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new SquaredHingeLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"SquaredHingeLoss = {value:F4}");
```

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDerivative(Vector<>,Vector<>)` | Calculates the derivative of the Squared Hinge Loss function. |
| `CalculateLoss(Vector<>,Vector<>)` | Calculates the Squared Hinge Loss between predicted and actual values. |
| `CalculateLossAndGradientGpu(Tensor<>,Tensor<>)` | Calculates both Squared Hinge loss and gradient on GPU in a single efficient pass. |
| `ComputeTapeLoss(Tensor<>,Tensor<>)` |  |

