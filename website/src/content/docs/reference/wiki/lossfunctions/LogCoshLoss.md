---
title: "LogCoshLoss<T>"
description: "Implements the Log-Cosh loss function, a smooth approximation of Mean Absolute Error."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LossFunctions`

Implements the Log-Cosh loss function, a smooth approximation of Mean Absolute Error.

## For Beginners

Log-Cosh loss is a smooth approximation of the Mean Absolute Error.
It calculates the logarithm of the hyperbolic cosine of the difference between predictions and actual values.

This loss function has several desirable properties:

- It's smooth everywhere (unlike Huber loss which has a point where its derivative is not continuous)
- It's less affected by outliers than Mean Squared Error
- It behaves like Mean Squared Error for small errors and Mean Absolute Error for large errors
- Its derivatives are well-defined and bounded, which helps prevent gradient explosions during training

Log-Cosh loss is particularly useful for regression problems where:

- You want the smoothness of MSE but with better robustness to outliers
- You need stable gradients for model training
- You want a compromise between MSE and MAE

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new LogCoshLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"LogCoshLoss = {value:F4}");
```

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDerivative(Vector<>,Vector<>)` | Calculates the derivative of the Log-Cosh loss function. |
| `CalculateLoss(Vector<>,Vector<>)` | Calculates the Log-Cosh loss between predicted and actual values. |
| `CalculateLossAndGradientGpu(Tensor<>,Tensor<>)` | Calculates both Log-Cosh loss and gradient on GPU in a single efficient pass. |
| `ComputeTapeLoss(Tensor<>,Tensor<>)` |  |

