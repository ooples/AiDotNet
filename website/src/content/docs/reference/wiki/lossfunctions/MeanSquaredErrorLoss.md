---
title: "MeanSquaredErrorLoss<T>"
description: "Implements the Mean Squared Error (MSE) loss function."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LossFunctions`

Implements the Mean Squared Error (MSE) loss function.

## For Beginners

Mean Squared Error is one of the most common loss functions used in regression problems.
It measures the average squared difference between predicted and actual values.

The formula is: MSE = (1/n) * ?(predicted - actual)²

MSE has these key properties:

- It heavily penalizes large errors due to the squaring operation
- It treats all data points equally
- It's differentiable everywhere, making it suitable for gradient-based optimization
- It's always positive, with perfect predictions giving a value of zero

MSE is ideal for problems where:

- You're predicting continuous values (like prices, temperatures, etc.)
- Outliers should be given extra attention (due to the squaring)
- The prediction errors follow a normal distribution

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new MeanSquaredErrorLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"MeanSquaredErrorLoss = {value:F4}");
```

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDerivative(Vector<>,Vector<>)` | Calculates the derivative of the Mean Squared Error loss function. |
| `CalculateLoss(Vector<>,Vector<>)` | Calculates the Mean Squared Error between predicted and actual values. |
| `CalculateLossAndGradientGpu(Tensor<>,Tensor<>)` | Calculates both MSE loss and gradient on GPU in a single efficient pass. |
| `ComputeTapeLoss(Tensor<>,Tensor<>)` |  |

