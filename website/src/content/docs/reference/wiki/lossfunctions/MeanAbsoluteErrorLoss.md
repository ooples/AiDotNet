---
title: "MeanAbsoluteErrorLoss<T>"
description: "Implements the Mean Absolute Error (MAE) loss function."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LossFunctions`

Implements the Mean Absolute Error (MAE) loss function.

## For Beginners

Mean Absolute Error measures the average absolute difference between 
predicted and actual values.

The formula is: MAE = (1/n) * ?|predicted - actual|

MAE has these key properties:

- It treats all errors linearly (unlike MSE which squares errors)
- It's less sensitive to outliers than MSE
- It's simple to understand as the average magnitude of errors
- It's always positive, with perfect predictions giving a value of zero

MAE is ideal for problems where:

- You're predicting continuous values
- You want all errors to be treated equally (not emphasizing large errors)
- The prediction errors follow a Laplace distribution
- Outliers should not have a disproportionate influence on the model

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new MeanAbsoluteErrorLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"MeanAbsoluteErrorLoss = {value:F4}");
```

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDerivative(Vector<>,Vector<>)` | Calculates the derivative of the Mean Absolute Error loss function. |
| `CalculateLoss(Vector<>,Vector<>)` | Calculates the Mean Absolute Error between predicted and actual values. |
| `CalculateLossAndGradientGpu(Tensor<>,Tensor<>)` | Calculates both MAE loss and gradient on GPU in a single efficient pass. |
| `ComputeTapeLoss(Tensor<>,Tensor<>)` |  |

