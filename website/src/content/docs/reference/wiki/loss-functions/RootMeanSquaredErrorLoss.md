---
title: "RootMeanSquaredErrorLoss"
description: "Implements the Root Mean Squared Error (RMSE) loss function."
section: "Reference"
---

_Loss Functions_

Implements the Root Mean Squared Error (RMSE) loss function.

## How It Works

RMSE measures the square root of the average squared differences between predicted and actual values. It is particularly useful for regression problems and gives more weight to larger errors. Formula: RMSE = sqrt(mean((predicted - actual)^2)) The derivative with respect to predicted values is: d(RMSE)/d(predicted) = (predicted - actual) / (n * RMSE) where n is the number of samples and RMSE is the loss value. This implementation leverages the existing StatisticsHelper.CalculateRootMeanSquaredError() method for efficient and consistent calculation across the library.

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new RootMeanSquaredErrorLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"RootMeanSquaredErrorLoss = {value:F4}");
```

