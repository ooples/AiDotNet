---
title: "HuberLoss"
description: "Implements the Huber loss function, which combines properties of both MSE and MAE."
section: "Reference"
---

_Loss Functions_

Implements the Huber loss function, which combines properties of both MSE and MAE.

## For Beginners

Huber loss combines the best properties of Mean Squared Error and Mean Absolute Error. The formula is: - For errors smaller than delta: 0.5 * error² - For errors larger than delta: delta * (|error| - 0.5 * delta) Where "error" is the difference between predicted and actual values. Key properties: - For small errors, it behaves like MSE (quadratic/squared behavior) - For large errors, it behaves like MAE (linear behavior) - Less sensitive to outliers than MSE, but still provides smooth gradients - The delta parameter controls the transition point between quadratic and linear regions Huber loss is ideal for regression problems where: - You want to balance between MSE and MAE - Your data might contain outliers - You need stable gradients for learning The delta parameter lets you control the definition of an "outlier" - errors larger than delta are treated as outliers and handled using the more robust linear function.

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new HuberLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"HuberLoss = {value:F4}");
```

