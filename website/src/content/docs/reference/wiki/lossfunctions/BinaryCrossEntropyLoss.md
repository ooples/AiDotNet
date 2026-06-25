---
title: "BinaryCrossEntropyLoss<T>"
description: "Implements the Binary Cross Entropy loss function for binary classification problems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LossFunctions`

Implements the Binary Cross Entropy loss function for binary classification problems.

## For Beginners

Binary Cross Entropy is used when classifying data into two categories,
such as spam/not-spam, positive/negative sentiment, or disease/no-disease.

The formula is: BCE = -(1/n) * ?[actual * log(predicted) + (1-actual) * log(1-predicted)]

It measures how well predicted probabilities match actual binary outcomes:

- When the actual value is 1, it evaluates how close the prediction is to 1
- When the actual value is 0, it evaluates how close the prediction is to 0

Key properties:

- Predicted values must be probabilities (between 0 and 1)
- Actual values are typically 0 or 1 (binary labels)
- It heavily penalizes confident mistakes (predicting 0.01 when the true value is 1)
- It's the preferred loss function for binary classification problems

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new BinaryCrossEntropyLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"BinaryCrossEntropyLoss = {value:F4}");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BinaryCrossEntropyLoss` | Initializes a new instance of the BinaryCrossEntropyLoss class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDerivative(Vector<>,Vector<>)` | Calculates the derivative of the Binary Cross Entropy loss function. |
| `CalculateLoss(Vector<>,Vector<>)` | Calculates the Binary Cross Entropy loss between predicted and actual values. |
| `CalculateLossAndGradientGpu(Tensor<>,Tensor<>)` | Calculates both BCE loss and gradient on GPU in a single efficient pass. |
| `ComputeTapeLoss(Tensor<>,Tensor<>)` |  |

