---
title: "CrossEntropyLoss<T>"
description: "Implements the Cross Entropy loss function for multi-class classification problems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LossFunctions`

Implements the Cross Entropy loss function for multi-class classification problems.

## For Beginners

Cross-Entropy loss measures how different two probability distributions are.
It's commonly used for classification problems where the model outputs probabilities.

The formula is: -?(actual_i * log(predicted_i))

Key properties:

- Lower values indicate that the predicted distribution is closer to the actual distribution
- It encourages the model to be confident about correct predictions
- It heavily penalizes confident but incorrect predictions
- It's particularly suited for training classifiers

This loss function is often used in conjunction with the softmax activation function
in the output layer for multi-class classification problems.

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new CrossEntropyLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"CrossEntropyLoss = {value:F4}");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CrossEntropyLoss` | Initializes a new instance of the CrossEntropyLoss class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDerivative(Vector<>,Vector<>)` | Calculates the derivative of the Cross-Entropy loss function. |
| `CalculateLoss(Vector<>,Vector<>)` | Calculates the Cross-Entropy loss between predicted and actual probability distributions. |
| `CalculateLossAndGradientGpu(Tensor<>,Tensor<>)` | Calculates both Cross-Entropy loss and gradient on GPU in a single efficient pass. |
| `ComputeTapeLoss(Tensor<>,Tensor<>)` |  |

