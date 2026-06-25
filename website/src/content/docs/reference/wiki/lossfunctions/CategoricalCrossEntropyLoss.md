---
title: "CategoricalCrossEntropyLoss<T>"
description: "Implements the Categorical Cross Entropy loss function for multi-class classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LossFunctions`

Implements the Categorical Cross Entropy loss function for multi-class classification.

## For Beginners

Categorical Cross Entropy is used for multi-class classification problems,
where you need to assign inputs to one of several categories (like classifying images as dog, cat, bird, etc.).

It measures how well the predicted probability distribution matches the actual distribution of classes.

The formula is: CCE = -(1/n) * ?[?(actual_j * log(predicted_j))]

Where:

- actual_j is usually a one-hot encoded vector (1 for the correct class, 0 for others)
- predicted_j is the predicted probability for each class (typically from a softmax output)
- The inner sum is over all classes, and the outer sum is over all samples

Key properties:

- Predicted values should be probabilities (between 0 and 1) that sum to 1 across classes
- It heavily penalizes confident incorrect predictions
- It's the standard loss function for multi-class neural network classifiers
- Often used together with the softmax activation function in the output layer

This loss function is ideal when your model needs to choose one option from multiple possibilities.

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new CategoricalCrossEntropyLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"CategoricalCrossEntropyLoss = {value:F4}");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CategoricalCrossEntropyLoss` | Initializes a new instance of the CategoricalCrossEntropyLoss class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDerivative(Vector<>,Vector<>)` | Calculates the derivative of the Categorical Cross Entropy loss function. |
| `CalculateLoss(Vector<>,Vector<>)` | Calculates the Categorical Cross Entropy loss between predicted and actual values. |
| `CalculateLossAndGradientGpu(Tensor<>,Tensor<>)` | Calculates both Categorical Cross Entropy loss and gradient on GPU in a single efficient pass. |
| `ComputeTapeLoss(Tensor<>,Tensor<>)` |  |

