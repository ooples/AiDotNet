---
title: "ExponentialLoss<T>"
description: "Implements the Exponential Loss function, commonly used in boosting algorithms."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LossFunctions`

Implements the Exponential Loss function, commonly used in boosting algorithms.

## For Beginners

Exponential Loss is a loss function that heavily penalizes incorrect predictions,
especially those that are far off from the true values.

The formula is: exp(-y * f(x))
Where:

- y is the true label (usually -1 or 1 for binary classification)
- f(x) is the model's prediction

Key properties:

- It grows exponentially as the error increases
- Correct predictions with high confidence result in values close to zero
- Incorrect predictions result in very large values
- It's especially sensitive to outliers and misclassifications

Exponential Loss is primarily used in:

- AdaBoost and other boosting algorithms
- Ensemble methods that need to focus on hard examples
- Learning problems where avoiding mistakes is critical

The exponential nature makes the model pay more attention to difficult examples
and outliers compared to other loss functions like hinge loss or log loss.

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new ExponentialLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"ExponentialLoss = {value:F4}");
```

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDerivative(Vector<>,Vector<>)` | Calculates the derivative of the Exponential Loss function. |
| `CalculateLoss(Vector<>,Vector<>)` | Calculates the Exponential Loss between predicted and actual values. |
| `CalculateLossAndGradientGpu(Tensor<>,Tensor<>)` | Calculates both Exponential loss and gradient on GPU in a single efficient pass. |
| `ComputeTapeLoss(Tensor<>,Tensor<>)` |  |

