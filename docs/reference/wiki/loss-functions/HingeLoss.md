---
title: "HingeLoss"
description: "Implements the Hinge loss function commonly used in support vector machines."
section: "Reference"
---

_Loss Functions_

Implements the Hinge loss function commonly used in support vector machines.

## For Beginners

Hinge loss is used for binary classification problems, particularly in support vector machines (SVMs).
It measures how well your model separates different classes.

The formula is: max(0, 1 - y * f(x)), where:

- y is the true label (usually -1 or 1)
- f(x) is the model's prediction

Key properties of hinge loss:

- It penalizes predictions that are incorrect or not confident enough
- It's zero when the prediction is correct and confident (y*f(x) = 1)
- It increases linearly when the prediction is incorrect or not confident enough
- It encourages the model to find a decision boundary with a large margin between classes

This loss function is ideal for binary classification tasks where you want to maximize
the margin between different classes, which often improves generalization to new data.

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new HingeLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"HingeLoss = {value:F4}");
```

