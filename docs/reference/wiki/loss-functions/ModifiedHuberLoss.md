---
title: "ModifiedHuberLoss"
description: "Implements the Modified Huber Loss function, a smoother version of the hinge loss."
section: "Reference"
---

_Loss Functions_

Implements the Modified Huber Loss function, a smoother version of the hinge loss.

## For Beginners

Modified Huber Loss is a smoother version of the hinge loss that's less sensitive to outliers.
It combines quadratic behavior near zero with linear behavior for large negative values.

The formula is:

- For z = -1: max(0, 1 - z)²
- For z < -1: -4 * z

Where z = y * f(x), with y being the true label and f(x) the prediction.

Key properties:

- It's smoother than hinge loss, making optimization easier
- It's more robust to outliers than squared hinge loss
- It combines the benefits of both quadratic and linear losses
- It has a continuous first derivative

Modified Huber Loss is particularly useful for:

- Binary classification problems
- Datasets with noisy labels
- Problems where you want to balance between being sensitive to errors but not overly influenced by extreme mistakes

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new ModifiedHuberLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"ModifiedHuberLoss = {value:F4}");
```

