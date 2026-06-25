---
title: "PoissonLoss"
description: "Implements the Poisson loss function for count data modeling."
section: "Reference"
---

_Loss Functions_

Implements the Poisson loss function for count data modeling.

## For Beginners

Poisson loss is designed for modeling count data where the target values represent the number of occurrences of an event in a fixed interval. Examples include: - Number of customer arrivals per hour - Number of network failures per day - Number of disease cases per region The loss is derived from the Poisson probability distribution, which is ideal for modeling rare events where we know the average rate of occurrence. The formula is: predicted - actual * log(predicted) + log(actual!) Since log(actual!) is constant with respect to predictions, it can be omitted during optimization, so the loss is often implemented as just: predicted - actual * log(predicted) Poisson loss is appropriate when: - Your target values are non-negative counts - The variance of the data is approximately equal to the mean - You're modeling the rate or frequency of events

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new PoissonLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"PoissonLoss = {value:F4}");
```

