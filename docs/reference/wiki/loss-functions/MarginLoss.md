---
title: "MarginLoss"
description: "Implements the Margin loss function, specifically designed for Capsule Networks."
section: "Reference"
---

_Loss Functions_

Implements the Margin loss function, specifically designed for Capsule Networks.

## For Beginners

Margin loss is a special loss function used in Capsule Networks. The formula is: T_c * max(0, m+ - ||v_c||)^2 + lambda * (1 - T_c) * max(0, ||v_c|| - m-)^2 Where: - T_c is 1 if class c is present, 0 otherwise - ||v_c|| is the length of the output vector of the capsule for class c - m+ is the upper bound (usually 0.9) - m- is the lower bound (usually 0.1) - lambda is a down-weighting factor (usually 0.5) Key properties: - Encourages the network to output high values for correct classes - Discourages high outputs for incorrect classes - Helps in learning to represent different aspects of the input Margin loss is ideal for Capsule Networks because: - It allows multiple classes to be present in the same image - It encourages the network to learn to represent different viewpoints and transformations - It helps in achieving equivariance, a key property of Capsule Networks

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new MarginLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"MarginLoss = {value:F4}");
```

