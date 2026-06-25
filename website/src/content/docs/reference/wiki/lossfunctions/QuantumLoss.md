---
title: "QuantumLoss<T>"
description: "Represents a quantum-specific loss function for quantum neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LossFunctions`

Represents a quantum-specific loss function for quantum neural networks.

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new QuantumLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"QuantumLoss = {value:F4}");
```

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDerivative(Vector<>,Vector<>)` | Calculates the derivative of the quantum loss function. |
| `CalculateLoss(Vector<>,Vector<>)` | Calculates the quantum loss between predicted and expected quantum states. |
| `ComputeTapeLoss(Tensor<>,Tensor<>)` |  |

