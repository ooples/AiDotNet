---
title: "FocalLoss<T>"
description: "Implements the Focal Loss function, which gives more weight to hard-to-classify examples."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LossFunctions`

Implements the Focal Loss function, which gives more weight to hard-to-classify examples.

## For Beginners

Focal Loss was designed to handle class imbalance in classification problems,
especially for object detection tasks where background examples vastly outnumber foreground objects.

It modifies the standard cross-entropy loss by adding a factor that reduces the loss contribution
from easy-to-classify examples and increases the importance of hard-to-classify examples.

The formula is: -a(1-p)^? * log(p) for positive class
-(1-a)p^? * log(1-p) for negative class
Where:

- p is the model's estimated probability for the correct class
- a is a weighting factor that balances positive vs negative examples
- ? (gamma) is the focusing parameter that adjusts how much to focus on hard examples

Key properties:

- When ?=0, Focal Loss equals Cross-Entropy Loss
- Higher ? values increase focus on hard-to-classify examples
- a helps handle class imbalance by giving more weight to the minority class

This loss function is ideal for:

- Highly imbalanced datasets
- One-stage object detectors
- Any classification task where easy negatives dominate training

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new FocalLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"FocalLoss = {value:F4}");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FocalLoss(Double,Double)` | Initializes a new instance of the FocalLoss class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDerivative(Vector<>,Vector<>)` | Calculates the derivative of the Focal Loss with respect to the predicted values. |
| `CalculateLoss(Vector<>,Vector<>)` | Calculates the Focal Loss between predicted and actual values. |
| `CalculateLossAndGradientGpu(Tensor<>,Tensor<>)` | Calculates both Focal Loss and gradient on GPU in a single efficient pass. |
| `ComputeTapeLoss(Tensor<>,Tensor<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_alpha` | The weighting factor that balances positive vs negative examples. |
| `_gamma` | The focusing parameter that down-weights easy examples. |

