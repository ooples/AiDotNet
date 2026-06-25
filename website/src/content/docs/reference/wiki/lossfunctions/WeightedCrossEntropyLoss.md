---
title: "WeightedCrossEntropyLoss<T>"
description: "Implements the Weighted Cross Entropy loss function for classification problems with uneven class importance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LossFunctions`

Implements the Weighted Cross Entropy loss function for classification problems with uneven class importance.

## For Beginners

Weighted Cross Entropy is a variation of the standard cross-entropy loss that applies
different weights to different samples or classes.

The regular cross-entropy penalizes all misclassifications equally, but in some cases:

- Some classes might be more important to classify correctly
- Some classes might be rare in the training data but important in practice
- Some samples might be more reliable or representative than others

Weighted Cross Entropy lets you control the importance of different samples by applying weights
to them. Higher weights mean the model will focus more on getting those specific samples right.

This loss function is particularly useful for:

- Imbalanced datasets where some classes are underrepresented
- Problems where misclassifying certain classes is more costly than others
- Situations where you have varying confidence in your training data

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new WeightedCrossEntropyLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"WeightedCrossEntropyLoss = {value:F4}");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WeightedCrossEntropyLoss(Vector<>)` | Initializes a new instance of the WeightedCrossEntropyLoss class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDerivative(Vector<>,Vector<>)` | Calculates the derivative of the Weighted Cross Entropy loss function. |
| `CalculateLoss(Vector<>,Vector<>)` | Calculates the Weighted Cross Entropy loss between predicted and actual values. |
| `ComputeTapeLoss(Tensor<>,Tensor<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_weights` | The weights to apply to each sample. |

