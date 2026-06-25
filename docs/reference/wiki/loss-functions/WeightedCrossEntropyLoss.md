---
title: "WeightedCrossEntropyLoss"
description: "Implements the Weighted Cross Entropy loss function for classification problems with uneven class importance."
section: "Reference"
---

_Loss Functions_

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

