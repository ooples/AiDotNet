---
title: "DiceLoss<T>"
description: "Implements the Dice loss function, commonly used for image segmentation tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LossFunctions`

Implements the Dice loss function, commonly used for image segmentation tasks.

## For Beginners

Dice loss measures the overlap between predicted and actual segments in an image.
It's based on the Dice coefficient (also known as F1 score), which is a statistical measure of similarity.

The formula is: DiceLoss = 1 - (2 * intersection) / (sum of predicted + sum of actual)

Where:

- intersection is the sum of element-wise multiplication of predicted and actual values
- A value of 0 means perfect overlap (ideal predictions)
- A value of 1 means no overlap at all (worst predictions)

Key properties:

- It's ideal for problems where the positive class (what you're trying to detect) is rare
- Handles imbalanced data better than cross-entropy in many cases
- Focuses on maximizing the overlap between predictions and ground truth
- Commonly used in medical image segmentation, satellite imagery, and other segmentation tasks

Unlike cross-entropy, which treats each pixel independently, Dice loss considers the global
relationship between predicted and actual masks, which often leads to better segmentation results.

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new DiceLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"DiceLoss = {value:F4}");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiceLoss` | Initializes a new instance of the DiceLoss class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDerivative(Vector<>,Vector<>)` | Calculates the derivative of the Dice loss function. |
| `CalculateLoss(Vector<>,Vector<>)` | Calculates the Dice loss between predicted and actual values. |
| `ComputeTapeLoss(Tensor<>,Tensor<>)` |  |

