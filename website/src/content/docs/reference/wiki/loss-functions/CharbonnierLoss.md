---
title: "CharbonnierLoss"
description: "Implements the Charbonnier loss function, a smooth approximation of L1 loss."
section: "Reference"
---

_Loss Functions_

Implements the Charbonnier loss function, a smooth approximation of L1 loss.

## For Beginners

Charbonnier loss is a differentiable approximation of the absolute error (L1 loss). The formula is: L(x, y) = sqrt((x - y)² + ε²) Where: - x is the predicted value - y is the actual value - ε (epsilon) is a small constant (typically 1e-6 or 1e-9) Key properties: - Like L1 loss, it's robust to outliers - Unlike L1 loss, it's differentiable everywhere (smooth at zero) - Provides more stable gradients for training deep neural networks - Widely used in video super-resolution and image restoration Charbonnier loss is preferred over L1 loss in deep learning because: - L1 loss has an undefined derivative at zero - Charbonnier loss provides smooth gradients that help with optimization - The epsilon parameter controls the "sharpness" of the approximation

## How It Works

**Reference:** Charbonnier et al., "Two deterministic half-quadratic regularization algorithms for computed imaging", ICIP 1994.

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new CharbonnierLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"CharbonnierLoss = {value:F4}");
```

