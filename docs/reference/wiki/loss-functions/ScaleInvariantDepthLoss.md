---
title: "ScaleInvariantDepthLoss"
description: "Scale-invariant depth loss function for depth estimation training."
section: "Reference"
---

_Loss Functions_

Scale-invariant depth loss function for depth estimation training.

## For Beginners

This loss function is specifically designed for training depth estimation models. It handles the inherent scale ambiguity in monocular depth estimation by focusing on the relative depth relationships between pixels rather than absolute depth values.

## How It Works

**Technical Details:** The loss is computed as: (1/n) * Σ(d²) - (λ/n²) * (Σd)² where d = log(pred) - log(actual), and λ controls the scale-invariance penalty. 

**Reference:** Eigen et al., "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network"

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new ScaleInvariantDepthLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"ScaleInvariantDepthLoss = {value:F4}");
```

