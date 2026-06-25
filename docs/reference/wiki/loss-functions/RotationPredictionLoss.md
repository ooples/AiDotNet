---
title: "RotationPredictionLoss"
description: "Self-supervised loss function based on rotation prediction for images."
section: "Reference"
---

_Loss Functions_

Self-supervised loss function based on rotation prediction for images.

## For Beginners

This teaches the model to understand image structure without labels. Imagine showing someone 100 photos, each rotated randomly: - They learn to recognize: which way is "up", spatial relationships, object orientations - They don't need to know: what the objects are (no labels needed) After this training, when you show them 5 labeled cat photos: - They already understand image structure - They just need to learn: "cats look like THIS" - Much faster than learning everything from scratch! **How it works:** 1. Take each unlabeled image 2. Create 4 versions: rotated by 0°, 90°, 180°, 270° 3. Label each version: 0, 1, 2, 3 (which rotation was applied) 4. Train model to predict the rotation **What the model learns:** - Edge orientations - Spatial relationships - Object structure - "Natural" vs "unnatural" orientations These features are very useful for actual classification tasks!

## How It Works

Rotation prediction is a self-supervised task where: 1. Images are rotated by 0°, 90°, 180°, or 270° 2. Model predicts which rotation was applied (4-class classification) 3. Model learns spatial relationships and features without needing class labels

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new RotationPredictionLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"RotationPredictionLoss = {value:F4}");
```

