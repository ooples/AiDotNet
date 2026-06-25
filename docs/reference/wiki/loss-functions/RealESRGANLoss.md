---
title: "RealESRGANLoss"
description: "Combined loss function for Real-ESRGAN super-resolution training."
section: "Reference"
---

_Loss Functions_

Combined loss function for Real-ESRGAN super-resolution training.

## For Beginners

This loss function guides Real-ESRGAN training by balancing three goals: 1. **L1 Loss (pixel accuracy)**: Makes sure each pixel is close to the ground truth. Like comparing photos pixel-by-pixel. 2. **Perceptual Loss (looks right)**: Uses a pre-trained network (VGG) to compare high-level features. Ensures the output "looks right" even if pixels aren't exact. 3. **GAN Loss (realistic details)**: The discriminator judges if output looks real. This adds fine details and textures that make images look natural. The weights control how much each goal matters: - Higher L1 weight = more pixel-accurate but potentially blurry - Higher perceptual weight = better visual quality - Higher GAN weight = more realistic textures but potential artifacts The default weights (1.0, 1.0, 0.1) are from the Real-ESRGAN paper.

## How It Works

Real-ESRGAN uses a combination of three loss functions for training: - L1 (pixel-wise) loss: Ensures pixel-level accuracy - Perceptual (VGG) loss: Ensures perceptual quality using deep features - GAN (adversarial) loss: Ensures realistic details and textures 

The total loss is computed as: 

**Reference:** Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data", ICCV 2021. https://arxiv.org/abs/2107.10833

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new RealESRGANLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"RealESRGANLoss = {value:F4}");
```

