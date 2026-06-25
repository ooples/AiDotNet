---
title: "Normalize<T>"
description: "Normalizes an image tensor with per-channel mean and standard deviation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Normalizes an image tensor with per-channel mean and standard deviation.

## For Beginners

Neural networks work better when input values are small numbers
centered around zero. Raw images have pixel values from 0 to 255 (or 0 to 1). Normalization
adjusts these values so they have a mean of 0 and standard deviation of 1, which makes
training more stable.

## How It Works

Normalization transforms pixel values using: `output = (input - mean) / std` for each
channel. This centers the data around zero and scales it to unit variance, which helps
neural networks train faster and converge more reliably.

**Common normalization values:**

- **ImageNet**: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- **CLIP**: mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
- **Simple**: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] (maps [0,1] to [-1,1])

**When to use:**

- As the last step in preprocessing before model input
- When using pretrained models (must match training normalization)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Normalize(Double[],Double[],Double)` | Creates a new normalization augmentation with per-channel mean and std. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Mean` | Gets the per-channel mean values. |
| `Std` | Gets the per-channel standard deviation values. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Applies normalization: output = (input - mean) / std. |
| `Clip(Double)` | Creates a normalization with CLIP statistics. |
| `GetParameters` |  |
| `ImageNet(Double)` | Creates a normalization with ImageNet statistics. |
| `NegativeOneToOne(Int32,Double)` | Creates a normalization that maps [0,1] to [-1,1]. |

