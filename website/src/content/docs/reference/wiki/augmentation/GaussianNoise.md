---
title: "GaussianNoise<T>"
description: "Adds Gaussian noise to an image."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Adds Gaussian noise to an image.

## For Beginners

Think of this like the "grain" you see in photos taken in low light.
Adding random noise to training images teaches your model to focus on the real features
rather than memorizing exact pixel values.

## How It Works

Gaussian noise adds random values drawn from a normal (Gaussian) distribution to each pixel.
This simulates sensor noise in cameras and helps the model become robust to noisy inputs.

**When to use:**

- When training data is too clean (synthetic or studio images)
- When deployed images may have sensor noise
- As a regularization technique to prevent overfitting

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GaussianNoise(Double,Double,Double,Double,Double,Double)` | Creates a new Gaussian noise augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxStd` | Gets the maximum standard deviation of the noise. |
| `MaxValue` | Gets the maximum valid pixel value (for clamping). |
| `Mean` | Gets the mean of the Gaussian distribution. |
| `MinStd` | Gets the minimum standard deviation of the noise. |
| `MinValue` | Gets the minimum valid pixel value (for clamping). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Applies Gaussian noise to the image. |
| `GetParameters` |  |
| `SampleGaussian(AugmentationContext<>,Double,Double)` | Samples from a Gaussian distribution. |

