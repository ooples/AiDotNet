---
title: "GaussianBlur<T>"
description: "Applies Gaussian blur to an image."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Applies Gaussian blur to an image.

## For Beginners

Think of this like looking at a photo through frosted glass.
The image becomes softer and details are less sharp. This teaches your model to recognize
objects even when they're not perfectly in focus.

## How It Works

Gaussian blur smooths the image by convolving it with a Gaussian kernel. This simulates
out-of-focus images or motion blur, helping the model become robust to blurry inputs.

**When to use:**

- When deployed images may be out of focus
- When training data is too sharp compared to real-world images
- To reduce high-frequency noise sensitivity

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GaussianBlur(Double,Double,Int32,Double)` | Creates a new Gaussian blur augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `KernelSize` | Gets the kernel size for the Gaussian blur. |
| `MaxSigma` | Gets the maximum sigma (standard deviation) for the Gaussian kernel. |
| `MinSigma` | Gets the minimum sigma (standard deviation) for the Gaussian kernel. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Applies Gaussian blur to the image. |
| `ApplyConvolution(ImageTensor<>,Double[0:,0:])` | Applies a convolution with the given kernel. |
| `GenerateGaussianKernel(Int32,Double)` | Generates a 2D Gaussian kernel. |
| `GetParameters` |  |
| `ReflectIndex(Int32,Int32)` | Reflects an index at boundaries. |

