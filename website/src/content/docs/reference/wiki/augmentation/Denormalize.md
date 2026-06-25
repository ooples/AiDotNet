---
title: "Denormalize<T>"
description: "Reverses normalization of an image tensor, restoring original pixel value ranges."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Reverses normalization of an image tensor, restoring original pixel value ranges.

## For Beginners

After your model processes an image, the pixel values are
in a normalized range (roughly -2 to +2). To display or save the image, you need to
convert back to normal pixel values (0-255 or 0-1). This operation does that.

## How It Works

Denormalization reverses the normalization operation: `output = input * std + mean`.
This is the inverse of `Normalize` and restores pixel values to their
original range for visualization or saving.

**When to use:**

- Visualizing model inputs or intermediate activations
- Saving processed images back to disk
- Converting model output back to displayable format

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Denormalize(Double[],Double[],Boolean,Double)` | Creates a new denormalization augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClampOutput` | Gets whether to clamp output values to [0, 1]. |
| `Mean` | Gets the per-channel mean values used in the original normalization. |
| `Std` | Gets the per-channel standard deviation values used in the original normalization. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Applies denormalization: output = input * std + mean. |
| `Clip(Boolean,Double)` | Creates a denormalization for CLIP statistics. |
| `GetParameters` |  |
| `ImageNet(Boolean,Double)` | Creates a denormalization for ImageNet statistics. |

