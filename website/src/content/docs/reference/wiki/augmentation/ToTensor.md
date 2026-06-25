---
title: "ToTensor<T>"
description: "Converts an image to a normalized tensor with values in [0, 1]."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Converts an image to a normalized tensor with values in [0, 1].

## For Beginners

Digital images store colors as numbers from 0 to 255.
Neural networks prefer small numbers close to zero, so we divide by 255 to get
values between 0 and 1. This is almost always the first thing you do to an image
before feeding it to a model.

## How It Works

ToTensor converts pixel values from [0, 255] integer range to [0, 1] floating point range
by dividing by 255. This is the standard first step in most image preprocessing pipelines,
equivalent to `torchvision.transforms`.

**When to use:**

- As the first preprocessing step for images with [0, 255] values
- Before applying normalization (Normalize expects [0, 1] input)

**When NOT to use:**

- If values are already in [0, 1] range
- If the image was loaded as float (many frameworks do this automatically)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ToTensor(Double,Double)` | Creates a new ToTensor augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ScaleFactor` | Gets the scale factor used for conversion. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Converts pixel values to [0, 1] range. |
| `GetParameters` |  |

