---
title: "Contrast<T>"
description: "Adjusts the contrast of an image by scaling pixel values around the mean."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Adjusts the contrast of an image by scaling pixel values around the mean.

## For Beginners

Think of contrast like the difference between a sunny day
(high contrast with bright lights and dark shadows) and a foggy day (low contrast where
everything looks similar in brightness). This teaches your model to recognize objects
in both crisp and hazy conditions.

## How It Works

Contrast adjustment changes the difference between light and dark areas of an image.
Higher contrast makes light areas lighter and dark areas darker, while lower contrast
makes the image appear more "washed out" or flat.

**When to use:**

- Images from cameras with different quality or settings
- Scenes with varying lighting conditions
- Data from different sources with different post-processing

**When NOT to use:**

- Tasks where pixel intensity relationships are critical
- Medical imaging where contrast carries diagnostic information
- Already preprocessed images with standardized contrast

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Contrast(Double,Double,Double)` | Creates a new contrast augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxFactor` | Gets the maximum contrast adjustment factor. |
| `MinFactor` | Gets the minimum contrast adjustment factor. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Applies the contrast adjustment to the image. |
| `GetParameters` |  |

