---
title: "Brightness<T>"
description: "Adjusts the brightness of an image by adding a random offset to all pixel values."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Adjusts the brightness of an image by adding a random offset to all pixel values.

## For Beginners

Think of this like adjusting the brightness slider on your phone.
Making an image brighter adds light to all pixels, making it darker subtracts light.
This teaches your model to recognize objects whether they're in bright sunlight or shade.

## How It Works

Brightness adjustment simulates different lighting conditions by uniformly increasing
or decreasing all pixel values. This helps models become robust to variations in
ambient lighting and exposure settings.

**When to use:**

- Outdoor photography where lighting varies by time of day
- Indoor scenes with different lighting conditions
- Any task where exposure/lighting might vary between training and deployment

**When NOT to use:**

- Tasks where absolute brightness is meaningful (e.g., astronomy)
- Images already normalized to a specific brightness range
- Medical imaging where pixel intensity has diagnostic meaning

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Brightness(Double,Double,Double)` | Creates a new brightness augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxFactor` | Gets the maximum brightness adjustment factor. |
| `MinFactor` | Gets the minimum brightness adjustment factor. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Applies the brightness adjustment to the image. |
| `GetParameters` |  |

