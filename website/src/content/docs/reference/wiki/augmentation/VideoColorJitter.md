---
title: "VideoColorJitter<T>"
description: "Applies color jitter (brightness, contrast, saturation) to video frames."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Video`

Applies color jitter (brightness, contrast, saturation) to video frames.

## For Beginners

Color jitter randomly adjusts brightness, contrast,
and saturation of video frames. This helps models become robust to different
lighting conditions and camera settings.

## How It Works

**When to use:**

- Videos captured with different cameras or lighting
- Outdoor videos with varying light conditions
- Reducing overfitting to specific color characteristics

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VideoColorJitter(Double,Double,Double,Double,Double)` | Creates a new video color jitter augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BrightnessRange` | Gets the maximum brightness adjustment. |
| `ContrastRange` | Gets the maximum contrast adjustment. |
| `SaturationRange` | Gets the maximum saturation adjustment. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>[],AugmentationContext<>)` |  |
| `GetParameters` |  |

