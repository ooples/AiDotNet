---
title: "OpticalDistortion<T>"
description: "Simulates barrel and pincushion lens distortion."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Simulates barrel and pincushion lens distortion.

## For Beginners

Camera lenses aren't perfect — they bend straight lines
slightly. Wide-angle lenses make edges bulge outward (barrel distortion), while telephoto
lenses make edges pinch inward (pincushion). This augmentation simulates these effects.

## How It Works

Optical distortion simulates the radial distortion produced by camera lenses. Barrel
distortion (positive k) makes straight lines bow outward, while pincushion distortion
(negative k) makes them bow inward. This is a common effect in wide-angle and telephoto lenses.

**When to use:**

- Training models robust to different camera lenses
- Autonomous driving (wide-angle dash cameras)
- Surveillance camera footage processing

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OpticalDistortion(Double,Double,Double,Double)` | Creates a new optical distortion augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FillValue` | Gets the fill value for out-of-bounds pixels. |
| `MaxDistortionK` | Gets the maximum distortion coefficient. |
| `MinDistortionK` | Gets the minimum distortion coefficient. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Applies the optical distortion. |
| `GetParameters` |  |

