---
title: "RgbToGrayscale<T>"
description: "Converts an RGB image to grayscale using configurable channel weights."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Converts an RGB image to grayscale using configurable channel weights.

## For Beginners

Converting to grayscale removes color information, keeping
only brightness. This is useful when color doesn't matter for your task (like reading
text or detecting edges) and reduces computation by working with 1 channel instead of 3.

## How It Works

Grayscale conversion reduces a 3-channel RGB image to a single luminance channel
using weighted combination of the color channels. The default weights (0.2989, 0.5870, 0.1140)
follow the ITU-R BT.601 standard, which accounts for human perception of brightness
(green appears brightest, blue appears darkest).

**When to use:**

- Document analysis and OCR
- Edge detection and structural analysis
- Reducing model size when color is not informative
- As a data augmentation to teach color invariance

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RgbToGrayscale(Double,Double,Double,Int32,Double)` | Creates a new RGB to grayscale conversion. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BlueWeight` | Gets the weight for the blue channel. |
| `GreenWeight` | Gets the weight for the green channel. |
| `OutputChannels` | Gets the number of output channels (1 or 3). |
| `RedWeight` | Gets the weight for the red channel. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Converts the RGB image to grayscale. |
| `GetParameters` |  |

