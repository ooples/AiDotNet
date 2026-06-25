---
title: "ColorJitter<T>"
description: "Applies random combinations of brightness, contrast, saturation, and hue adjustments."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Applies random combinations of brightness, contrast, saturation, and hue adjustments.

## For Beginners

Think of this as applying multiple photo filters randomly.
Just like how the same scene looks different when photographed with different phones
or in different lighting, ColorJitter creates these natural variations automatically.
This is one of the most commonly used augmentations for image classification.

## How It Works

ColorJitter is a powerful composite augmentation that randomly adjusts multiple color
properties in a single operation. This simulates the wide variety of color variations
that occur in real-world photography due to different cameras, lighting, and environments.

**When to use:**

- General image classification tasks
- When training data comes from a single camera/source but deployment varies
- To make models robust to different lighting and camera settings

**When NOT to use:**

- Color is the primary classification feature
- Medical/scientific imaging with calibrated color
- Tasks where specific color accuracy is required

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ColorJitter(Double,Double,Double,Double,Double)` | Creates a new color jitter augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BrightnessRange` | Gets the brightness adjustment range (0.0 = no change possible). |
| `ContrastRange` | Gets the contrast adjustment range (0.0 = no change possible). |
| `HueRange` | Gets the hue adjustment range in degrees (0.0 = no change possible). |
| `SaturationRange` | Gets the saturation adjustment range (0.0 = no change possible). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Applies the color jitter to the image. |
| `GetParameters` |  |

