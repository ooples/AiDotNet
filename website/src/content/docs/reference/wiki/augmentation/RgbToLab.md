---
title: "RgbToLab<T>"
description: "Converts an image between RGB and CIE L*a*b* color space."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Converts an image between RGB and CIE L*a*b* color space.

## For Beginners

L*a*b* is a special color space where the distance between
two colors matches how different they look to the human eye. L* is lightness (0=black,
100=white), a* goes from green (negative) to red (positive), and b* goes from blue
(negative) to yellow (positive).

## How It Works

The CIE L*a*b* color space is designed to be perceptually uniform, meaning that equal
numerical changes correspond to roughly equal perceived color differences. The conversion
goes through XYZ color space as an intermediate step.

**Channel layout:**

- **L* (Lightness)**: [0, 100]
- **a***: Approximately [-128, 127] (green to red)
- **b***: Approximately [-128, 127] (blue to yellow)

**When to use:**

- Color difference calculations (Delta E)
- Perceptually uniform color augmentation
- Color transfer between images
- Image quality assessment

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RgbToLab(Boolean,Double)` | Creates a new RGB to L*a*b* conversion. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NormalizeOutput` | Gets whether to normalize output to [0, 1] range. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Converts the image from RGB to L*a*b* color space. |
| `GetParameters` |  |
| `LabToRgb(Double,Double,Double)` | Converts L*a*b* values back to RGB. |

