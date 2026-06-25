---
title: "ToFloat<T>"
description: "Converts an image tensor to floating-point representation with configurable scaling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Converts an image tensor to floating-point representation with configurable scaling.

## For Beginners

Sometimes images come in different formats: 8-bit (0-255),
16-bit (0-65535), or already floating point. This transform lets you convert between
any of these ranges. For example, converting 16-bit medical images to [0, 1] range.

## How It Works

ToFloat provides flexible conversion of image pixel values to floating-point range.
Unlike `ToTensor` which always divides by 255, ToFloat allows custom
source and target ranges.

**When to use:**

- Converting 16-bit images (e.g., medical, satellite) to [0, 1]
- Mapping to custom ranges like [-1, 1]
- When you need more control than ToTensor provides

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ToFloat(Double,Double,Double,Double,Double)` | Creates a new ToFloat augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SourceMax` | Gets the maximum value of the source range. |
| `SourceMin` | Gets the minimum value of the source range. |
| `TargetMax` | Gets the maximum value of the target range. |
| `TargetMin` | Gets the minimum value of the target range. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Converts pixel values to the target range. |
| `From16Bit(Double)` | Creates a ToFloat for 16-bit images to [0, 1]. |
| `GetParameters` |  |
| `ToNegativeOneToOne(Double)` | Creates a ToFloat that maps [0, 255] to [-1, 1]. |

