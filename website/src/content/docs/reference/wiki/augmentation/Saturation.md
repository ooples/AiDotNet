---
title: "Saturation<T>"
description: "Adjusts the saturation (color intensity) of an image."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Adjusts the saturation (color intensity) of an image.

## For Beginners

Think of saturation like the vibrancy slider on photo apps.
High saturation makes colors pop (like a sunset photo), while low saturation
makes colors look faded (like an old photograph). A saturation of 0 would make
the image completely grayscale.

## How It Works

Saturation adjustment changes how vivid or muted colors appear in an image.
Higher saturation makes colors more vibrant, while lower saturation makes
colors appear more gray or washed out.

**When to use:**

- Images from different cameras with varying color profiles
- Scenes with different lighting temperatures (warm vs cool)
- When you want the model to focus on shapes rather than colors

**When NOT to use:**

- Color is the primary classification feature (e.g., traffic lights)
- Medical imaging where color accuracy matters
- Tasks involving color matching or color recognition

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Saturation(Double,Double,Double)` | Creates a new saturation augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxFactor` | Gets the maximum saturation adjustment factor. |
| `MinFactor` | Gets the minimum saturation adjustment factor. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Applies the saturation adjustment to the image. |
| `GetParameters` |  |

