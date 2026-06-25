---
title: "HistogramEqualization<T>"
description: "Applies standard histogram equalization to improve image contrast."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Applies standard histogram equalization to improve image contrast.

## For Beginners

If your image looks washed out or too dark, histogram
equalization stretches the range of colors to use the full spectrum, making details
more visible.

## How It Works

Histogram equalization redistributes pixel intensity values to produce a more uniform
histogram, effectively spreading out the most frequent intensity values. This improves
contrast, especially for images that are too dark or too bright.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HistogramEqualization(Boolean,Double)` | Creates a new histogram equalization augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `PerChannel` | Gets whether to apply per-channel equalization. |

