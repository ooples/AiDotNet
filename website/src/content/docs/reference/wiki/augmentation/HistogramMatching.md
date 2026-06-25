---
title: "HistogramMatching<T>"
description: "Matches the histogram of an image to a reference histogram or target distribution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Matches the histogram of an image to a reference histogram or target distribution.

## For Beginners

If you have images taken under different lighting conditions,
histogram matching can make them look more similar by adjusting one image's color
distribution to match another's.

## How It Works

Histogram matching transforms pixel intensities so the output histogram matches a specified
reference. This is useful for normalizing images from different sources to have similar
appearance, or for style transfer by matching color distributions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HistogramMatching(Double[][],Double,Double)` | Creates a new histogram matching augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BlendFactor` | Gets the blend factor [0, 1]. |
| `ReferenceHistograms` | Gets the reference histogram to match (per channel, normalized CDF). |

