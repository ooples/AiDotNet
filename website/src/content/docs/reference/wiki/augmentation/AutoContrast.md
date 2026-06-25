---
title: "AutoContrast<T>"
description: "Maximizes image contrast by stretching the intensity range to fill [0, max]."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Maximizes image contrast by stretching the intensity range to fill [0, max].

## For Beginners

If your darkest pixel is 50 and brightest is 200, this stretches
them to 0 and 255, making the image use the full brightness range.

## How It Works

AutoContrast finds the minimum and maximum pixel values per channel and linearly
scales them to span the full range. This is equivalent to a per-channel min-max normalization.

## Properties

| Property | Summary |
|:-----|:--------|
| `Cutoff` | Gets the percentage of lightest/darkest pixels to ignore when finding min/max. |

