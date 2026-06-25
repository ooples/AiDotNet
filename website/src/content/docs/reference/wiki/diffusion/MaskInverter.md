---
title: "MaskInverter<T>"
description: "Inverts a mask so masked regions become unmasked and vice versa."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.MaskUtilities`

Inverts a mask so masked regions become unmasked and vice versa.

## For Beginners

If your mask highlights a person (white = person, black = background),
inverting it highlights the background instead. This is useful when different tools
use opposite conventions for what "masked" means.

## How It Works

Mask inversion computes (1 - mask) for each pixel, flipping the masked and unmasked
regions. This is commonly used when you want to inpaint the background instead of
the foreground, or to convert between "keep" and "replace" mask conventions.

## Properties

| Property | Summary |
|:-----|:--------|
| `ColumnIndices` |  |
| `IsFitted` |  |
| `SupportsInverseTransform` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(Tensor<>)` | Inverts a mask tensor by computing (1 - value) for each pixel. |
| `Fit(Tensor<>)` |  |
| `FitTransform(Tensor<>)` |  |
| `GetFeatureNamesOut(String[])` |  |
| `InverseTransform(Tensor<>)` |  |
| `Transform(Tensor<>)` |  |

