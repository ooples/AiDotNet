---
title: "MaskDilation<T>"
description: "Dilates a mask by expanding masked regions, filling small holes and connecting nearby regions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.MaskUtilities`

Dilates a mask by expanding masked regions, filling small holes and connecting nearby regions.

## For Beginners

Dilation makes the white (masked) area larger. If your mask
has small gaps or holes, dilation fills them in. It's like "padding" the edge
of the masked region outward. Often used with erosion for mask cleanup.

## How It Works

Morphological dilation expands white (masked) regions by taking the maximum value
in a local neighborhood. This fills small holes, connects nearby components, and
grows mask boundaries outward.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MaskDilation(Int32)` | Initializes a new instance of the `MaskDilation` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ColumnIndices` |  |
| `IsFitted` |  |
| `SupportsInverseTransform` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(Tensor<>)` | Applies morphological dilation to a mask tensor. |
| `Fit(Tensor<>)` |  |
| `FitTransform(Tensor<>)` |  |
| `GetFeatureNamesOut(String[])` |  |
| `InverseTransform(Tensor<>)` |  |
| `Transform(Tensor<>)` |  |

