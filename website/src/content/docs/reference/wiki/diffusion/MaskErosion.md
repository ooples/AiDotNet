---
title: "MaskErosion<T>"
description: "Erodes a mask by shrinking masked regions, removing thin protrusions and small artifacts."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.MaskUtilities`

Erodes a mask by shrinking masked regions, removing thin protrusions and small artifacts.

## For Beginners

Erosion makes the white (masked) area smaller. If your mask
has small white specks or thin white lines, erosion removes them. It's like
"peeling" a layer off the edge of the masked region.

## How It Works

Morphological erosion shrinks white (masked) regions by taking the minimum value
in a local neighborhood. This removes small noise, thin connections, and smooths
mask boundaries inward.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MaskErosion(Int32)` | Initializes a new instance of the `MaskErosion` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ColumnIndices` |  |
| `IsFitted` |  |
| `SupportsInverseTransform` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(Tensor<>)` | Applies morphological erosion to a mask tensor. |
| `Fit(Tensor<>)` |  |
| `FitTransform(Tensor<>)` |  |
| `GetFeatureNamesOut(String[])` |  |
| `InverseTransform(Tensor<>)` |  |
| `Transform(Tensor<>)` |  |

