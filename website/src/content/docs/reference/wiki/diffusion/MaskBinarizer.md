---
title: "MaskBinarizer<T>"
description: "Converts a soft mask (continuous values) into a binary mask (0 or 1) using a threshold."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.MaskUtilities`

Converts a soft mask (continuous values) into a binary mask (0 or 1) using a threshold.

## For Beginners

Some masks have "soft" edges with values like 0.3 or 0.7.
Binarizing converts everything above a threshold to 1 (masked) and everything
below to 0 (unmasked), creating a clean on/off mask.

## How It Works

Binarization converts masks with gradual transitions into hard masks where each
pixel is either fully masked (1) or fully unmasked (0). Useful when downstream
operations require strict binary masks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MaskBinarizer(Double)` | Initializes a new instance of the `MaskBinarizer` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ColumnIndices` |  |
| `IsFitted` |  |
| `SupportsInverseTransform` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(Tensor<>)` | Binarizes a mask tensor. |
| `Fit(Tensor<>)` |  |
| `FitTransform(Tensor<>)` |  |
| `GetFeatureNamesOut(String[])` |  |
| `InverseTransform(Tensor<>)` |  |
| `Transform(Tensor<>)` |  |

