---
title: "MaskFromSegmentation<T>"
description: "Generates a binary mask from a segmentation map by selecting specific class labels."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.MaskUtilities`

Generates a binary mask from a segmentation map by selecting specific class labels.

## For Beginners

Segmentation models label every pixel (e.g., "sky", "person",
"building"). This utility converts those labels into a mask — for example, you
could select "sky" to create a mask that covers only the sky region for replacement.

## How It Works

Takes a segmentation tensor where each pixel holds a class label and produces
a binary mask where selected classes are 1 and all others are 0. This bridges
semantic segmentation output to inpainting mask input.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MaskFromSegmentation(IEnumerable<Int32>)` | Initializes a new instance of the `MaskFromSegmentation` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ColumnIndices` |  |
| `IsFitted` |  |
| `SupportsInverseTransform` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(Tensor<>)` | Generates a binary mask from a segmentation map. |
| `Fit(Tensor<>)` |  |
| `FitTransform(Tensor<>)` |  |
| `GetFeatureNamesOut(String[])` |  |
| `InverseTransform(Tensor<>)` |  |
| `Transform(Tensor<>)` |  |

