---
title: "MaskFromBoundingBox<T>"
description: "Generates a mask from one or more bounding box regions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.MaskUtilities`

Generates a mask from one or more bounding box regions.

## For Beginners

If you have a bounding box around an object (like from an
object detector), this utility converts that box into a mask you can use for
inpainting or editing that specific region.

## How It Works

Creates a binary mask where pixels inside any of the specified bounding boxes
are masked (1) and all other pixels are unmasked (0). Useful for region-based
inpainting from object detection results.

When used as an `IDataTransformer`, the Transform method
takes an image tensor, uses its dimensions for height/width, and generates a mask from
the bounding boxes configured in the constructor.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MaskFromBoundingBox(ValueTuple<Int32,Int32,Int32,Int32>[])` | Initializes a new instance of the `MaskFromBoundingBox` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ColumnIndices` |  |
| `IsFitted` |  |
| `SupportsInverseTransform` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Tensor<>)` |  |
| `FitTransform(Tensor<>)` |  |
| `Generate(Int32,Int32,ValueTuple<Int32,Int32,Int32,Int32>[])` | Generates a mask covering the specified bounding boxes. |
| `GetFeatureNamesOut(String[])` |  |
| `InverseTransform(Tensor<>)` |  |
| `Transform(Tensor<>)` | Generates a bounding box mask using the input tensor's dimensions and the boxes configured in the constructor. |

