---
title: "MaskFromPoints<T>"
description: "Generates a mask from point prompts with circular regions, similar to SAM-style point selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.MaskUtilities`

Generates a mask from point prompts with circular regions, similar to SAM-style point selection.

## For Beginners

Instead of drawing a mask by hand, you can click on points
in an image. Each click creates a circular masked area. You can also use negative
points to "un-mask" areas within a masked region.

## How It Works

Creates a mask by placing circles of a specified radius at each point location.
Points can be positive (add to mask) or negative (remove from mask). This mirrors
the Segment Anything Model (SAM) point prompt interface.

When used as an `IDataTransformer`, the Transform method
takes an image tensor, uses its dimensions for height/width, and generates a mask from
the points configured in the constructor.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MaskFromPoints(ValueTuple<Int32,Int32>[],ValueTuple<Int32,Int32>[],Int32)` | Initializes a new instance of the `MaskFromPoints` class. |

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
| `Generate(Int32,Int32,ValueTuple<Int32,Int32>[],ValueTuple<Int32,Int32>[])` | Generates a mask from positive and negative point prompts. |
| `GetFeatureNamesOut(String[])` |  |
| `InverseTransform(Tensor<>)` |  |
| `Transform(Tensor<>)` | Generates a point mask using the input tensor's dimensions and the points configured in the constructor. |

