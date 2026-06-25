---
title: "MaskFeatherer<T>"
description: "Applies feathering (soft blurring) to mask edges for smooth transitions in inpainting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.MaskUtilities`

Applies feathering (soft blurring) to mask edges for smooth transitions in inpainting.

## For Beginners

When you paint a mask for inpainting, the edges can be
very sharp (pixel is either 100% masked or 0%). Feathering blurs those edges
so the transition is gradual, producing more natural-looking inpainting results.

## How It Works

Feathering smooths the boundary between masked and unmasked regions using a
Gaussian-like blur, producing gradual transitions that prevent harsh edges
in inpainted results.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MaskFeatherer(Int32)` | Initializes a new instance of the `MaskFeatherer` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ColumnIndices` |  |
| `IsFitted` |  |
| `Radius` | Gets the feathering radius in pixels. |
| `SupportsInverseTransform` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(Tensor<>)` | Applies feathering to a single-channel mask tensor. |
| `Fit(Tensor<>)` |  |
| `FitTransform(Tensor<>)` |  |
| `GetFeatureNamesOut(String[])` |  |
| `InverseTransform(Tensor<>)` |  |
| `Transform(Tensor<>)` |  |

