---
title: "MaskBlur<T>"
description: "Applies Gaussian blur to a mask for smooth transitions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.MaskUtilities`

Applies Gaussian blur to a mask for smooth transitions.

## For Beginners

This blurs a mask like blurring a photo — sharp edges become
soft gradients. It's similar to feathering but affects the whole mask, not just edges.
Use this when you want an overall softer mask with gradual transitions everywhere.

## How It Works

Blurs the entire mask using a Gaussian kernel, softening all edges and transitions.
Unlike feathering which targets edges specifically, blur affects the entire mask
uniformly. Useful for creating soft masks from hard binary masks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MaskBlur(Double)` | Initializes a new instance of the `MaskBlur` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ColumnIndices` |  |
| `IsFitted` |  |
| `SupportsInverseTransform` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(Tensor<>)` | Applies Gaussian blur to a mask tensor. |
| `Fit(Tensor<>)` |  |
| `FitTransform(Tensor<>)` |  |
| `GetFeatureNamesOut(String[])` |  |
| `InverseTransform(Tensor<>)` |  |
| `Transform(Tensor<>)` |  |

