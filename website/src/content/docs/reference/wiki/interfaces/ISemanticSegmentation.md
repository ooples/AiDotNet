---
title: "ISemanticSegmentation<T>"
description: "Interface for semantic segmentation models that assign a class label to every pixel."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for semantic segmentation models that assign a class label to every pixel.

## For Beginners

Semantic segmentation answers "what is this pixel?" for every pixel.

Example output for a street scene:

- All road pixels labeled "road"
- All sky pixels labeled "sky"
- All car pixels labeled "car" (but car #1 and car #2 are not distinguished)

Models implementing this interface:

- SegFormer (lightweight transformer, NeurIPS 2021)
- SegNeXt (efficient CNN+attention, NeurIPS 2022)
- InternImage (large-scale CNN, CVPR 2023)
- ViT-Adapter, ViT-CoMer (transformer adapters)
- DiffCut, DiffSeg (diffusion-based, zero-shot)

## How It Works

Semantic segmentation classifies every pixel in an image into a predefined category without
distinguishing between individual object instances. For example, all cars are labeled "car"
regardless of how many there are.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetClassMap(Tensor<>)` | Gets the per-pixel class map from the most recent segmentation. |
| `GetProbabilityMap(Tensor<>)` | Gets the class-wise confidence scores for the segmentation. |

