---
title: "ImagePreprocessor<T>"
description: "Unified preprocessing pipeline builder for chaining image transformations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Unified preprocessing pipeline builder for chaining image transformations.

## Properties

| Property | Summary |
|:-----|:--------|
| `Transforms` | Gets the list of transforms in this pipeline. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Add(IAugmentation<,ImageTensor<>>)` | Adds a transform to the pipeline. |
| `CenterCrop(Int32,Int32)` | Adds center crop. |
| `Normalize(Double[],Double[])` | Adds normalization. |
| `Process(ImageTensor<>,AugmentationContext<>)` | Applies the entire pipeline to an image. |
| `RandomCrop(Int32,Int32)` | Adds random crop augmentation. |
| `RandomHorizontalFlip(Double)` | Adds horizontal flip augmentation. |
| `Resize(Int32,Int32,InterpolationMode)` | Adds a resize transform. |
| `ToTensor(Double)` | Adds ToTensor conversion. |

