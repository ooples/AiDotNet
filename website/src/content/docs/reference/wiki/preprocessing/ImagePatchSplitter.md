---
title: "ImagePatchSplitter<T>"
description: "Image patch splitter for image segmentation and patch-based learning tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.DomainSpecific`

Image patch splitter for image segmentation and patch-based learning tasks.

## For Beginners

In image analysis, we often work with patches (small regions)
extracted from larger images. Adjacent patches can be highly correlated,
so we need to ensure patches from the same region don't appear in both train and test.

## How It Works

**How It Works:**

1. Assume each sample is a patch with (imageId, x, y) metadata
2. Group patches by source image
3. Either split entire images, or split with spatial buffer

**When to Use:**

- Medical image segmentation
- Satellite/aerial image analysis
- Microscopy image analysis
- Any patch-based computer vision task

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ImagePatchSplitter(Double,Int32,Int32,Int32,Boolean,Int32)` | Creates a new image patch splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Split(Matrix<>,Vector<>)` |  |

