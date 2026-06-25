---
title: "ImageNet1kDataLoader<T>"
description: "Loads the ImageNet-1K (ILSVRC 2012) image classification dataset (~1.28M train / 50K val, 1000 classes)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads the ImageNet-1K (ILSVRC 2012) image classification dataset (~1.28M train / 50K val, 1000 classes).

## How It Works

ImageNet-1K expects a directory structure of:

Images are loaded as RGB, resized to ImageSize x ImageSize, and optionally normalized to [0,1].
Labels are one-hot encoded across 1000 classes, ordered by synset ID alphabetically.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ImageNet1kDataLoader(ImageNet1kDataLoaderOptions)` | Creates a new ImageNet-1K data loader. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `FeatureCount` |  |
| `Name` |  |
| `OutputDimension` |  |
| `TotalCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

