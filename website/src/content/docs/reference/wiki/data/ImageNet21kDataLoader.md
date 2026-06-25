---
title: "ImageNet21kDataLoader<T>"
description: "Loads the ImageNet-21K dataset (~14.2M images, 21,841 categories from the full WordNet hierarchy)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads the ImageNet-21K dataset (~14.2M images, 21,841 categories from the full WordNet hierarchy).

## How It Works

ImageNet-21K uses the same directory structure as ImageNet-1K (synset folders), but with 21K+ categories.
The dataset must be downloaded manually from https://image-net.org/.
Use MaxClasses and MaxSamples to load manageable subsets.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ImageNet21kDataLoader(ImageNet21kDataLoaderOptions)` | Creates a new ImageNet-21K data loader. |

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

