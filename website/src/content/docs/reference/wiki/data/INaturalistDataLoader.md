---
title: "INaturalistDataLoader<T>"
description: "Loads the iNaturalist species classification dataset (~2.7M images, 10,000 species in 2021 version)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads the iNaturalist species classification dataset (~2.7M images, 10,000 species in 2021 version).

## How It Works

iNaturalist uses a JSON annotation file with COCO-style format:

The dataset has a long-tailed class distribution, making it valuable for imbalanced learning research.
Labels are one-hot encoded across the number of species categories.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `INaturalistDataLoader(INaturalistDataLoaderOptions)` | Creates a new iNaturalist data loader. |

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

