---
title: "FMoWDataLoader<T>"
description: "Loads the Functional Map of the World (fMoW) satellite imagery dataset (1M+ images, 62 categories)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads the Functional Map of the World (fMoW) satellite imagery dataset (1M+ images, 62 categories).

## How It Works

fMoW expects:

Each category has sequenced image folders with RGB images and JSON metadata.
Labels are one-hot encoded as Tensor[N, 62].

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FMoWDataLoader(FMoWDataLoaderOptions)` | Creates a new fMoW data loader. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassNames` | Gets the class names. |
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

