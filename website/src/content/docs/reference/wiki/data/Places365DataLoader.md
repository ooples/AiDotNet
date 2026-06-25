---
title: "Places365DataLoader<T>"
description: "Loads the Places365 scene recognition dataset (1.8M train / 36.5K val, 256x256 RGB, 365 scene categories)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads the Places365 scene recognition dataset (1.8M train / 36.5K val, 256x256 RGB, 365 scene categories).

## How It Works

Places365 expects a directory structure where each scene category is a subfolder:

Alternatively, expects a categories file (categories_places365.txt) and image list files
(places365_train_standard.txt, places365_val.txt) for label mapping.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Places365DataLoader(Places365DataLoaderOptions)` | Creates a new Places365 data loader. |

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

