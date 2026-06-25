---
title: "UrbanSound8kDataLoader<T>"
description: "Loads the UrbanSound8K environmental sound dataset (8732 clips, 10 classes, 10 folds)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Audio.Benchmarks`

Loads the UrbanSound8K environmental sound dataset (8732 clips, 10 classes, 10 folds).

## How It Works

UrbanSound8K expects:

CSV columns: slice_file_name, fsID, start, end, salience, fold, classID, class.
Features are raw waveform Tensor[N, MaxSamples]. Labels are one-hot Tensor[N, 10].

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UrbanSound8kDataLoader(UrbanSound8kDataLoaderOptions)` | Creates a new UrbanSound8K data loader. |

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

