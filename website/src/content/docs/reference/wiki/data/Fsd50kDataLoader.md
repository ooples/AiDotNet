---
title: "Fsd50kDataLoader<T>"
description: "Loads the FSD50K audio event dataset (51,197 clips, 200 sound event classes)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Audio.Benchmarks`

Loads the FSD50K audio event dataset (51,197 clips, 200 sound event classes).

## How It Works

FSD50K expects:

CSV columns: fname, labels (comma-separated class names), mids, split.
Features are raw waveform Tensor[N, MaxSamples]. Labels are multi-hot Tensor[N, 200].

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Fsd50kDataLoader(Fsd50kDataLoaderOptions)` | Creates a new FSD50K data loader. |

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
| `ParseCsvLine(String)` | Parses a CSV line respecting quoted fields (handles commas inside quotes). |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

