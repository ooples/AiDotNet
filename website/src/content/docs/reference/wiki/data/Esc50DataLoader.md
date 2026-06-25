---
title: "Esc50DataLoader<T>"
description: "Loads the ESC-50 environmental sound classification dataset (2000 clips, 50 classes)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Audio.Benchmarks`

Loads the ESC-50 environmental sound classification dataset (2000 clips, 50 classes).

## How It Works

ESC-50 expects:

CSV columns: filename, fold, target, category, esc10, src_file, take.
Features are raw waveform Tensor[N, MaxSamples]. Labels are class index Tensor[N, 1].

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Esc50DataLoader(Esc50DataLoaderOptions)` | Creates a new ESC-50 data loader. |

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

