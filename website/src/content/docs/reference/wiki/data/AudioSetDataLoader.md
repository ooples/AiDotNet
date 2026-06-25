---
title: "AudioSetDataLoader<T>"
description: "Loads the AudioSet large-scale audio event dataset (2M+ 10-second clips, 527 categories)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Audio.Benchmarks`

Loads the AudioSet large-scale audio event dataset (2M+ 10-second clips, 527 categories).

## How It Works

AudioSet expects pre-downloaded audio and CSV label files:

CSV format: YTID, start_seconds, end_seconds, positive_labels (comma-separated label IDs).
Features are raw waveform Tensor[N, MaxSamples]. Labels are multi-hot Tensor[N, 527].

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioSetDataLoader(AudioSetDataLoaderOptions)` | Creates a new AudioSet data loader. |

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

