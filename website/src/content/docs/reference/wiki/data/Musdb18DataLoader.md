---
title: "Musdb18DataLoader<T>"
description: "Loads the MUSDB18 music source separation dataset (150 tracks, 4 stems: vocals, drums, bass, other)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Audio.Benchmarks`

Loads the MUSDB18 music source separation dataset (150 tracks, 4 stems: vocals, drums, bass, other).

## How It Works

MUSDB18 expects pre-extracted WAV stems:

Features are mixture waveform segments Tensor[N, SegmentSamples].
Labels are concatenated stem segments Tensor[N, SegmentSamples * 4] (vocals, drums, bass, other).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Musdb18DataLoader(Musdb18DataLoaderOptions)` | Creates a new MUSDB18 data loader. |

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

