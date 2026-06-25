---
title: "VoxPopuliDataLoader<T>"
description: "Loads the VoxPopuli multilingual speech dataset from European Parliament recordings."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Audio.Benchmarks`

Loads the VoxPopuli multilingual speech dataset from European Parliament recordings.

## How It Works

VoxPopuli expects pre-downloaded audio and TSV transcription files:

Features are raw waveform samples as Tensor[N, MaxSamples].
Labels are character-encoded transcripts as Tensor[N, MaxTextLen].

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VoxPopuliDataLoader(VoxPopuliDataLoaderOptions)` | Creates a new VoxPopuli data loader. |

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

