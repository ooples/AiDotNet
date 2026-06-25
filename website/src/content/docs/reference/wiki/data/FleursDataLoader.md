---
title: "FleursDataLoader<T>"
description: "Loads the FLEURS multilingual speech benchmark (102 languages, ~12 hours per language)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Audio.Benchmarks`

Loads the FLEURS multilingual speech benchmark (102 languages, ~12 hours per language).

## How It Works

FLEURS expects:

TSV columns: id, file_name, raw_transcription, transcription, num_samples, gender.
Features are raw waveform Tensor[N, MaxSamples]. Labels are character-encoded transcripts Tensor[N, MaxTextLen].

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FleursDataLoader(FleursDataLoaderOptions)` | Creates a new FLEURS data loader. |

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

