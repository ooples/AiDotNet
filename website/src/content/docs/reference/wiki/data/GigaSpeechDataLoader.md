---
title: "GigaSpeechDataLoader<T>"
description: "Loads the GigaSpeech multi-domain English ASR dataset (10K hours from audiobooks, podcasts, YouTube)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Audio.Benchmarks`

Loads the GigaSpeech multi-domain English ASR dataset (10K hours from audiobooks, podcasts, YouTube).

## How It Works

GigaSpeech expects JSON manifest files and pre-converted WAV audio:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GigaSpeechDataLoader(GigaSpeechDataLoaderOptions)` | Creates a new GigaSpeech data loader. |

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

