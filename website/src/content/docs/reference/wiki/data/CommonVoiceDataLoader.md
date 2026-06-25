---
title: "CommonVoiceDataLoader<T>"
description: "Loads the Mozilla Common Voice multilingual speech dataset (19K+ hours, 100+ languages)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Audio.Benchmarks`

Loads the Mozilla Common Voice multilingual speech dataset (19K+ hours, 100+ languages).

## How It Works

Common Voice expects pre-converted WAV files (original MP3 format not supported):

The TSV files contain columns: client_id, path, sentence, up_votes, down_votes, age, gender, accents, locale, segment.
Features are raw waveform Tensor[N, MaxSamples]. Labels are character-encoded transcripts Tensor[N, MaxTextLen].

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CommonVoiceDataLoader(CommonVoiceDataLoaderOptions)` | Creates a new Common Voice data loader. |

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

