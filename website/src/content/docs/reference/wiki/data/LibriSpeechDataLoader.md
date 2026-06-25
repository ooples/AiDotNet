---
title: "LibriSpeechDataLoader<T>"
description: "Loads the LibriSpeech automatic speech recognition dataset (~1000 hours of 16kHz English speech)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Audio.Benchmarks`

Loads the LibriSpeech automatic speech recognition dataset (~1000 hours of 16kHz English speech).

## How It Works

LibriSpeech expects:

Features are raw waveform samples as Tensor[N, MaxSamples] (mono, 16kHz).
Labels are transcript text encoded as character indices in Tensor[N, MaxTextLen].
For pre-converted WAV files, audio is loaded directly. FLAC requires pre-conversion to WAV.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LibriSpeechDataLoader(LibriSpeechDataLoaderOptions)` | Creates a new LibriSpeech data loader. |

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

