---
title: "AudioFileDataset<T>"
description: "Loads audio files from directories for audio classification and processing tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Audio`

Loads audio files from directories for audio classification and processing tasks.

## For Beginners

Organize your audio files into class folders (like ImageFolder),
and this loader reads the raw waveforms into tensors for training.

## How It Works

Supports WAV and raw PCM files. Audio is loaded as raw waveform samples
and can optionally be converted to mono and normalized.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioFileDataset(AudioFileDatasetOptions)` | Creates a new AudioFileDataset with the specified options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassNames` | Gets the class names discovered from directory names. |
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

