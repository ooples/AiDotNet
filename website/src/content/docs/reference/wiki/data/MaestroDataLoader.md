---
title: "MaestroDataLoader<T>"
description: "Loads the MAESTRO piano performance dataset (~200 hours, aligned MIDI and audio)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Audio.Benchmarks`

Loads the MAESTRO piano performance dataset (~200 hours, aligned MIDI and audio).

## How It Works

MAESTRO expects:

CSV columns: canonical_composer, canonical_title, split, year, midi_filename, audio_filename, duration.
Features are raw waveform Tensor[N, MaxSamples]. Labels are MIDI note activations Tensor[N, 128] (one per MIDI note).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MaestroDataLoader(MaestroDataLoaderOptions)` | Creates a new MAESTRO data loader. |

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
| `ParseCsvLine(String)` | Parses a CSV line respecting quoted fields (handles commas and escaped quotes inside quotes). |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

