---
title: "MaestroDataLoaderOptions"
description: "Configuration options for the MAESTRO data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Audio.Benchmarks`

Configuration options for the MAESTRO data loader.

## How It Works

MAESTRO (MIDI and Audio Edited for Synchronous TRacks and Organization) contains ~200 hours
of piano performances with aligned MIDI and audio (WAV, 44.1kHz stereo).
Used for piano transcription and music generation tasks.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `MaxDurationSeconds` | Maximum audio duration in seconds. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `SampleRate` | Sample rate in Hz. |
| `Split` | Dataset split to load. |
| `Version` | MAESTRO version. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

