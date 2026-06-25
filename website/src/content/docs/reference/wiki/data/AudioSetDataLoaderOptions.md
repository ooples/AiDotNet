---
title: "AudioSetDataLoaderOptions"
description: "Configuration options for the AudioSet data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Audio.Benchmarks`

Configuration options for the AudioSet data loader.

## How It Works

AudioSet is a large-scale audio event dataset with 2M+ 10-second YouTube clips
labeled with 527 audio event categories. Multi-label classification task.
Requires pre-downloaded audio (YouTube clips converted to WAV).

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `ClipDurationSeconds` | Audio clip duration in seconds. |
| `DataPath` | Root data path. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `SampleRate` | Sample rate in Hz. |
| `Split` | Dataset split to load. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

