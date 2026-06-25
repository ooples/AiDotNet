---
title: "Musdb18DataLoaderOptions"
description: "Configuration options for the MUSDB18 data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Audio.Benchmarks`

Configuration options for the MUSDB18 data loader.

## How It Works

MUSDB18 is a music source separation benchmark with 150 full-length tracks (100 train / 50 test)
with isolated stems: vocals, drums, bass, and other. Audio is stereo 44.1kHz.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `SampleRate` | Sample rate in Hz. |
| `SegmentDurationSeconds` | Segment duration in seconds (random segments are extracted from tracks). |
| `Split` | Dataset split to load. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

