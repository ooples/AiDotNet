---
title: "Fsd50kDataLoaderOptions"
description: "Configuration options for the FSD50K data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Audio.Benchmarks`

Configuration options for the FSD50K data loader.

## How It Works

FSD50K (Freesound Dataset 50K) contains 51,197 audio clips with 200 sound event classes
from Freesound. Multi-label classification with hierarchical AudioSet ontology labels.
Audio clips range from 0.3s to 30s.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `MaxDurationSeconds` | Maximum audio duration in seconds. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `SampleRate` | Sample rate in Hz. |
| `Split` | Dataset split to load. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

