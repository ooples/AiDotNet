---
title: "GigaSpeechDataLoaderOptions"
description: "Configuration options for the GigaSpeech data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Audio.Benchmarks`

Configuration options for the GigaSpeech data loader.

## How It Works

GigaSpeech is a multi-domain English ASR dataset with 10,000 hours of labeled audio
from audiobooks, podcasts, and YouTube. Subsets: XS (10h), S (250h), M (1000h), L (2500h), XL (10000h).

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `MaxDurationSeconds` | Maximum audio duration in seconds. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `SampleRate` | Sample rate in Hz. |
| `Split` | Dataset split to load. |
| `Subset` | Subset to load. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

