---
title: "LibriSpeechDataLoaderOptions"
description: "Configuration options for the LibriSpeech data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Audio.Benchmarks`

Configuration options for the LibriSpeech data loader.

## How It Works

LibriSpeech is a corpus of ~1000 hours of 16kHz English speech derived from audiobooks.
Subsets: train-clean-100, train-clean-360, train-other-500, dev-clean, dev-other, test-clean, test-other.
Audio is stored as FLAC files with corresponding text transcriptions.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `MaxDurationSeconds` | Maximum audio duration in seconds. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `SampleRate` | Sample rate in Hz. |
| `Split` | Dataset split to load. |
| `Subset` | LibriSpeech subset. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

