---
title: "FleursDataLoaderOptions"
description: "Configuration options for the FLEURS data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Audio.Benchmarks`

Configuration options for the FLEURS data loader.

## How It Works

FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech) is a
multilingual speech benchmark covering 102 languages with ~12 hours per language.
Each utterance has a text transcription.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `Language` | Language code. |
| `MaxDurationSeconds` | Maximum audio duration in seconds. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `SampleRate` | Sample rate in Hz. |
| `Split` | Dataset split to load. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

