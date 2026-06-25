---
title: "Esc50DataLoaderOptions"
description: "Configuration options for the ESC-50 data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Audio.Benchmarks`

Configuration options for the ESC-50 data loader.

## How It Works

ESC-50 (Environmental Sound Classification) contains 2,000 5-second environmental audio
recordings organized into 50 classes (e.g., dog bark, rain, clock tick) with 40 clips per class.
5 predefined cross-validation folds.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `SampleRate` | Sample rate in Hz. |
| `Split` | Dataset split to load. |
| `TestFold` | Cross-validation fold to use as test (1-5). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

