---
title: "VoxPopuliDataLoaderOptions"
description: "Configuration options for the VoxPopuli data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Audio.Benchmarks`

Configuration options for the VoxPopuli data loader.

## How It Works

VoxPopuli is a large-scale multilingual speech corpus from European Parliament recordings.
Contains 400K+ hours of unlabeled speech and 1800+ hours of transcribed speech in 23 languages.

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

