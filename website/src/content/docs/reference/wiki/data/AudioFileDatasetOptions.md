---
title: "AudioFileDatasetOptions"
description: "Configuration options for the `AudioFileDataset`."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Audio`

Configuration options for the `AudioFileDataset`.

## Properties

| Property | Summary |
|:-----|:--------|
| `DurationSeconds` | Maximum duration in seconds. |
| `Extensions` | File extensions to include. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `Mono` | Whether to convert to mono. |
| `Normalize` | Whether to normalize audio values to [-1, 1]. |
| `RandomSeed` | Optional random seed for reproducible sampling. |
| `RootDirectory` | Root directory containing audio files or class subdirectories. |
| `SampleRate` | Target sample rate in Hz. |
| `UseDirectoryLabels` | Whether class labels are determined by subdirectory names. |

