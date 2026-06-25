---
title: "Ucf101DataLoaderOptions"
description: "Configuration options for the UCF101 data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Video.Benchmarks`

Configuration options for the UCF101 data loader.

## How It Works

UCF101 contains 13,320 video clips from 101 action categories. Videos are sourced from
YouTube with realistic camera motion and varying conditions.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `FrameHeight` | Frame height. |
| `FrameWidth` | Frame width. |
| `FramesPerVideo` | Number of frames to sample per video. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `Normalize` | Whether to normalize pixel values to [0, 1]. |
| `Split` | Dataset split to load. |
| `SplitNumber` | Split number (1, 2, or 3). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

