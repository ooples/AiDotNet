---
title: "Hmdb51DataLoaderOptions"
description: "Configuration options for the HMDB51 data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Video.Benchmarks`

Configuration options for the HMDB51 data loader.

## How It Works

HMDB51 contains 6,766 video clips from 51 human action categories. Each category has
at least 101 clips. 3 train/test splits are provided.

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

