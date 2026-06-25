---
title: "Kinetics400DataLoaderOptions"
description: "Configuration options for the Kinetics-400 data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Video.Benchmarks`

Configuration options for the Kinetics-400 data loader.

## How It Works

Kinetics-400 contains ~300K 10-second video clips covering 400 human action classes.
Videos are sourced from YouTube. Requires pre-extracted frames or video files.

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

