---
title: "SomethingSomethingV2DataLoaderOptions"
description: "Configuration options for the Something-Something V2 data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Video.Benchmarks`

Configuration options for the Something-Something V2 data loader.

## How It Works

Something-Something V2 contains 220K video clips of humans performing 174 pre-defined
actions with everyday objects. Temporal reasoning is required to distinguish actions.

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

