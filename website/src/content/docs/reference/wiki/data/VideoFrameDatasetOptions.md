---
title: "VideoFrameDatasetOptions"
description: "Configuration options for the `VideoFrameDataset`."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Video`

Configuration options for the `VideoFrameDataset`.

## How It Works

Videos are represented as directories of sequentially numbered image frames.
This is the standard format for preprocessed ML video datasets (UCF-101, Kinetics, etc.).
Structure: root/class_name/video_name/frame_001.bmp, frame_002.bmp, ...

## Properties

| Property | Summary |
|:-----|:--------|
| `Channels` | Number of color channels per frame. |
| `FrameExtensions` | File extensions for frame images. |
| `FrameHeight` | Target frame height in pixels. |
| `FrameWidth` | Target frame width in pixels. |
| `FramesPerVideo` | Number of frames to sample per video. |
| `MaxSamples` | Optional maximum number of videos to load. |
| `NormalizePixels` | Whether to normalize pixel values to [0, 1]. |
| `RandomSeed` | Optional random seed for reproducible sampling. |
| `RootDirectory` | Root directory containing class subdirectories, each with video subdirectories containing frame images. |
| `UseDirectoryLabels` | Whether class labels are determined by parent directory names. |

