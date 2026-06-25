---
title: "VideoFrameDataset<T>"
description: "Loads videos represented as directories of sequentially numbered image frames."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Video`

Loads videos represented as directories of sequentially numbered image frames.

## How It Works

This loader expects videos stored as frame directories (the standard preprocessing format
for ML video datasets like UCF-101, Kinetics, HMDB51, etc.). Each video is a directory
containing sequentially numbered image frames (BMP, PPM, or PGM format).

Expected structure:

Frames are sampled uniformly across the video duration to produce a fixed number of frames
per video. Output tensor shape is [N, FramesPerVideo, Height, Width, Channels].

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VideoFrameDataset(VideoFrameDatasetOptions)` | Creates a new VideoFrameDataset with the specified options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassNames` | Gets the class names discovered from directory names. |
| `Description` |  |
| `FeatureCount` |  |
| `Name` |  |
| `OutputDimension` |  |
| `TotalCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

