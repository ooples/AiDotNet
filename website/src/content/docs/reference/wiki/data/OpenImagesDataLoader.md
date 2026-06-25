---
title: "OpenImagesDataLoader<T>"
description: "Loads the Open Images V7 object detection dataset (~9M images, 600 categories)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads the Open Images V7 object detection dataset (~9M images, 600 categories).

## How It Works

Open Images expects:

The CSV annotation format contains: ImageID, Source, LabelName, Confidence, XMin, XMax, YMin, YMax.
Bounding box coordinates are already normalized to [0, 1].
Labels are stored as Tensor[N, MaxDetections, 5] (class_index, xmin, ymin, width, height).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OpenImagesDataLoader(OpenImagesDataLoaderOptions)` | Creates a new Open Images data loader. |

## Properties

| Property | Summary |
|:-----|:--------|
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

