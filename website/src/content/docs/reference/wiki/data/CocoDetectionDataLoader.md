---
title: "CocoDetectionDataLoader<T>"
description: "Loads the COCO 2017 object detection dataset (118K train / 5K val, 80 categories)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads the COCO 2017 object detection dataset (118K train / 5K val, 80 categories).

## How It Works

COCO Detection expects:

Labels are stored as Tensor[N, MaxDetections, 5] where each detection is (class_id, x, y, w, h)
with bounding box coordinates normalized to [0, 1]. Unused detection slots are zero-padded.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CocoDetectionDataLoader(CocoDetectionDataLoaderOptions)` | Creates a new COCO Detection data loader. |

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

