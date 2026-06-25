---
title: "PascalVocDataLoader<T>"
description: "Loads the Pascal VOC object detection dataset (20 categories, XML annotations)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads the Pascal VOC object detection dataset (20 categories, XML annotations).

## How It Works

Pascal VOC expects a standard directory structure:

Each XML annotation file contains bounding boxes with class names.
Labels are stored as Tensor[N, MaxDetections, 5] (class_id, x, y, w, h) normalized to [0, 1].

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PascalVocDataLoader(PascalVocDataLoaderOptions)` | Creates a new Pascal VOC data loader. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassNames` | Gets the class names. |
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

