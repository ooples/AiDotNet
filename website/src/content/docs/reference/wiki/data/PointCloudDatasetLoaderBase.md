---
title: "PointCloudDatasetLoaderBase<T>"
description: "Base class for point cloud dataset loaders that expose tensor inputs and outputs."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Data.Geometry`

Base class for point cloud dataset loaders that expose tensor inputs and outputs.

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureCount` |  |
| `FeatureDimension` | Number of features per point. |
| `OutputDimension` |  |
| `PointsPerSample` | Number of points per sample. |
| `TotalCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildSampleIndices(Int32,Int32,PointSamplingStrategy,PointPaddingStrategy,Random)` | Builds sample indices for point selection with sampling and padding strategies. |
| `ExtractBatch(Int32[])` |  |
| `SetLoadedData(Tensor<>,Tensor<>)` | Assigns loaded tensors and initializes indexing metadata. |
| `Split(Double,Double,Nullable<Int32>)` |  |

