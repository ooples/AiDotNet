---
title: "ShapeNetCorePartSegmentationDataLoaderOptions"
description: "Configuration options for the ShapeNetCore part segmentation data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Geometry`

Configuration options for the ShapeNetCore part segmentation data loader.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download the dataset if not present. |
| `DataPath` | Root data path. |
| `IncludeNormals` | Whether to include normals when available. |
| `MaxSamples` | Optional cap on the number of samples to load. |
| `NumClasses` | Number of part classes in the dataset. |
| `PaddingStrategy` | Strategy for padding when fewer points exist than requested. |
| `PointsPerSample` | Number of points per sample. |
| `RandomSeed` | Optional random seed for reproducible sampling. |
| `SamplingStrategy` | Strategy for sampling points from each shape. |
| `Split` | Dataset split to load. |

