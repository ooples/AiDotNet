---
title: "ScanNetSemanticSegmentationDataLoaderOptions"
description: "Configuration options for the ScanNet semantic segmentation data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Geometry`

Configuration options for the ScanNet semantic segmentation data loader.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDetectLabelColumn` | Whether to auto-detect a label column in preprocessed text files. |
| `AutoDownload` | Automatically download the dataset if not present. |
| `DataPath` | Root data path. |
| `IncludeColors` | Whether to include RGB colors when available. |
| `IncludeNormals` | Whether to include normals when available. |
| `IncludeUnknownClass` | Whether to reserve an explicit unknown class at index 0. |
| `InputFormat` | Input data format selection. |
| `LabelMode` | Label mapping mode. |
| `MaxSamples` | Optional cap on the number of samples to load. |
| `NormalizeColors` | Whether to normalize colors from 0-255 to 0-1. |
| `PaddingStrategy` | Strategy for padding when fewer points exist than requested. |
| `PointsPerSample` | Number of points per sample. |
| `RandomSeed` | Optional random seed for reproducible sampling. |
| `SamplingStrategy` | Strategy for sampling points from each scene. |
| `Split` | Dataset split to load. |

