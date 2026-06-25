---
title: "SemanticKittiDataLoaderOptions"
description: "Configuration options for the SemanticKITTI data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Geometry`

Configuration options for the SemanticKITTI data loader.

## How It Works

SemanticKITTI provides per-point semantic labels for the KITTI Odometry benchmark.
28 semantic classes for LiDAR point cloud segmentation.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `NumClasses` | Number of semantic classes. |
| `PointsPerSample` | Number of points per sample. |
| `Split` | Dataset split to load. |

