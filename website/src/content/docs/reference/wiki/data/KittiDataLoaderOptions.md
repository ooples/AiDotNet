---
title: "KittiDataLoaderOptions"
description: "Configuration options for the KITTI 3D object detection data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Geometry`

Configuration options for the KITTI 3D object detection data loader.

## How It Works

KITTI contains LiDAR point clouds from autonomous driving scenarios with 3D bounding box annotations.
Point clouds are stored as binary files (4 floats per point: x, y, z, reflectance).

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `IncludeReflectance` | Include reflectance as 4th channel. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `PointsPerSample` | Number of points per sample. |
| `Split` | Dataset split to load. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

