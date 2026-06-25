---
title: "WaymoDataLoaderOptions"
description: "Configuration options for the Waymo Open Dataset data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Geometry`

Configuration options for the Waymo Open Dataset data loader.

## How It Works

Waymo Open Dataset contains high-quality LiDAR and camera data from autonomous driving.
Includes 3D bounding boxes for vehicles, pedestrians, and cyclists.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `IncludeIntensity` | Include intensity as 4th channel. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `PointsPerSample` | Number of points per sample. |
| `Split` | Dataset split to load. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

