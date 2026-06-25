---
title: "NuScenesDataLoaderOptions"
description: "Configuration options for the nuScenes data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Geometry`

Configuration options for the nuScenes data loader.

## How It Works

nuScenes is a large-scale autonomous driving dataset with LiDAR, camera, and radar data.
Contains 1000 scenes with full 3D bounding box annotations for 23 object classes.

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

