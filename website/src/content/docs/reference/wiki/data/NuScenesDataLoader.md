---
title: "NuScenesDataLoader<T>"
description: "Loads the nuScenes dataset (LiDAR point clouds with 3D bounding box annotations)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Geometry`

Loads the nuScenes dataset (LiDAR point clouds with 3D bounding box annotations).

## How It Works

nuScenes expects pre-extracted point cloud binary files:

Features are point cloud Tensor[N, PointsPerSample * Channels].
Labels are the dominant object class per frame Tensor[N, 1] (0=car, 1=truck, 2=bus,
3=trailer, 4=construction_vehicle, 5=pedestrian, 6=motorcycle, 7=bicycle,
8=traffic_cone, 9=barrier). Parsed from lidarseg or text label files.

## Methods

| Method | Summary |
|:-----|:--------|
| `ParseNuScenesLabel(String,String,Boolean)` | Parses a nuScenes label and returns the dominant object class. |

