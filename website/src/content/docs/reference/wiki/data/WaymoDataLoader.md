---
title: "WaymoDataLoader<T>"
description: "Loads the Waymo Open Dataset (LiDAR point clouds with 3D bounding boxes)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Geometry`

Loads the Waymo Open Dataset (LiDAR point clouds with 3D bounding boxes).

## How It Works

Waymo expects pre-extracted point cloud binary files:

Features are point cloud Tensor[N, PointsPerSample * Channels].
Labels are the dominant object class per frame Tensor[N, 1] (0=Vehicle, 1=Pedestrian, 2=Cyclist, 3=Sign).

## Methods

| Method | Summary |
|:-----|:--------|
| `ParseWaymoLabelFile(String,String)` | Parses a Waymo label file and returns the dominant object class index. |

