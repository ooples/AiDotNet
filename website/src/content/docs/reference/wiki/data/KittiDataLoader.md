---
title: "KittiDataLoader<T>"
description: "Loads the KITTI 3D object detection dataset (LiDAR point clouds with 3D bounding boxes)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Geometry`

Loads the KITTI 3D object detection dataset (LiDAR point clouds with 3D bounding boxes).

## How It Works

KITTI expects:

Features are point cloud Tensor[N, PointsPerSample * Channels].
Labels are the dominant object class per frame Tensor[N, 1] (0=Car, 1=Van, 2=Truck,
3=Pedestrian, 4=Person_sitting, 5=Cyclist, 6=Tram, 7=Misc).

## Methods

| Method | Summary |
|:-----|:--------|
| `ParseKittiLabelFile(String,String)` | Parses a KITTI label file and returns the dominant object class index. |

