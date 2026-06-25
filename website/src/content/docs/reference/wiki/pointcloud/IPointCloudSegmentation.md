---
title: "IPointCloudSegmentation<T>"
description: "Defines functionality for point cloud segmentation tasks."
section: "API Reference"
---

`Interfaces` · `AiDotNet.PointCloud.Interfaces`

Defines functionality for point cloud segmentation tasks.

## How It Works

**For Beginners:** Point cloud segmentation assigns a label to each point in a 3D point cloud.

Think of segmentation as coloring a 3D model:

- Each point in the cloud gets assigned to a category
- Points belonging to the same object or part get the same label
- This allows you to identify and separate different components

Common segmentation tasks:

- Semantic segmentation: Label each point by object type (car, road, building, etc.)
- Instance segmentation: Separate individual objects (this car vs that car)
- Part segmentation: Identify parts of an object (chair leg, chair back, seat)

Applications:

- Autonomous driving: Identify pedestrians, vehicles, road surfaces
- Robotics: Recognize and grasp specific parts of objects
- 3D scene understanding: Parse indoor/outdoor environments

## Methods

| Method | Summary |
|:-----|:--------|
| `SegmentPointCloud(Tensor<>)` | Performs semantic segmentation on a point cloud. |

