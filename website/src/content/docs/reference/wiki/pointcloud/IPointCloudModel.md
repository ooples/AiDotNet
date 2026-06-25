---
title: "IPointCloudModel<T>"
description: "Defines the core functionality for point cloud processing models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.PointCloud.Interfaces`

Defines the core functionality for point cloud processing models.

## How It Works

**For Beginners:** A point cloud is a collection of 3D points that represent the surface of an object or scene.

Think of a point cloud as a 3D scan of the real world:

- Each point has X, Y, Z coordinates representing its position in 3D space
- Points can also have additional features like color, intensity, or surface normals
- Point clouds are commonly collected by LIDAR sensors, depth cameras, or 3D scanners

Common applications:

- Autonomous vehicles use LIDAR to create point clouds of their surroundings
- Robotics uses point clouds for object recognition and manipulation
- AR/VR applications use point clouds for 3D reconstruction
- Architecture and construction use point clouds for building modeling

This interface defines operations for processing point cloud data with neural networks.

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractGlobalFeatures(Tensor<>)` | Extracts global features from a point cloud. |
| `ExtractPointFeatures(Tensor<>)` | Extracts per-point features from a point cloud. |

