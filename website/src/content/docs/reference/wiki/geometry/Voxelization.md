---
title: "Voxelization<T>"
description: "Provides voxelization utilities for converting point clouds and meshes to voxel grids."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Geometry.Preprocessing`

Provides voxelization utilities for converting point clouds and meshes to voxel grids.

## Methods

| Method | Summary |
|:-----|:--------|
| `ClampValue(Int32,Int32,Int32)` | Clamps a value between a minimum and maximum. |
| `Dilate(VoxelGridData<>,Int32)` | Applies morphological dilation to a voxel grid. |
| `Erode(VoxelGridData<>,Int32)` | Applies morphological erosion to a voxel grid. |
| `IntersectionOverUnion(VoxelGridData<>,VoxelGridData<>,Double)` | Computes the Intersection over Union (IoU) between two voxel grids. |
| `VoxelizeMeshSurface(TriangleMeshData<>,Int32,Double)` | Converts a triangle mesh to an occupancy voxel grid using surface voxelization. |
| `VoxelizePointCloud(PointCloudData<>,Int32,Double)` | Converts a point cloud to an occupancy voxel grid. |

