---
title: "VoxelGridData<T>"
description: "Represents a voxel grid with world-space metadata."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Geometry.Data`

Represents a voxel grid with world-space metadata.

## How It Works

**For Beginners:** A voxel grid is like a 3D image made of cubes
(voxels). Each voxel stores a value, such as occupancy or density.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VoxelGridData(Tensor<>,Vector<>,Vector<>)` | Initializes a new instance of the VoxelGridData class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Channels` | Number of channels per voxel (1 if no channel dimension). |
| `Depth` | Depth (Z dimension) of the grid. |
| `Height` | Height (Y dimension) of the grid. |
| `Metadata` | Optional metadata associated with the voxel grid. |
| `Origin` | World-space origin of the voxel grid (corner of voxel [0,0,0]). |
| `VoxelSize` | Size of a voxel along each axis in world units. |
| `Voxels` | Voxel values of shape [depth, height, width] or [depth, height, width, channels]. |
| `Width` | Width (X dimension) of the grid. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeBounds` | Computes the axis-aligned bounds of the grid in world coordinates. |
| `GetVoxelCenter(Int32,Int32,Int32)` | Gets the world-space center of a voxel at the given indices. |
| `ToPointCloud(,Int32,Boolean)` | Converts occupied voxels to a point cloud using voxel centers. |

