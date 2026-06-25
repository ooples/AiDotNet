---
title: "PointCloudNormalization<T>"
description: "Provides standardization and normalization utilities for 3D point cloud data."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Geometry.Preprocessing`

Provides standardization and normalization utilities for 3D point cloud data.

## Methods

| Method | Summary |
|:-----|:--------|
| `Center(PointCloudData<>)` | Centers a point cloud at the origin by subtracting the centroid. |
| `NormalizeColors(PointCloudData<>,Int32,Double)` | Normalizes color values to the range [0, 1]. |
| `ScaleToUnitCube(PointCloudData<>,Boolean)` | Scales a point cloud to fit within a unit cube [-0.5, 0.5]^3. |
| `ScaleToUnitSphere(PointCloudData<>,Boolean)` | Scales a point cloud to fit within a unit sphere (radius = 1). |

