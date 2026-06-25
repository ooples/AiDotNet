---
title: "MeshOperations<T>"
description: "Provides mesh processing operations for triangle meshes."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Geometry.Preprocessing`

Provides mesh processing operations for triangle meshes.

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildVertexAdjacency(TriangleMeshData<>)` | Builds a vertex adjacency list (which vertices are connected by edges). |
| `ComputeFaceNormals(TriangleMeshData<>)` | Computes face normals for a triangle mesh. |
| `ComputeStatistics(TriangleMeshData<>)` | Computes mesh statistics (surface area, volume, bounding box). |
| `ComputeVertexNormals(TriangleMeshData<>)` | Computes vertex normals by averaging adjacent face normals. |
| `SamplePoints(TriangleMeshData<>,Int32,Nullable<Int32>,Boolean)` | Samples points uniformly from the mesh surface. |

