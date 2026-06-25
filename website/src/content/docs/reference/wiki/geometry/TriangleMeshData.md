---
title: "TriangleMeshData<T>"
description: "Represents a triangle mesh with vertex and face attributes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Geometry.Data`

Represents a triangle mesh with vertex and face attributes.

## How It Works

**For Beginners:** A triangle mesh is a surface defined by vertices
(points in 3D space) and faces (triangles that connect three vertices).
Meshes are a common way to represent 3D objects in graphics and AI.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TriangleMeshData(Tensor<>,Tensor<Int32>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Boolean)` | Initializes a new instance of the TriangleMeshData class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FaceNormals` | Optional per-face normals of shape [numFaces, 3]. |
| `Faces` | Triangle indices of shape [numFaces, 3]. |
| `Metadata` | Optional metadata associated with the mesh (units, source, labels, etc.). |
| `NumFaces` | Number of faces in the mesh. |
| `NumVertices` | Number of vertices in the mesh. |
| `VertexColors` | Optional per-vertex colors of shape [numVertices, 3] or [numVertices, 4]. |
| `VertexNormals` | Optional per-vertex normals of shape [numVertices, 3]. |
| `VertexUVs` | Optional per-vertex UV coordinates of shape [numVertices, 2]. |
| `Vertices` | Vertex positions of shape [numVertices, 3]. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeBounds` | Computes the axis-aligned bounding box for the mesh. |
| `ComputeFaceNormals` | Computes and stores per-face normals. |
| `ComputeVertexNormals` | Computes and stores per-vertex normals by averaging adjacent face normals. |
| `ToPointCloud(Boolean,Boolean,Boolean)` | Converts the mesh to a point cloud using vertex positions and optional attributes. |

