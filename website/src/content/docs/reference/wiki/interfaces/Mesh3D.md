---
title: "Mesh3D<T>"
description: "Represents a 3D mesh with vertices, faces, and optional textures."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

Represents a 3D mesh with vertices, faces, and optional textures.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Mesh3D` | Initializes a new empty mesh. |
| `Mesh3D(Tensor<>,Int32[0:,0:])` | Initializes a mesh with vertices and faces. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Colors` | Gets or sets the vertex colors [numVertices, 3] (optional). |
| `Faces` | Gets or sets the face indices [numFaces, 3] for triangular faces. |
| `Normals` | Gets or sets the vertex normals [numVertices, 3] (optional). |
| `TextureImage` | Gets or sets the texture image [height, width, channels] (optional). |
| `UVCoordinates` | Gets or sets the UV texture coordinates [numVertices, 2] (optional). |
| `Vertices` | Gets or sets the vertex positions [numVertices, 3]. |

