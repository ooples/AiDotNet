---
title: "I3DDiffusionModel<T>"
description: "Interface for 3D diffusion models that generate 3D content like point clouds, meshes, and scenes."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for 3D diffusion models that generate 3D content like point clouds, meshes, and scenes.

## For Beginners

3D diffusion models create 3D objects instead of flat images.

Types of 3D generation:

- Point Clouds: Sets of 3D points that form a shape
- Meshes: Surfaces made of triangles (like in games)
- Textured Models: Meshes with colors and materials
- Novel Views: New angles of an object from one image

How it works:

1. Text-to-3D: Describe what you want → 3D model
2. Image-to-3D: Single image → full 3D model
3. Score Distillation: Use 2D diffusion to guide 3D optimization

Applications:

- Game asset creation
- Product design visualization
- VR/AR content generation
- Architectural modeling

## How It Works

3D diffusion models extend diffusion to generate three-dimensional content,
including point clouds, meshes, textured models, and full 3D scenes. They
are used in computer graphics, game development, and virtual reality.

This interface extends `IDiffusionModel` with 3D-specific operations.

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultPointCount` | Gets the default number of points in generated point clouds. |
| `SupportsMesh` | Gets whether this model supports mesh generation. |
| `SupportsNovelView` | Gets whether this model supports novel view synthesis. |
| `SupportsPointCloud` | Gets whether this model supports point cloud generation. |
| `SupportsScoreDistillation` | Gets whether this model supports score distillation sampling (SDS). |
| `SupportsTexture` | Gets whether this model supports texture generation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ColorizePointCloud(Tensor<>,String,Int32,Nullable<Int32>)` | Adds colors/normals to a point cloud. |
| `ComputeScoreDistillationGradients(Tensor<>,String,Int32,Double)` | Computes score distillation gradients for 3D optimization. |
| `GenerateMesh(String,String,Int32,Int32,Double,Nullable<Int32>)` | Generates a mesh from a text description. |
| `GeneratePointCloud(String,String,Nullable<Int32>,Int32,Double,Nullable<Int32>)` | Generates a point cloud from a text description. |
| `ImageTo3D(Tensor<>,Int32,Int32,Double,Nullable<Int32>)` | Generates a 3D model from a single input image. |
| `PointCloudToMesh(Tensor<>,SurfaceReconstructionMethod)` | Converts a point cloud to a mesh. |
| `SynthesizeNovelViews(Tensor<>,ValueTuple<Double,Double>[],Int32,Double,Nullable<Int32>)` | Synthesizes novel views of an object from a reference image. |

