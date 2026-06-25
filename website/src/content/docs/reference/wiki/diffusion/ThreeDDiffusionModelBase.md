---
title: "ThreeDDiffusionModelBase<T>"
description: "Base class for 3D diffusion models that generate 3D content like point clouds, meshes, and scenes."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Diffusion`

Base class for 3D diffusion models that generate 3D content like point clouds, meshes, and scenes.

## For Beginners

This is the foundation for 3D generation models like Point-E and Shap-E.
It extends diffusion to create 3D objects instead of 2D images.

## How It Works

This abstract base class provides common functionality for all 3D diffusion models,
including point cloud generation, mesh generation, image-to-3D, novel view synthesis,
and score distillation sampling.

Types of 3D generation:

- Point Clouds: Sets of 3D points that form a shape
- Meshes: Surfaces made of triangles (like in video games)
- Textured Models: Meshes with colors and materials
- Novel Views: New angles of an object from one image

How 3D diffusion works:

1. Text-to-3D: Describe what you want and get a 3D model
2. Image-to-3D: Turn a single photo into a full 3D model
3. Score Distillation: Use 2D diffusion knowledge to guide 3D optimization

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ThreeDDiffusionModelBase(DiffusionModelOptions<>,INoiseScheduler<>,Int32,NeuralNetworkArchitecture<>)` | Initializes a new instance of the ThreeDDiffusionModelBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CoordinateScale` | Gets the coordinate scale for normalizing 3D positions. |
| `DefaultPointCount` |  |
| `SupportsMesh` |  |
| `SupportsNovelView` |  |
| `SupportsPointCloud` |  |
| `SupportsScoreDistillation` |  |
| `SupportsTexture` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ColorizePointCloud(Tensor<>,String,Int32,Nullable<Int32>)` |  |
| `CombineImageAndViewConditioning(Tensor<>,Tensor<>)` | Combines image latent with view embedding for conditioning. |
| `ComputeScoreDistillationGradients(Tensor<>,String,Int32,Double)` |  |
| `ConcatenatePointsAndColors(Tensor<>,Tensor<>)` | Concatenates point positions with RGB colors. |
| `CreateSimpleMeshFromPoints(Tensor<>)` | Creates a simple triangulated mesh from point cloud using nearest neighbors. |
| `CreateViewEmbedding(Double,Double)` | Creates a view embedding from azimuth and elevation angles. |
| `GenerateMesh(String,String,Int32,Int32,Double,Nullable<Int32>)` |  |
| `GeneratePointCloud(String,String,Nullable<Int32>,Int32,Double,Nullable<Int32>)` |  |
| `GenerateViewAngles(Int32)` | Generates evenly distributed view angles around an object. |
| `ImageTo3D(Tensor<>,Int32,Int32,Double,Nullable<Int32>)` |  |
| `NormalizeColors(Tensor<>)` | Normalizes color values to [0, 1] range. |
| `NormalizePointCloud(Tensor<>)` | Normalizes point cloud coordinates to specified range. |
| `PointCloudToMesh(Tensor<>,SurfaceReconstructionMethod)` |  |
| `PointCloudToMeshAlphaShape(Tensor<>)` | Converts point cloud to mesh using alpha shape algorithm. |
| `PointCloudToMeshBallPivoting(Tensor<>)` | Converts point cloud to mesh using ball pivoting algorithm. |
| `PointCloudToMeshMarchingCubes(Tensor<>)` | Converts point cloud to mesh using marching cubes on a voxel grid. |
| `PointCloudToMeshPoisson(Tensor<>)` | Converts point cloud to mesh using Poisson surface reconstruction. |
| `PredictColorNoise(Tensor<>,Int32,Tensor<>,Tensor<>)` | Predicts noise for point cloud colorization. |
| `PredictNovelViewNoise(Tensor<>,Int32,Tensor<>,Tensor<>,Double)` | Predicts noise for novel view synthesis. |
| `PredictPointCloudNoise(Tensor<>,Int32,Tensor<>)` | Predicts noise for point cloud denoising. |
| `ReconstructFromViews(Tensor<>[],ValueTuple<Double,Double>[])` | Reconstructs a 3D mesh from multiple view images. |
| `SynthesizeNovelViews(Tensor<>,ValueTuple<Double,Double>[],Int32,Double,Nullable<Int32>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_defaultPointCount` | Default number of points in generated point clouds. |

