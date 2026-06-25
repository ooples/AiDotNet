---
title: "MVDreamModel<T>"
description: "MVDream - Multi-View Diffusion Model for 3D-consistent image generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ThreeD`

MVDream - Multi-View Diffusion Model for 3D-consistent image generation.

## For Beginners

MVDream creates multiple images of the same object
from different angles, all consistent with each other:

Example: "A red sports car"

- Front view: shows the car's front grille and headlights
- Side view: shows the profile with wheels and doors
- Back view: shows tail lights and rear design
- All views are consistent (same car, same color, same features)

This consistency is what enables 3D reconstruction:

- Multiple views + triangulation = 3D model
- Can be used with Score Distillation for high-quality 3D

## How It Works

MVDream is a multi-view diffusion model that generates 3D-consistent images
from multiple viewpoints simultaneously. It enables high-quality 3D generation
by leveraging multi-view supervision during training.

Key capabilities:

1. Multi-View Generation: Generate multiple consistent views of an object
2. Text-to-3D: Create 3D content from text descriptions
3. Image-to-3D: Convert single images to 3D representations
4. Score Distillation Sampling (SDS): Guide NeRF/3DGS optimization
5. Novel View Synthesis: Generate unseen viewpoints of objects

Technical specifications:

- Image resolution: 256x256 per view (default)
- Number of views: 4 (orthogonal) or 8 (comprehensive)
- Latent channels: 4 (Stable Diffusion compatible)
- Context dimension: 1024 (CLIP/T5 embeddings)
- Camera model: Spherical coordinates (azimuth, elevation, radius)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MVDreamModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,MultiViewUNet<>,StandardVAE<>,IConditioningModule<>,IConditioningModule<>,MVDreamConfig,Nullable<Int32>)` | Initializes a new MVDream model with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CameraEmbedding` | Gets the camera embedding module. |
| `Conditioner` |  |
| `Config` | Gets the model configuration. |
| `ImageConditioner` | Gets the image conditioner for image-to-3D tasks. |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `SupportsMesh` |  |
| `SupportsNovelView` |  |
| `SupportsPointCloud` |  |
| `SupportsScoreDistillation` |  |
| `SupportsTexture` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildCameraMatrix(Double,Double,Double)` | Builds a camera matrix from spherical coordinates. |
| `Clone` |  |
| `CombineConditions(Tensor<>,Tensor<>)` | Combines text/image conditioning with camera embedding. |
| `ComputeScoreDistillationGradients(Tensor<>,String,Int32,Double)` | Computes Score Distillation Sampling gradients for 3D optimization. |
| `DeepCopy` |  |
| `EstimateDepthFromView(Tensor<>,Int32,Int32)` | Estimates depth from a single view using gradient-based cues. |
| `ExtractSilhouette(Tensor<>,Int32,Int32)` | Extracts a silhouette mask identifying foreground pixels. |
| `ExtractView(Tensor<>,Int32)` | Extracts a single view from multi-view tensor. |
| `ExtractViewFromBatch(Tensor<>,Int32)` | Extracts a single view from a batch tensor with batch dimension. |
| `FilterPointCloud(List<ValueTuple<Double,Double,Double,Double>>,Int32)` | Filters and downsamples a point cloud using voxel grid filtering. |
| `FlattenLatent(Tensor<>)` | Flattens latent for conditioning. |
| `GenerateCameraPositions(Int32,Double)` | Generates uniformly distributed camera positions around the object. |
| `GenerateMultiView(String,String,Int32,Int32,Double,Double,Nullable<Int32>)` | Generates multiple consistent views from a text prompt. |
| `GenerateNovelViewsFromImage(Tensor<>,Int32,Int32,Double,Nullable<Int32>)` | Generates novel views from a single input image. |
| `GetParameters` |  |
| `ImageTo3D(Tensor<>,Int32,Int32,Double,Nullable<Int32>)` | Generates 3D from a single input image. |
| `InitializeLayers(MultiViewUNet<>,StandardVAE<>,Nullable<Int32>)` | Initializes the model layers, using provided components or creating defaults. |
| `ReconstructFromMultiView(Tensor<>[],ValueTuple<Double,Double,Double>[])` | Reconstructs mesh from multiple views using depth-based back-projection and visual hull carving. |
| `SetParameters(Vector<>)` |  |
| `SetView(Tensor<>,Int32,Tensor<>)` | Sets a single view in multi-view tensor. |
| `TransformCameraToWorld(Double,Double,Double,Double,Double,Double)` | Transforms a point from camera space to world space. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DEFAULT_CAMERA_DISTANCE` | Default camera distance from object. |
| `MVDREAM_BASE_CHANNELS` | MVDream base channels for U-Net. |
| `MVDREAM_CONTEXT_DIM` | Context dimension for text/image conditioning. |
| `MVDREAM_IMAGE_SIZE` | Default image resolution for each view. |
| `MVDREAM_LATENT_CHANNELS` | MVDream latent channels (Stable Diffusion compatible). |
| `_cameraEmbedding` | Camera embedding layer. |
| `_config` | Model configuration. |
| `_imageConditioner` | Image conditioning module for image-to-3D. |
| `_imageVAE` | The VAE for image encoding/decoding. |
| `_multiViewUNet` | The multi-view aware U-Net noise predictor. |
| `_textConditioner` | Text conditioning module (CLIP/T5). |

