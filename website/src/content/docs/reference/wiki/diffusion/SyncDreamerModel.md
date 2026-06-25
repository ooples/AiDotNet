---
title: "SyncDreamerModel<T>"
description: "SyncDreamer model for synchronized multi-view diffusion with 3D-consistent generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ThreeD`

SyncDreamer model for synchronized multi-view diffusion with 3D-consistent generation.

## For Beginners

SyncDreamer creates consistent 3D views from a single image.

How SyncDreamer works:

1. Input image is encoded by CLIP into 768-dim features
2. 16 viewpoint branches share the same U-Net weights
3. 3D-aware attention synchronizes features between all views
4. Volume attention ensures spatial consistency across viewpoints
5. All 16 views are denoised simultaneously for 3D consistency
6. NeuS reconstruction creates a mesh from the consistent views

Key characteristics:

- 16 synchronized views generated simultaneously
- 3D-aware attention prevents inconsistent geometry
- Volume attention for global spatial understanding
- Single-image input to multi-view output
- NeuS-based mesh extraction

When to use SyncDreamer:

- Multi-view consistent image generation
- Single-image 3D reconstruction
- Research on 3D-consistent diffusion
- When view consistency is critical

Limitations:

- Fixed 16-view output configuration
- Higher compute than single-view methods
- Quality limited by SD 1.5 backbone resolution
- NeuS reconstruction can be slow

## How It Works

SyncDreamer generates multiple 3D-consistent views simultaneously by synchronizing
intermediate features across viewpoints during the diffusion process. A 3D-aware
feature attention mechanism and volume attention ensure spatial consistency.

Architecture components:

- Synchronized U-Net with shared weights across 16 viewpoints
- 3D-aware feature attention for cross-view consistency
- Volume attention for spatial understanding across views
- SD 1.5 backbone (320 base channels, 768-dim CLIP)
- NeuS reconstruction from synchronized multi-view outputs

Technical specifications:

- Architecture: Synchronized U-Net with 3D-aware and volume attention
- Backbone: SD 1.5 (320 base channels, [1,2,4,4] multipliers)
- Cross-attention: 768-dim (CLIP)
- Synchronized views: 16
- Reconstruction: NeuS
- Default point count: 4,096
- Scheduler: DDIM

Reference: Liu et al., "SyncDreamer: Generating Multiview-consistent Images from a Single-view Image", ICLR 2024

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `NumViews` | Gets the number of synchronized views generated simultaneously. |
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
| `Clone` |  |
| `DeepCopy` |  |
| `GenerateMesh(String,String,Int32,Int32,Double,Nullable<Int32>)` |  |
| `GeneratePointCloud(String,String,Nullable<Int32>,Int32,Double,Nullable<Int32>)` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

