---
title: "Wonder3DModel<T>"
description: "Wonder3D model for multi-view cross-domain diffusion with simultaneous RGB and normal map generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ThreeD`

Wonder3D model for multi-view cross-domain diffusion with simultaneous RGB and normal map generation.

## For Beginners

Wonder3D creates both color views and normal maps for high-quality 3D reconstruction.

How Wonder3D works:

1. Input image is encoded by CLIP into 768-dim features
2. Shared U-Net backbone processes both RGB and normal map branches
3. Cross-domain attention exchanges information between color and geometry branches
4. 6 canonical views are generated simultaneously for both domains
5. Normal maps provide geometric detail that color images alone cannot capture
6. NeuS reconstruction combines both domains for high-quality textured meshes

Key characteristics:

- Dual-branch architecture: RGB images + normal maps
- Cross-domain attention ensures geometric consistency
- 6 canonical viewpoints for complete object coverage
- Normal maps improve reconstruction quality significantly
- SD 1.5 backbone with domain-specific adapters
- NeuS-based mesh reconstruction

When to use Wonder3D:

- High-quality single-image 3D reconstruction
- When geometric detail (from normal maps) is important
- When both texture and geometry quality matter
- Research on cross-domain multi-view generation

Limitations:

- Fixed 6-view output configuration
- Higher compute than single-branch methods (dual processing)
- Quality limited by SD 1.5 backbone resolution
- NeuS reconstruction can be slow for high-resolution meshes

## How It Works

Wonder3D generates multi-view color images and normal maps simultaneously using
cross-domain attention between RGB and normal map branches. A shared SD 1.5 backbone
processes both domains with domain-specific adapters, producing 6 canonical views
that are reconstructed into a textured 3D mesh via NeuS.

Architecture components:

- Dual-branch U-Net with shared SD 1.5 backbone (320 base channels, [1,2,4,4])
- Cross-domain attention between RGB and normal map branches
- CLIP image encoder for 768-dim conditioning
- 6 canonical viewpoints (front, back, left, right, top, bottom)
- NeuS reconstruction from cross-domain multi-view outputs
- Domain-specific adapters for RGB and normal map generation

Technical specifications:

- Architecture: Dual-branch U-Net with cross-domain attention
- Backbone: SD 1.5 (320 base channels, [1,2,4,4] multipliers)
- Cross-attention: 768-dim (CLIP)
- Output views: 6 (front, back, left, right, top, bottom)
- Output domains: RGB + normal maps
- Reconstruction: NeuS
- Default point count: 4,096
- Scheduler: DDIM with scaled linear beta

Reference: Long et al., "Wonder3D: Single Image to 3D using Cross-Domain Diffusion", CVPR 2024

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `NumViews` | Gets the number of canonical viewpoints generated simultaneously. |
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

