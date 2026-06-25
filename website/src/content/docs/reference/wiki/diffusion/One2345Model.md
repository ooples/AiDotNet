---
title: "One2345Model<T>"
description: "One-2-3-45 model for single-image to 3D mesh generation in 45 seconds."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ThreeD`

One-2-3-45 model for single-image to 3D mesh generation in 45 seconds.

## For Beginners

One-2-3-45 creates a 3D mesh from a single photo in ~45 seconds.

How One-2-3-45 works:

1. Input image is encoded by CLIP into 768-dim features
2. Zero123-based U-Net generates views from multiple angles
3. Each view is conditioned on relative camera pose
4. SparseNeuS reconstructs 3D mesh from the sparse multi-view images
5. Texture is mapped from generated views onto the mesh

Key characteristics:

- Two-stage: multi-view generation + 3D reconstruction
- ~45 seconds total pipeline (fast for image-to-3D)
- No per-shape optimization required
- Produces textured meshes directly
- Works from a single input image

When to use One-2-3-45:

- Quick image-to-3D reconstruction
- Single-image 3D mesh generation
- When moderate quality at fast speed is acceptable
- Prototyping 3D assets from reference images

Limitations:

- Quality limited by sparse view generation
- May struggle with complex occlusions
- Texture quality depends on view consistency
- Limited to object-centric scenes

## How It Works

One-2-3-45 uses a two-stage pipeline: Zero123-based viewpoint diffusion generates
multi-view images, then a SparseNeuS module reconstructs a textured 3D mesh from
the sparse views without per-shape optimization.

Architecture components:

- Zero123-based U-Net for multi-view generation (8 input channels, 768-dim CLIP)
- SparseNeuS module for 3D reconstruction from sparse views
- Standard SD VAE for view encoding/decoding
- 8 input channels (4 latent + 4 view-conditioned)
- DDIM scheduler for efficient multi-view generation

Technical specifications:

- Stage 1: Zero123-based U-Net (8 input channels, 768-dim CLIP)
- Stage 2: SparseNeuS reconstruction
- Base channels: 320, multipliers [1, 2, 4, 4]
- Input: 8 channels (4 latent noise + 4 view conditioning)
- Pipeline time: ~45 seconds
- Default point count: 4,096
- Scheduler: DDIM

Reference: Liu et al., "One-2-3-45: Any Single Image to 3D Mesh in 45 Seconds without Per-Shape Optimization", NeurIPS 2023

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
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
| `Clone` |  |
| `DeepCopy` |  |
| `GenerateMesh(String,String,Int32,Int32,Double,Nullable<Int32>)` |  |
| `GeneratePointCloud(String,String,Nullable<Int32>,Int32,Double,Nullable<Int32>)` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

