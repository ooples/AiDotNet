---
title: "Magic3DModel<T>"
description: "Magic3D model for high-quality text-to-3D generation using score distillation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ThreeD`

Magic3D model for high-quality text-to-3D generation using score distillation.

## For Beginners

Magic3D creates 3D models from text descriptions:

How Magic3D works:

1. Coarse stage: Optimize a neural radiance field (NeRF) using SDS from a low-res diffusion model
2. Fine stage: Convert NeRF to a mesh and optimize with SDS from a high-res latent diffusion model
3. Result: High-quality textured 3D mesh

Key characteristics:

- Two-stage coarse-to-fine optimization
- Coarse stage: NeRF + low-res SDS (64x64 base model)
- Fine stage: DMTet mesh + high-res SDS (latent diffusion)
- 2x faster than DreamFusion
- 8x higher resolution meshes than DreamFusion
- Uses both pixel-space and latent-space diffusion guidance

Advantages:

- High-quality textured 3D meshes
- Much faster than DreamFusion
- Better geometry through mesh refinement
- Supports both NeRF and mesh representations

Limitations:

- Multi-view consistency (Janus problem)
- Optimization takes minutes per object
- Quality depends on 2D diffusion prior

## How It Works

Magic3D is a two-stage coarse-to-fine text-to-3D generation framework by NVIDIA.
It uses Score Distillation Sampling (SDS) from a 2D diffusion model to optimize
a 3D representation, first as a coarse NeRF then refined as a textured mesh.

Technical specifications:

- Coarse stage: Instant-NGP NeRF, 64x64 diffusion guidance, ~40 min optimization
- Fine stage: DMTet mesh, latent diffusion guidance, ~20 min optimization
- Diffusion prior: eDiff-I (coarse) + Stable Diffusion (fine)
- SDS guidance scale: 100 (coarse) → 7.5 (fine)
- NeRF resolution: 128^3 hash grid
- Mesh resolution: 512^3 DMTet grid

Reference: Lin et al., "Magic3D: High-Resolution Text-to-3D Content Creation", CVPR 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Magic3DModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Int32,Nullable<Int32>)` | Initializes a new instance of Magic3DModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CoarseModel` | Gets the coarse-stage noise predictor (pixel-space SDS). |
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
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

