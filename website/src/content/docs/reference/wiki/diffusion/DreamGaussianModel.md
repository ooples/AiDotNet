---
title: "DreamGaussianModel<T>"
description: "DreamGaussian model for fast 3D Gaussian splatting generation with Score Distillation Sampling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ThreeD`

DreamGaussian model for fast 3D Gaussian splatting generation with Score Distillation Sampling.

## For Beginners

DreamGaussian creates 3D objects using Gaussian splats in ~2 minutes.

How DreamGaussian works:

1. Initialize random 3D Gaussians (position, color, opacity, covariance)
2. Render Gaussians from random viewpoints using differentiable rasterization
3. Compute SDS loss by comparing renders against the diffusion model's guidance
4. Optimize Gaussian parameters via gradient descent (~500 steps)
5. Extract mesh via marching cubes on the Gaussian opacity field
6. Refine UV textures using a second SDS stage on the extracted mesh

Key characteristics:

- ~2 minutes total generation (vs hours for NeRF-based methods)
- Gaussian splatting is much faster to optimize than NeRF
- Two-stage: Gaussian optimization + UV texture refinement
- Supports text-to-3D and image-to-3D
- Produces textured meshes ready for rendering

When to use DreamGaussian:

- Fast 3D content prototyping
- Text-to-3D generation with reasonable quality
- Image-to-3D reconstruction
- When speed matters more than maximum quality

Limitations:

- Lower quality than longer optimization methods (DreamFusion, Magic3D)
- Janus problem (multi-face artifacts) from SDS loss
- Mesh quality depends on Gaussian-to-mesh conversion
- Texture refinement is limited by UV unwrapping quality

## How It Works

DreamGaussian combines 3D Gaussian Splatting with Score Distillation Sampling (SDS)
from a pretrained diffusion model for fast text-to-3D and image-to-3D generation.
A second stage refines UV-space textures on the extracted mesh.

Architecture components:

- SD 1.5 U-Net backbone for SDS guidance (320 base channels, 768-dim CLIP)
- 3D Gaussian Splatting as 3D representation
- SDS loss from pretrained 2D diffusion model for 3D optimization
- UV-space texture refinement stage for mesh appearance
- Differentiable rasterization for Gaussian rendering
- Marching cubes mesh extraction from opacity field

Technical specifications:

- 3D representation: 3D Gaussian Splatting (~10,000 Gaussians)
- Guidance backbone: SD 1.5 U-Net (320 base channels, 768-dim CLIP)
- Optimization: ~500 SDS steps for Gaussians + UV refinement
- Total time: ~2 minutes on a single GPU
- Mesh extraction: Marching cubes from opacity field
- Texture: UV-space refinement with SDS
- Noise schedule: Scaled linear beta [0.00085, 0.012], 1000 timesteps
- Scheduler: DDIM

Reference: Tang et al., "DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation", ICLR 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DreamGaussianModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Int32,Nullable<Int32>)` | Initializes a new instance of DreamGaussianModel with full customization support. |

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
| `InitializeLayers(UNetNoisePredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the U-Net and VAE layers using custom or default configurations. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `BASE_CHANNELS` | Base channel count for the SD 1.5 U-Net backbone (320). |
| `CROSS_ATTENTION_DIM` | Cross-attention dimension from CLIP text encoder (768). |
| `DEFAULT_POINT_COUNT` | Default number of 3D Gaussian points (10,000). |
| `LATENT_CHANNELS` | Number of latent channels (4, standard SD VAE). |
| `_conditioner` | The CLIP text encoder conditioning module. |
| `_unet` | The SD 1.5 U-Net noise predictor for SDS guidance. |
| `_vae` | The standard SD VAE for encoding/decoding rendered views. |

