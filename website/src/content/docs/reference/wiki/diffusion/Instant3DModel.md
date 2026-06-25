---
title: "Instant3DModel<T>"
description: "Instant3D model -- fast text-to-3D with feed-forward generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ThreeD`

Instant3D model -- fast text-to-3D with feed-forward generation.

## For Beginners

Instant3D creates 3D from text in under 1 second:

Key characteristics:

- Feed-forward: no per-shape optimization needed
- Multi-view generation + instant reconstruction
- Sub-second 3D generation
- NeRF output with mesh extraction

Reference: Li et al., "Instant3D: Fast Text-to-3D with Sparse-View Generation and Large Reconstruction Model", ICLR 2024

## How It Works

Instant3D generates 3D objects from text in a single forward pass using
a multi-view diffusion model with a feed-forward reconstruction network.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Instant3DModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Int32)` | Initializes a new instance of Instant3DModel with full customization support. |

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

