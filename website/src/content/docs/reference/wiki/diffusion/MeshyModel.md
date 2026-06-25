---
title: "MeshyModel<T>"
description: "Meshy model for production-grade text/image to 3D generation with PBR texturing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ThreeD`

Meshy model for production-grade text/image to 3D generation with PBR texturing.

## For Beginners

Meshy creates game-ready 3D models with full material textures.

How Meshy works:

1. Text/image input generates multiple consistent views
2. Views are reconstructed into a 3D mesh
3. Topology is optimized for clean triangle connectivity
4. PBR texturing stage creates material maps (albedo, normal, roughness, metallic)
5. Output is a game-engine-ready asset (Unity, Unreal compatible)

Key characteristics:

- Full PBR material pipeline (4 texture maps)
- Topology optimization for clean meshes
- Game-engine ready output (Unity, Unreal)
- Text-to-3D and image-to-3D support
- Production-quality assets

When to use Meshy:

- Game asset creation
- Production 3D content
- PBR material generation
- Rapid prototyping for game/film

Limitations:

- Commercial API service
- Limited control over mesh topology
- PBR quality depends on view consistency
- Not suitable for organic/deformable models

## How It Works

Meshy combines multi-view generation with a dedicated PBR (Physically Based Rendering)
texturing stage to produce game-engine-ready 3D assets with full material maps.

Architecture components:

- Multi-view generation U-Net (320 base channels, [1,2,4], 1024-dim)
- Dedicated PBR texturing stage for albedo, normal, roughness, metallic
- Topology optimization for clean, game-ready meshes
- Standard SD VAE for view encoding/decoding
- DDIM scheduler for efficient inference

Technical specifications:

- Architecture: Multi-view U-Net + PBR texturing pipeline
- Base channels: 320, multipliers [1, 2, 4]
- Cross-attention: 1024-dim
- PBR outputs: albedo, normal, roughness, metallic maps
- Mesh topology: optimized triangle mesh
- Scheduler: DDIM
- Default vertex count: 8,192

Reference: Meshy AI, "Meshy: AI 3D Model Generator", 2024

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

