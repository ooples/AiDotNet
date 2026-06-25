---
title: "TripoSRModel<T>"
description: "TripoSR model for ultra-fast feed-forward single-image 3D reconstruction using LRM transformer."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ThreeD`

TripoSR model for ultra-fast feed-forward single-image 3D reconstruction using LRM transformer.

## For Beginners

TripoSR creates 3D meshes from a single photo in half a second.

How TripoSR works:

1. Input image is encoded by DINO-v2 into 768-dim features
2. Transformer predicts triplane features (3 orthogonal 2D feature planes)
3. Any 3D point is queried by projecting onto each plane and interpolating
4. NeRF-like MLP decodes triplane features to density and color
5. Marching cubes extracts mesh from the density field
6. Texture is extracted from the color predictions

Key characteristics:

- ~0.5 second generation on GPU (fastest image-to-3D)
- Large Reconstruction Model (LRM) architecture
- Triplane representation for efficient 3D encoding
- Feed-forward: no iterative optimization or diffusion sampling
- High-quality textured meshes from single images
- Open-source (StabilityAI + Tripo)

When to use TripoSR:

- Real-time 3D reconstruction from images
- Interactive applications requiring instant 3D
- High-throughput 3D asset pipelines
- When speed is the primary concern

Limitations:

- Quality may be lower than optimization-based methods
- Limited to single-object, object-centric scenes
- Triplane resolution limits geometric detail
- Less accurate for thin structures

## How It Works

TripoSR uses a Large Reconstruction Model (LRM) architecture with a transformer backbone
that predicts triplane features from a single image in ~0.5 seconds. The triplane
representation is decoded into a textured 3D mesh via marching cubes.

Architecture components:

- Transformer backbone (1024 hidden, 16 layers, 16 heads) for triplane prediction
- DINO-v2 image encoder for 768-dim conditioning
- Triplane representation (3 orthogonal feature planes)
- NeRF-like volume decoder from triplane features
- Marching cubes mesh extraction
- Feed-forward: single pass, no diffusion iteration

Technical specifications:

- Architecture: LRM (Large Reconstruction Model) with transformer
- Hidden dimension: 1024
- Transformer layers: 16
- Attention heads: 16
- Image encoder: DINO-v2 (768-dim)
- 3D representation: Triplane features
- Mesh extraction: Marching cubes
- Generation time: ~0.5 seconds
- Feed-forward: Yes (1 inference step)
- Default point count: 8,192
- Open-source: Yes (MIT license)

Reference: Tochilkin et al., "TripoSR: Fast 3D Object Reconstruction from a Single Image", 2024

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

