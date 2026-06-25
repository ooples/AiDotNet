---
title: "LGMModel<T>"
description: "LGM (Large Gaussian Model) for feed-forward 3D Gaussian generation from multi-view images."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ThreeD`

LGM (Large Gaussian Model) for feed-forward 3D Gaussian generation from multi-view images.

## For Beginners

LGM creates 3D Gaussian splats instantly from multi-view images.

How LGM works:

1. Generate 4 views of the object using a multi-view diffusion model
2. Encode views with DINO-v2 into 1024-dim features
3. Concatenate 4 view latents (4x4 = 16 input channels)
4. Asymmetric U-Net predicts 14 Gaussian parameters per pixel
5. Unproject pixel predictions to 3D Gaussian positions
6. Result: 50,000+ 3D Gaussians in ~5 seconds

Key characteristics:

- Feed-forward: single pass, no optimization loop
- ~5 seconds on GPU (vs minutes for SDS-based methods)
- 50,000 Gaussians for high-quality reconstruction
- Asymmetric U-Net: large decoder for detail, compact encoder
- Real-time Gaussian splatting rendering

When to use LGM:

- Real-time 3D content generation
- Interactive 3D from images
- High-throughput 3D asset creation
- When speed is more important than maximum quality

Limitations:

- Requires multi-view input (needs separate multi-view generation)
- Quality depends on multi-view consistency
- Fixed number of Gaussians per prediction
- Less detail than optimization-based methods

## How It Works

LGM uses a large asymmetric U-Net backbone to predict 3D Gaussians from multi-view images
in a single forward pass, enabling real-time 3D generation without per-shape optimization.

Architecture components:

- Asymmetric U-Net for Gaussian parameter prediction (64 base channels, [1,2,4,8])
- 14 output channels per pixel (position 3 + color 3 + opacity 1 + covariance 7)
- 16 input channels (4 latent channels x 4 views)
- DINO-v2 image encoder for 1024-dim conditioning
- Standard SD VAE for multi-view image encoding
- Feed-forward: no iterative optimization required

Technical specifications:

- Architecture: Asymmetric U-Net for Gaussian prediction
- Input channels: 16 (4 views x 4 latent channels)
- Output channels: 14 (position + color + opacity + covariance)
- Base channels: 64, multipliers [1, 2, 4, 8]
- Attention resolutions: [8, 4]
- Image encoder: DINO-v2 (1024-dim)
- Default Gaussians: 50,000
- Generation time: ~5 seconds
- Feed-forward: Yes (no iterative optimization)

Reference: Tang et al., "LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation", ECCV 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LGMModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Int32,Nullable<Int32>)` | Initializes a new instance of LGMModel with full customization support. |

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

## Fields

| Field | Summary |
|:-----|:--------|
| `BASE_CHANNELS` | Base channel count for the asymmetric U-Net (64). |
| `CROSS_ATTENTION_DIM` | Cross-attention dimension from DINO-v2 (1024). |
| `DEFAULT_POINT_COUNT` | Default number of 3D Gaussian points (50,000). |
| `INPUT_CHANNELS` | Input channels (4 views x 4 latent channels = 16). |
| `LATENT_CHANNELS` | Number of latent channels (4). |
| `OUTPUT_CHANNELS` | Output channels per pixel (position 3 + color 3 + opacity 1 + covariance 7 = 14). |
| `_conditioner` | The DINO-v2 image encoder conditioning module. |
| `_unet` | The asymmetric U-Net for Gaussian parameter prediction. |
| `_vae` | The standard SD VAE for multi-view image encoding. |

