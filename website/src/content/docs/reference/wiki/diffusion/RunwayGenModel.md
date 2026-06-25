---
title: "RunwayGenModel<T>"
description: "Runway Gen model for multi-modal video generation with structure and content disentanglement."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

Runway Gen model for multi-modal video generation with structure and content disentanglement.

## For Beginners

Runway Gen creates professional-quality videos from text and images.

How Runway Gen works:

1. Text/image input is encoded by CLIP (Gen-2) or dual encoder (Gen-3)
2. Video is compressed by the temporal VAE into 4-channel latent space
3. The Video U-Net denoises with temporal attention for frame consistency
4. Multi-modal conditioning guides structure and content separately
5. The temporal VAE decodes the latent back to video frames

Key characteristics:

- Gen-2: CLIP-based, 320-channel U-Net, 25 frames at 24 FPS
- Gen-3: Enhanced dual encoder, 384-channel U-Net, 150 frames, causal VAE
- Multi-modal: text, image, video, and motion conditioning
- Structure-content disentanglement for precise editing
- Cascaded generation for high resolution output

When to use Runway Gen:

- Professional video generation and editing
- Multi-modal conditioning (text + image + video)
- Video-to-video style transfer
- Image animation with motion control

Limitations:

- Commercial/proprietary model (API-only access)
- Expensive generation costs
- Limited control over internal architecture
- Gen-3 requires significant compute resources

## How It Works

Runway Gen (Gen-1/Gen-2/Gen-3) uses temporal diffusion with multi-modal conditioning
for photorealistic video generation and editing. The model supports text, image, video,
and motion conditioning with structure-content disentanglement.

Architecture components:

- Video U-Net with temporal attention and cross-frame consistency
- Gen-2: 320 base channels, 1024-dim CLIP, 8 heads, 1 temporal layer
- Gen-3: 384 base channels, 2048-dim dual encoder, 16 heads, 3 temporal layers
- Temporal VAE for inter-frame coherence (causal in Gen-3)
- Multi-modal conditioning: text, image, video, and motion
- Structure and style disentanglement for editing control

Technical specifications:

- Architecture: Video U-Net with temporal attention and multi-modal conditioning
- Gen-2: 320 base channels, 1024 cross-attention, 8 heads, 1 temporal layer
- Gen-3: 384 base channels, 2048 cross-attention, 16 heads, 3 temporal layers
- Latent channels: 4 (temporal VAE)
- Channel multipliers: [1, 2, 4, 4]
- Gen-2 default: 25 frames at 24 FPS (~1 second)
- Gen-3 default: 150 frames at 24 FPS (~6.25 seconds)
- Scheduler: DDIM for efficient inference
- Supports: text-to-video, image-to-video, video-to-video

Reference: Esser et al., "Structure and Content-Guided Video Synthesis with Diffusion Models", ICCV 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RunwayGenModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,VideoUNetPredictor<>,TemporalVAE<>,IConditioningModule<>,Boolean,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of RunwayGenModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `IsGen3` | Gets whether this is a Gen-3 variant with enhanced architecture. |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `SupportsImageToVideo` |  |
| `SupportsTextToVideo` |  |
| `SupportsVideoToVideo` |  |
| `TemporalVAE` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `CreateGen3Alpha(IConditioningModule<>)` | Creates a Gen-3 Alpha variant with enhanced architecture. |
| `DeepCopy` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `InitializeLayers(VideoUNetPredictor<>,TemporalVAE<>,Nullable<Int32>)` | Initializes the Video U-Net and temporal VAE using custom or variant-appropriate defaults. |
| `PredictVideoNoise(Tensor<>,Int32,Tensor<>,Tensor<>)` |  |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DEFAULT_FPS` | Default frames per second (24). |
| `DEFAULT_NUM_FRAMES` | Default number of frames for Gen-2 (25). |
| `GEN2_BASE_CHANNELS` | Base channel count for Gen-2 (320). |
| `GEN2_CROSS_ATTENTION_DIM` | Cross-attention dimension for Gen-2 (1024, CLIP). |
| `GEN3_BASE_CHANNELS` | Base channel count for Gen-3 (384). |
| `GEN3_CROSS_ATTENTION_DIM` | Cross-attention dimension for Gen-3 (2048, dual encoder). |
| `LATENT_CHANNELS` | Number of latent channels (4). |
| `_conditioner` | Optional conditioning module for multi-modal guided generation. |
| `_isGen3` | Whether this is a Gen-3 variant. |
| `_temporalVAE` | The temporal VAE for inter-frame coherent encoding/decoding. |
| `_videoUNet` | The Video U-Net noise predictor with temporal attention. |

