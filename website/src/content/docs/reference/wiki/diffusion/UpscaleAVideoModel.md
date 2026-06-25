---
title: "UpscaleAVideoModel<T>"
description: "Upscale-A-Video model for temporally consistent video super-resolution with diffusion."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.SuperResolution`

Upscale-A-Video model for temporally consistent video super-resolution with diffusion.

## For Beginners

Upscale-A-Video increases video resolution by 4x without flickering.

How Upscale-A-Video works:

1. Each frame is encoded with a temporal VAE that considers neighboring frames
2. The low-resolution video is concatenated with latent noise (8 input channels)
3. Temporal attention ensures frames are consistent with each other
4. Flow-guided propagation maintains consistency across long sequences
5. The temporal VAE decodes the result to high-resolution flicker-free video

Key characteristics:

- 4x video upscaling with temporal consistency
- Temporal attention layers prevent inter-frame flickering
- Flow-guided recurrent propagation for long-range coherence
- Built on SD architecture with temporal extensions
- Processes 16 frames at a time at 24 FPS by default

When to use Upscale-A-Video:

- Upscaling low-resolution video recordings
- Enhancing video quality for streaming/display
- Restoring old or compressed video footage
- Improving AI-generated video resolution

Limitations:

- Fixed 4x upscale factor
- High VRAM requirements due to temporal processing
- Processing speed limited by number of frames per batch
- May introduce subtle artifacts at scene transitions

## How It Works

Upscale-A-Video extends image super-resolution to video with temporal consistency,
using temporal attention layers and flow-guided recurrent propagation to achieve
flicker-free 4x upscaling of video content.

Architecture components:

- Video U-Net with temporal attention and temporal convolutions
- 8 input channels (4 latent + 4 downscaled low-res conditioning)
- 2 temporal attention layers per block for inter-frame consistency
- Temporal VAE for temporally coherent encoding/decoding
- Flow-guided recurrent propagation for long-range temporal consistency
- DDIM scheduler for efficient inference

Technical specifications:

- Architecture: Video U-Net with temporal attention + temporal VAE
- Input channels: 8 (4 latent noise + 4 downscaled low-res)
- Output channels: 4 (latent space)
- Base channels: 320, multipliers [1, 2, 4, 4]
- Cross-attention dimension: 1024
- Temporal attention layers: 2 per block
- Temporal VAE: 3-frame kernel, 1 temporal layer
- Noise schedule: Scaled linear beta [0.00085, 0.012], 1000 timesteps
- Default frames: 16 at 24 FPS
- Upscale factor: 4x
- Scheduler: DDIM for efficient video inference

Reference: Zhou et al., "Upscale-A-Video: Temporal-Consistent Diffusion Model for Real-World Video Super-Resolution", CVPR 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UpscaleAVideoModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,VideoUNetPredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of UpscaleAVideoModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `SupportsImageToVideo` |  |
| `SupportsTextToVideo` |  |
| `SupportsVideoToVideo` |  |
| `TemporalVAE` |  |
| `UpscaleFactor` | Gets the video upscale factor (4x). |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `InitializeLayers(VideoUNetPredictor<>,TemporalVAE<>,Nullable<Int32>)` | Initializes the Video U-Net and Temporal VAE using custom or default configurations. |
| `PredictVideoNoise(Tensor<>,Int32,Tensor<>,Tensor<>)` |  |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `BASE_CHANNELS` | Base channel count for the Video U-Net (320). |
| `CROSS_ATTENTION_DIM` | Cross-attention dimension (1024). |
| `DEFAULT_FPS` | Default frames per second (24). |
| `DEFAULT_NUM_FRAMES` | Default number of frames per batch (16). |
| `DefaultHeight` | Default output video height (576, 4x upscaled from 144). |
| `DefaultWidth` | Default output video width (1024, 4x upscaled from 256). |
| `INPUT_CHANNELS` | Input channels for the Video U-Net input convolution (4 = latent channels). |
| `LATENT_CHANNELS` | Number of latent channels (4). |
| `NUM_HEADS` | Number of attention heads in the Video U-Net (8). |
| `NUM_TEMPORAL_LAYERS` | Number of temporal attention layers per block (2). |
| `UPSCALE_FACTOR` | Upscale factor (4x). |
| `_conditioner` | Optional conditioning module for guided video super-resolution. |
| `_temporalVAE` | The temporal VAE for temporally coherent video encoding/decoding. |
| `_videoUNet` | The Video U-Net noise predictor with temporal attention layers. |

