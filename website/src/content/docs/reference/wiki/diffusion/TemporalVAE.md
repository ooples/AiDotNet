---
title: "TemporalVAE<T>"
description: "Temporal-aware Variational Autoencoder for video diffusion models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.VAE`

Temporal-aware Variational Autoencoder for video diffusion models.

## For Beginners

While a standard VAE processes each frame independently,
TemporalVAE considers relationships between consecutive frames:

Standard VAE approach (per-frame):

- Frame 1 -> Latent 1 (no knowledge of other frames)
- Frame 2 -> Latent 2 (no knowledge of other frames)
- Result: Possible flickering/inconsistency between frames

TemporalVAE approach:

- Frames 1,2,3,... -> Encode with temporal awareness
- Latent knows about neighboring frames
- Result: Smoother, more consistent video

Key features:

- 3D convolutions that span across time dimension
- Temporal attention for long-range frame relationships
- Optional causal mode for streaming/autoregressive generation

Used in: Stable Video Diffusion, Video LDM, and similar models.

## How It Works

The TemporalVAE extends the standard VAE to handle video data by incorporating
temporal awareness into the encoding and decoding process. This helps maintain
temporal consistency across frames when used in video diffusion models.

Architecture details:

- Input: [batch, channels, frames, height, width] video tensor
- Encoder: 2D spatial blocks + 1D temporal blocks
- Latent: [batch, latentChannels, frames, height/8, width/8]
- Decoder: 2D spatial blocks + 1D temporal blocks
- Output: [batch, channels, frames, height, width] reconstructed video

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TemporalVAE(Int32,Int32,Int32,Int32[],Int32,Int32,Boolean,Nullable<Double>,ILossFunction<>,Nullable<Int32>)` | Initializes a new instance of the TemporalVAE class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DownsampleFactor` |  |
| `InputChannels` |  |
| `IsCausal` | Gets whether this VAE uses causal convolutions. |
| `LatentChannels` |  |
| `LatentScaleFactor` |  |
| `ParameterCount` |  |
| `SupportsSlicing` |  |
| `SupportsTiling` |  |
| `TemporalKernelSize` | Gets the temporal kernel size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyTemporalLayers(List<ILayer<>>,List<Tensor<>>)` | Applies temporal layers across frames. |
| `BackpropagateLossGradient(Tensor<>)` |  |
| `Clone` |  |
| `Decode(Tensor<>)` |  |
| `DecodeFrame(Tensor<>)` | Decodes a single frame latent. |
| `DecodeVideo(Tensor<>)` | Decodes a video latent with temporal awareness. |
| `DecodeVideoFromDiffusion(Tensor<>)` | Decodes a diffusion video latent back to video space. |
| `DeepCopy` |  |
| `Encode(Tensor<>,Boolean)` |  |
| `EncodeFrame(Tensor<>)` | Encodes a single frame. |
| `EncodeVideo(Tensor<>)` | Encodes a video with temporal awareness. |
| `EncodeVideoForDiffusion(Tensor<>,Boolean)` | Encodes a video and applies latent scaling for use in diffusion. |
| `EncodeWithDistribution(Tensor<>)` |  |
| `ExtractFrame(Tensor<>,Int32)` | Extracts a single frame from a video tensor. |
| `GetParameterChunks` |  |
| `GetParameters` |  |
| `InitializeLayers` | Initializes all encoder and decoder layers. |
| `SetParameters(Vector<>)` |  |
| `StackFrames(List<Tensor<>>)` | Stacks frames into a video tensor. |
| `StackFramesToVideo(List<Tensor<>>)` | Stacks frame features into a video tensor. |
| `TriggerLazyShapeResolution` | Materializes every lazy encoder/decoder weight tensor by running one tiny encode+decode probe, so `GetParameters`/`Vector{`/parameter-count agree before any real forward. |
| `UnstackVideoToFrames(Tensor<>,Int32)` | Unstacks a video tensor back to individual frames. |

## Fields

| Field | Summary |
|:-----|:--------|
| `SVD_LATENT_SCALE` | Standard Stable Video Diffusion latent scale factor. |
| `_baseChannels` | Base channel count. |
| `_cachedLogVar` | Cached log variance from encoding. |
| `_cachedMean` | Cached mean from encoding. |
| `_causalMode` | Whether to use causal convolutions (for streaming). |
| `_channelMultipliers` | Channel multipliers for each level. |
| `_decoderSpatialLayers` | Decoder spatial layers. |
| `_decoderTemporalLayers` | Decoder temporal layers. |
| `_downsampleFactor` | Downsampling factor. |
| `_encoderSpatialLayers` | Encoder spatial layers. |
| `_encoderTemporalLayers` | Encoder temporal layers. |
| `_inputChannels` | Input channels (3 for RGB video). |
| `_inputConv` | Input convolution. |
| `_latentChannels` | Latent channels. |
| `_latentScaleFactor` | Latent scale factor. |
| `_logVarConv` | Log variance projection layer. |
| `_meanConv` | Mean projection layer. |
| `_numTemporalLayers` | Number of temporal layers in encoder/decoder. |
| `_outputConv` | Output convolution. |
| `_postQuantConv` | Post-quant convolution. |
| `_preserveMaterializedParameters` | True once this VAE has runtime state that a clone must preserve. |
| `_temporalKernelSize` | Number of frames to process together. |

