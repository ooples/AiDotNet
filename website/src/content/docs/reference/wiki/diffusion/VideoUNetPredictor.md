---
title: "VideoUNetPredictor<T>"
description: "3D U-Net architecture for video noise prediction in diffusion models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.NoisePredictors`

3D U-Net architecture for video noise prediction in diffusion models.

## For Beginners

While a regular U-Net processes single images,
VideoUNet processes sequences of frames as a 3D volume:

Regular U-Net:

- Input: [batch, channels, height, width]
- 2D convolutions across spatial dimensions only
- Each image processed independently

Video U-Net:

- Input: [batch, channels, frames, height, width]
- 3D convolutions across space AND time
- Frames are processed together, understanding motion

Key features:

- Temporal convolutions capture motion patterns
- Temporal attention for long-range frame relationships
- Skip connections across both space and time
- Image conditioning for image-to-video generation

Used in: Stable Video Diffusion, ModelScope, VideoCrafter

## How It Works

The VideoUNetPredictor extends the standard U-Net architecture to handle
video data by incorporating 3D convolutions and temporal attention.
This is the core noise prediction network used in video diffusion models
like Stable Video Diffusion.

Architecture details:

- Encoder: 3D ResBlocks with temporal + spatial attention
- Middle: Multiple 3D attention blocks
- Decoder: 3D ResBlocks with skip connections
- Temporal convolutions with kernel size 3 across frames

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VideoUNetPredictor(Int32,Nullable<Int32>,Int32,Int32[],Int32,Int32[],Int32,Int32,Int32,Boolean,Int32,Int32,Int32,Int32,ILossFunction<>,Nullable<Int32>)` | Initializes a new instance of the VideoUNetPredictor class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseChannels` |  |
| `ContextDimension` |  |
| `InputChannels` |  |
| `NumTemporalLayers` | Gets the number of temporal transformer layers. |
| `OutputChannels` |  |
| `ParameterCount` |  |
| `SupportsCFG` |  |
| `SupportsCrossAttention` |  |
| `SupportsImageConditioning` | Gets whether this predictor supports image conditioning for image-to-video. |
| `TimeEmbeddingDim` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddImageCondition(Tensor<>,Tensor<>,Int32)` | Adds image condition to video features. |
| `ApplyDownsample(ILayer<>,Tensor<>,Boolean)` | Applies downsampling to video. |
| `ApplyFiLMConditioning(DenseLayer<>,Tensor<>,Tensor<>,Boolean)` | Applies Feature-wise Linear Modulation (FiLM) from the timestep embedding to the feature map `x`. |
| `ApplyTemporalAttention(ILayer<>,Tensor<>)` | Applies temporal attention across frames using GPU/CPU accelerated tensor operations. |
| `ApplyTemporalProcessing(ILayer<>,Tensor<>)` | Applies temporal processing with a residual connection, per Ho et al. |
| `ApplyUpsample(ILayer<>,Tensor<>,Boolean)` | Applies upsampling to video. |
| `ApplyVideoBlock(VideoUNetPredictor<>.VideoBlock,Tensor<>,Tensor<>,Tensor<>,Boolean)` | Applies a single video block: spatial ResBlock → FiLM timestep conditioning → temporal ResBlock → spatial attention → temporal attention → cross-attention. |
| `Clone` |  |
| `ConcatenateChannels(Tensor<>,Tensor<>,Boolean)` | Concatenates channels for skip connections. |
| `CreateTemporalMixingBlock` | Creates a temporal mixing block that learns a frame-axis transform. |
| `CreateTimeCondProjection(Int32)` | Creates a FiLM conditioning projection for a VideoBlock: timeEmbedDim → channels * 2. |
| `DeepCopy` |  |
| `EnumerateLayersInParameterOrder` | Enumerates every layer in the EXACT order used by `GetParameters` / `Vector{` (input conv, time-embed MLPs, image-cond projection, then each encoder/middle/decoder block's components, then the output conv). |
| `ExtractFrame(Tensor<>,Int32)` | Extracts a single frame from video. |
| `ForwardVideoUNet(Tensor<>,Tensor<>,Tensor<>,Tensor<>)` | Performs the forward pass through the Video U-Net. |
| `GetParameters` |  |
| `InitializeLayers` | Initializes all layers of the Video U-Net. |
| `PredictNoise(Tensor<>,Int32,Tensor<>)` |  |
| `PredictNoiseWithEmbedding(Tensor<>,Tensor<>,Tensor<>)` |  |
| `PredictNoiseWithImageCondition(Tensor<>,Int32,Tensor<>,Tensor<>)` | Predicts noise for image-to-video generation with image conditioning. |
| `ProcessVideoFrames(Tensor<>,Func<Tensor<>,Tensor<>>)` | Processes each frame of a video through a layer. |
| `ProjectTimeEmbedding(Tensor<>)` | Projects the sinusoidal timestep embedding through the MLP. |
| `ResolutionAtLevel(Int32)` | Returns the spatial resolution (height = width) at encoder/decoder level `level`. |
| `SetParameters(Vector<>)` |  |
| `StackFrames(List<Tensor<>>)` | Stacks frames into video tensor. |
| `TriggerLazyShapeResolution` | Runs a single dummy forward through the network at the configured spatial / frame size so every lazy layer (time-embedding MLPs, temporal + cross attention, and the image-condition projection) resolves its weight shapes. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_attentionResolutions` | Resolutions at which to apply attention. |
| `_baseChannels` | Base channel count. |
| `_channelMultipliers` | Channel multipliers for each resolution level. |
| `_clipTokenLength` | CLIP text token sequence length for cross-attention. |
| `_contextDim` | Context dimension for cross-attention. |
| `_decoderBlocks` | Decoder blocks. |
| `_encoderBlocks` | Encoder blocks. |
| `_imageCondProjection` | Image conditioning projection (for image-to-video). |
| `_inputChannels` | Number of input channels. |
| `_inputConv` | Input convolution. |
| `_inputHeight` | Latent spatial height. |
| `_inputWidth` | Latent spatial width. |
| `_lastInput` | Cached input for backward pass. |
| `_middleBlocks` | Middle blocks. |
| `_numFrames` | Typical number of video frames for temporal attention. |
| `_numHeads` | Number of attention heads. |
| `_numResBlocks` | Number of residual blocks per resolution level. |
| `_numTemporalLayers` | Number of temporal transformer layers. |
| `_outputChannels` | Number of output channels. |
| `_outputConv` | Output convolution. |
| `_supportsImageConditioning` | Whether to support image conditioning. |
| `_timeEmbedMlp1` | Time embedding MLP. |
| `_timeEmbeddingDim` | Time embedding dimension. |

