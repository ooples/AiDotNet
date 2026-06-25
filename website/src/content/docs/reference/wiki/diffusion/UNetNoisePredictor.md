---
title: "UNetNoisePredictor<T>"
description: "U-Net architecture for noise prediction in diffusion models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.NoisePredictors`

U-Net architecture for noise prediction in diffusion models.

## For Beginners

Think of U-Net like a funnel:

1. Encoder (going down): Compresses the image, capturing patterns at different scales
2. Middle: Processes the most compressed representation
3. Decoder (going up): Expands back to original size, using skip connections

Skip connections are like "shortcuts" that connect encoder layers directly to
decoder layers, helping the network preserve fine details that might otherwise
be lost during compression.

## How It Works

The U-Net is the most common architecture for diffusion model noise prediction.
It has an encoder-decoder structure with skip connections that help preserve
fine-grained details during the denoising process.

This implementation follows the Stable Diffusion architecture with:

- Residual blocks with group normalization
- Self-attention at lower resolutions
- Cross-attention for text conditioning
- Time embedding injection via adaptive normalization

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UNetNoisePredictor(NeuralNetworkArchitecture<>,Int32,Nullable<Int32>,Int32,Int32[],Int32,Int32[],Int32,Int32,Int32,List<UNetNoisePredictor<>.UNetBlock>,List<UNetNoisePredictor<>.UNetBlock>,List<UNetNoisePredictor<>.UNetBlock>,ILossFunction<>,Nullable<Int32>)` | Initializes a new instance of the UNetNoisePredictor class with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseChannels` |  |
| `ChannelMultipliers` | Gets the per-level channel multipliers of this UNet's encoder/decoder. |
| `ContextDimension` |  |
| `InputChannels` |  |
| `OutputChannels` |  |
| `ParameterCount` |  |
| `SupportsCFG` |  |
| `SupportsCrossAttention` |  |
| `TimeEmbeddingDim` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyCrossAttention(ILayer<>,Tensor<>,Tensor<>)` | Applies cross-attention between the sample and conditioning. |
| `ApplyResBlock(ILayer<>,Tensor<>,Tensor<>)` | Applies a residual block with time embedding conditioning. |
| `Clone` |  |
| `CompileForward(Tensor<>,Int32,Tensor<>)` | Eagerly compiles the UNet forward pass for the given sample input shape, storing the plan in the per-instance cache. |
| `ConcatenateChannels(Tensor<>,Tensor<>)` | Concatenates two tensors along the channel dimension. |
| `CreateDefaultDecoderBlocks` | Creates industry-standard decoder blocks (reverse of encoder). |
| `CreateDefaultEncoderBlocks` | Creates industry-standard encoder blocks based on the Stable Diffusion U-Net. |
| `CreateDefaultMiddleBlocks(Int32)` | Creates industry-standard middle (bottleneck) blocks. |
| `DeepCopy` |  |
| `ForwardUNet(Tensor<>,Tensor<>,Tensor<>)` | Forward pass for inference (no skip storage needed). |
| `ForwardUNetWithSkips(Tensor<>,Tensor<>,Tensor<>)` | Performs the forward pass through the U-Net architecture. |
| `GetParameterChunks` |  |
| `GetParameterGradients` |  |
| `GetParameters` |  |
| `InitializeLayers(NeuralNetworkArchitecture<>,List<UNetNoisePredictor<>.UNetBlock>,List<UNetNoisePredictor<>.UNetBlock>,List<UNetNoisePredictor<>.UNetBlock>)` | Initializes all layers of the U-Net, using custom layers from the user if provided or creating industry-standard layers from the Stable Diffusion paper. |
| `PredictCompiledForward(Tensor<>,Tensor<>,Tensor<>)` | Internal: dispatches to the compiled plan when available + enabled, falling back to eager `Tensor{` otherwise. |
| `PredictNoiseWithEmbedding(Tensor<>,Tensor<>,Tensor<>)` |  |
| `ProjectTimeEmbedding(Tensor<>)` | Projects the sinusoidal timestep embedding through the MLP. |
| `ResolveShapesViaForward` | Single-source-of-truth shape resolution: runs the real forward topology under `Action)` so every lazy layer resolves its TRUE shape exactly as the forward would — including decoder skip concatenation — WITHOUT allocating any weights. |
| `SaveForwardState(List<Tensor<>>,Tensor<>)` | Stores skip connections and time embedding during forward for backward pass. |
| `SetAllLayersTrainingMode(Boolean)` | Sets the training/eval mode on every layer in the UNet (input/output convs, time MLPs, and every encoder/middle/decoder block sublayer). |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `ForwardProfilingSink` | Optional sub-phase profiling hook. |
| `MaxConcatPoolEntries` | Maximum number of distinct concat-shape entries to retain. |
| `_architecture` | The neural network architecture configuration, if provided. |
| `_attentionResolutions` | Resolutions at which to apply attention. |
| `_baseChannels` | Base channel count. |
| `_channelMultipliers` | Channel multipliers for each resolution level. |
| `_contextDim` | Context dimension for cross-attention. |
| `_decoderBlocks` | Decoder blocks (upsampling path). |
| `_encoderBlocks` | Encoder blocks (downsampling path). |
| `_inputChannels` | Number of input channels. |
| `_inputConv` | Input convolution. |
| `_inputHeight` | Latent spatial height for the noise predictor (default: 64). |
| `_lastInput` | Cached input for backward pass. |
| `_lastOutput` | Cached output for backward pass. |
| `_layersInitialized` | Tracks whether the U-Net layer graph has been built. |
| `_lazyShapesResolved` | Runs a single dummy forward through the network at the configured `_inputHeight` spatial size so every lazy layer resolves its weight shapes. |
| `_middleBlocks` | Middle blocks (bottleneck). |
| `_numHeads` | Number of attention heads. |
| `_numResBlocks` | Number of residual blocks per resolution level. |
| `_outputChannels` | Number of output channels. |
| `_outputConv` | Output convolution. |
| `_preserveMaterializedParameters` | True once this instance has runtime weight state that a clone must preserve. |
| `_residentGraph` |  |
| `_timeEmbedMlp1` | Time embedding MLP. |
| `_timeEmbeddingDim` | Time embedding dimension. |

