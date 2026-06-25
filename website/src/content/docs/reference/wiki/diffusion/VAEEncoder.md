---
title: "VAEEncoder<T>"
description: "Convolutional encoder for VAE that compresses images to latent space."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.VAE`

Convolutional encoder for VAE that compresses images to latent space.

## For Beginners

The VAE encoder is like an intelligent image compressor.

What it does step by step:

1. Takes a high-resolution image (e.g., 512x512x3 RGB)
2. Initial conv: Expands channels (3 -> 128) at full resolution
3. DownBlocks: Progressively halves resolution while increasing channels
- Block 1: 128 channels, 512x512 -> 256x256
- Block 2: 256 channels, 256x256 -> 128x128
- Block 3: 512 channels, 128x128 -> 64x64
- Block 4: 512 channels, 64x64 -> 64x64 (no downsample at end)
4. Middle: Extra processing at the bottleneck
5. Output: Produces mean and log-variance for 4-channel latent

The result is a 64x64x4 latent that captures the image's essence
in a compressed form suitable for diffusion.

## How It Works

This implements the encoder portion of a VAE following the Stable Diffusion architecture:

- Input convolution to initial feature channels
- Multiple DownBlocks with ResBlocks and strided conv downsampling
- Middle blocks with attention at the bottleneck
- Final convolutions to produce mean and log variance for the latent distribution

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VAEEncoder(Int32,Int32,Int32,Int32[],Int32,Int32,Int32)` | Initializes a new instance of the VAEEncoder class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DownsampleFactor` | Gets the downsampling factor (spatial reduction from input to output). |
| `InputChannels` | Gets the number of input channels. |
| `LatentChannels` | Gets the number of latent channels. |
| `NamedParameterCount` |  |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildParameterRegistry` | Builds the parameter registry for named weight access. |
| `BuildParameterRegistryPublic` | Builds and returns the parameter registry for use inside the assembly. |
| `Deserialize(BinaryReader)` | Loads the encoder's state from a binary reader. |
| `EncodeAndSample(Tensor<>,Nullable<Int32>)` | Encodes an image and samples from the latent distribution. |
| `EncodeWithDistribution(Tensor<>)` | Encodes and returns mean and log variance separately. |
| `Forward(Tensor<>)` | Encodes an image to latent space, returning concatenated mean and log variance. |
| `ForwardAsync(Tensor<>,CancellationToken)` | Async overload of `Tensor{`. |
| `ForwardEager(Tensor<>)` | Eager forward pass (the body of the original Forward). |
| `GetParameterNames` |  |
| `GetParameterRegistry` | Gets or creates the parameter registry. |
| `GetParameterShape(String)` |  |
| `GetParameters` | Gets all trainable parameters as a single vector. |
| `InvalidateCompiledPlans` | Bumps the structure-version counter so the next call to `Tensor{` drops any plan compiled against an older graph topology. |
| `LoadWeights(Dictionary<String,Tensor<>>,Func<String,String>,Boolean)` |  |
| `RegisterWeightLoadable(ParameterRegistry<>,String,IWeightLoadable<>)` | Registers all parameters from an IWeightLoadable into the registry with a prefix. |
| `ResetState` | Resets the internal state of the encoder. |
| `Sample(Tensor<>,Tensor<>,Nullable<Int32>)` | Samples from the latent distribution using the reparameterization trick. |
| `SampleNoise(Int32[],Random)` | Samples random noise from a standard normal distribution. |
| `Serialize(BinaryWriter)` | Saves the encoder's state to a binary writer. |
| `SetParameter(String,Tensor<>)` |  |
| `SetParameters(Vector<>)` | Sets all trainable parameters from a single vector. |
| `TryGetParameter(String,Tensor<>)` |  |
| `UpdateParameters()` | Updates all learnable parameters using gradient descent. |
| `ValidateWeights(IEnumerable<String>,Func<String,String>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_baseChannels` | Base channel count. |
| `_bottleneckSize` | Spatial size at encoder output (bottleneck). |
| `_channelMults` | Channel multipliers for each level. |
| `_compileHost` | Per-instance compile host. |
| `_downBlocks` | Downsampling blocks. |
| `_inputChannels` | Number of input image channels. |
| `_inputConv` | Input convolution from image channels to base channels. |
| `_lastInput` | Cached intermediate values for backward pass. |
| `_latentChannels` | Number of latent channels. |
| `_logVarConv` | Convolution to project to log variance. |
| `_meanConv` | Convolution to project to mean. |
| `_midBlocks` | Middle residual blocks at the bottleneck. |
| `_normOut` | Group normalization before output projections. |
| `_numGroups` | Number of groups for GroupNorm. |
| `_parameterRegistry` | Parameter registry for named weight access. |
| `_quantConv` | Quant convolution for latent processing. |
| `_silu` | SiLU activation function. |

