---
title: "StableAudioModel<T>"
description: "Stable Audio model for generating high-quality audio from text descriptions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.StableAudio`

Stable Audio model for generating high-quality audio from text descriptions.

## For Beginners

Stable Audio creates professional-quality audio:

How it works:

1. You describe the audio you want ("upbeat electronic track")
2. T5 encodes your text into embeddings
3. Duration and timing are encoded as conditioning
4. DiT diffusion generates latent audio representations
5. VAE decoder converts latents to 44.1kHz stereo audio

Key features:

- CD-quality 44.1kHz stereo output
- Variable-length generation (up to 3 minutes)
- Music and sound effects generation
- Timing-aware conditioning

Usage:

## How It Works

Stable Audio is Stability AI's state-of-the-art audio generation model that uses
latent diffusion with a Diffusion Transformer (DiT) architecture for high-quality
music and sound effects generation.

Architecture components:

- **T5 Text Encoder:** Encodes text prompts into conditioning embeddings
- **VAE:** Compresses audio to/from latent space (44.1kHz to 21.5Hz latent)
- **DiT (Diffusion Transformer):** Predicts noise using transformer blocks
- **Timing Conditioning:** Encodes duration and timing information

Reference: "Stable Audio: Fast Timing-Conditioned Latent Audio Diffusion" by Evans et al., 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StableAudioModel(NeuralNetworkArchitecture<>,StableAudioOptions,ITokenizer,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a Stable Audio model using native layers for training from scratch. |
| `StableAudioModel(NeuralNetworkArchitecture<>,String,String,String,ITokenizer,StableAudioOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a Stable Audio model using pretrained ONNX models for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxDurationSeconds` | Gets the maximum duration of audio that can be generated. |
| `SampleRate` | Gets the sample rate of generated audio. |
| `SupportsAudioContinuation` | Gets whether this model supports audio continuation. |
| `SupportsAudioInpainting` | Gets whether this model supports audio inpainting. |
| `SupportsTextToAudio` | Gets whether this model supports text-to-audio generation. |
| `SupportsTextToMusic` | Gets whether this model supports text-to-music generation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddTimestepEmbedding(Tensor<>,Tensor<>)` | Adds timestep embedding to latents. |
| `AddTimingConditioning(Tensor<>,Tensor<>)` | Adds timing conditioning to latents. |
| `ApplyCrossAttentionConditioning(Tensor<>,Tensor<>)` | Applies cross-attention conditioning to latents. |
| `ApplyGuidance(Tensor<>,Tensor<>,Double)` | Applies classifier-free guidance. |
| `CombineConditionings(Tensor<>,Tensor<>)` | Combines audio and text conditionings. |
| `ContinueAudio(Tensor<>,String,Double,Int32,Nullable<Int32>)` | Continues existing audio by extending it. |
| `CreateNewInstance` | Creates a new instance for cloning. |
| `CreateTimestepTensor(Double)` | Creates a timestep tensor with sinusoidal embedding. |
| `CreateTimingConditioning(Double)` | Creates timing conditioning for the given duration. |
| `CreateUnconditionalEmbedding` | Creates unconditional embedding for classifier-free guidance. |
| `DecodeLatentsToAudio(Tensor<>)` | Decodes latents to audio waveform. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data. |
| `Dispose(Boolean)` | Disposes of model resources. |
| `EncodeAudioToLatents(Tensor<>)` | Encodes audio to latent space. |
| `EncodeTextEmbedding(Tensor<>)` | Encodes text embedding from tokens. |
| `EncodeTextToTensor(String)` | Encodes text prompt to token tensor. |
| `EulerStep(Tensor<>,Tensor<>,Double,Int32)` | Performs an Euler denoising step. |
| `ForwardThroughDiT(Tensor<>)` | Forwards input through DiT layers. |
| `GenerateAudio(String,String,Double,Int32,Double,Nullable<Int32>)` | Generates audio from a text description. |
| `GenerateAudioAsync(String,String,Double,Int32,Double,Nullable<Int32>,CancellationToken)` | Generates audio asynchronously. |
| `GenerateMusic(String,String,Double,Int32,Double,Nullable<Int32>)` | Generates music from a text description. |
| `GenerateTimesteps(Int32)` | Generates diffusion timesteps using Euler schedule. |
| `GetDefaultOptions` | Gets default generation options. |
| `GetModelDimensions(StableAudioModelSize)` | Gets model dimensions based on the selected size. |
| `GetModelMetadata` | Gets model metadata. |
| `GetOptions` |  |
| `InitializeLatents(Int32,Random)` | Initializes latent noise for diffusion. |
| `InitializeLayers` | Initializes the neural network layers following the golden standard pattern. |
| `InpaintAudio(Tensor<>,Tensor<>,String,Int32,Nullable<Int32>)` | Fills in missing or masked sections of audio. |
| `PostprocessOutput(Tensor<>)` | Postprocesses model output. |
| `PredictCore(Tensor<>)` | Makes a prediction using the model. |
| `PrepareDiTInput(Tensor<>,Tensor<>,Tensor<>,Tensor<>)` | Prepares DiT input by combining latents, text conditioning, timing, and timestep. |
| `PreprocessAudio(Tensor<>)` | Preprocesses raw audio for model input. |
| `RunDiffusionLoop(Tensor<>,Tensor<>,Tensor<>,Tensor<>,Int32,Double,Random)` | Runs the diffusion denoising loop with DiT. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data. |
| `Train(Tensor<>,Tensor<>)` | Trains the model on input data. |
| `UpdateParameters(Vector<>)` | Updates model parameters. |
| `ValidateLayerConfiguration(List<ILayer<>>)` | Validates that custom layers meet Stable Audio requirements. |

