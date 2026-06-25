---
title: "AudioLDMModel<T>"
description: "AudioLDM (Audio Latent Diffusion Model) for generating audio from text descriptions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.AudioLDM`

AudioLDM (Audio Latent Diffusion Model) for generating audio from text descriptions.

## For Beginners

AudioLDM creates realistic audio from your descriptions:

How it works:

1. You describe the sound you want ("a cat meowing")
2. CLAP encodes your text into an audio-aligned representation
3. The diffusion process generates a latent audio representation
4. The VAE decoder converts latents to mel spectrogram
5. HiFi-GAN vocoder converts the spectrogram to audio

Key features:

- General audio and music generation
- Environmental sounds, speech, music
- Controllable through text prompts
- High-quality 16kHz or 48kHz output

Usage:

## How It Works

AudioLDM is a latent diffusion model that generates audio by learning to reverse
a diffusion process in a compressed latent space. It uses CLAP (Contrastive Language-Audio
Pretraining) for text conditioning and a VAE for efficient latent space learning.

Architecture components:

- **CLAP Encoder:** Contrastive text encoder that aligns text with audio features
- **VAE:** Variational autoencoder that compresses mel spectrograms to latent space
- **U-Net Denoiser:** Predicts noise to be removed at each diffusion step
- **HiFi-GAN Vocoder:** Converts mel spectrograms to audio waveforms

Reference: "AudioLDM: Text-to-Audio Generation with Latent Diffusion Models" by Liu et al., 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioLDMModel(NeuralNetworkArchitecture<>,AudioLDMOptions,ITokenizer,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates an AudioLDM model using native layers for training from scratch. |
| `AudioLDMModel(NeuralNetworkArchitecture<>,String,String,String,String,ITokenizer,AudioLDMOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates an AudioLDM model using pretrained ONNX models for inference. |

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
| `ApplyCrossAttentionConditioning(Tensor<>,Tensor<>)` | Applies cross-attention conditioning to latents. |
| `ApplyGriffinLim(Tensor<>)` | Applies Griffin-Lim algorithm for phase reconstruction. |
| `ApplyGuidance(Tensor<>,Tensor<>,Double)` | Applies classifier-free guidance. |
| `ApplyMelFilterbank(Double[],Int32,Int32,Int32)` | Applies mel filterbank to power spectrum. |
| `CombineConditionings(Tensor<>,Tensor<>)` | Combines audio and text conditionings. |
| `ComputeMelSpectrogram(Tensor<>)` | Computes mel spectrogram from audio. |
| `ContinueAudio(Tensor<>,String,Double,Int32,Nullable<Int32>)` | Continues existing audio by extending it. |
| `CreateNewInstance` | Creates a new instance for cloning. |
| `CreateTimestepTensor(Int32)` | Creates a timestep tensor with sinusoidal embedding. |
| `CreateUnconditionalEmbedding` | Creates unconditional embedding for classifier-free guidance. |
| `DdpmStep(Tensor<>,Tensor<>,Int32,Random)` | Performs a DDPM denoising step. |
| `DecodeLatentsToAudio(Tensor<>)` | Decodes latents to audio waveform. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data. |
| `Dispose(Boolean)` | Disposes of model resources. |
| `EncodeAudioToLatents(Tensor<>)` | Encodes audio to latent space. |
| `EncodeClapEmbedding(Tensor<>)` | Encodes CLAP embedding from tokens. |
| `EncodeTextToTensor(String)` | Encodes text prompt to token tensor. |
| `ForwardThroughUNet(Tensor<>)` | Forwards input through U-Net layers. |
| `GenerateAudio(String,String,Double,Int32,Double,Nullable<Int32>)` | Generates audio from a text description. |
| `GenerateAudioAsync(String,String,Double,Int32,Double,Nullable<Int32>,CancellationToken)` | Generates audio asynchronously. |
| `GenerateMusic(String,String,Double,Int32,Double,Nullable<Int32>)` | Generates music from a text description. |
| `GenerateTimesteps(Int32)` | Generates diffusion timesteps. |
| `GetAlphaCumprod(Int32)` | Gets cumulative product of alphas up to timestep. |
| `GetBetaSchedule(Int32)` | Gets beta value for a timestep. |
| `GetDefaultOptions` | Gets default generation options. |
| `GetModelDimensions(AudioLDMModelSize)` | Gets model dimensions based on the selected size. |
| `GetModelMetadata` | Gets model metadata. |
| `GetOptions` |  |
| `InitializeLatents(Int32,Random)` | Initializes latent noise for diffusion. |
| `InitializeLayers` | Initializes the neural network layers following the golden standard pattern. |
| `InpaintAudio(Tensor<>,Tensor<>,String,Int32,Nullable<Int32>)` | Fills in missing or masked sections of audio. |
| `PostprocessOutput(Tensor<>)` | Postprocesses model output. |
| `PredictCore(Tensor<>)` | Makes a prediction using the model. |
| `PrepareUNetInput(Tensor<>,Tensor<>,Tensor<>)` | Prepares U-Net input by combining latents, conditioning, and timestep. |
| `PreprocessAudio(Tensor<>)` | Preprocesses raw audio for model input. |
| `RunDiffusionLoop(Tensor<>,Tensor<>,Tensor<>,Int32,Double,Random)` | Runs the diffusion denoising loop. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data. |
| `Train(Tensor<>,Tensor<>)` | Trains the model on input data. |
| `UpdateParameters(Vector<>)` | Updates model parameters. |
| `ValidateLayerConfiguration(List<ILayer<>>)` | Validates that custom layers meet AudioLDM requirements. |

