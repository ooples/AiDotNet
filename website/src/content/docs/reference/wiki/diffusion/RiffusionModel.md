---
title: "RiffusionModel<T>"
description: "Riffusion model for music generation via spectrogram diffusion."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Audio`

Riffusion model for music generation via spectrogram diffusion.

## For Beginners

Riffusion creates music by first generating a "picture" of
the sound (spectrogram), then converting that picture back into actual audio.

How it works:

1. You describe the music you want: "jazz piano solo"
2. Riffusion generates a spectrogram (visual representation of sound)
3. The spectrogram is converted to playable audio

Key features:

- Text-to-music generation
- Style interpolation (blend two music styles)
- Real-time streaming generation
- Works with any Stable Diffusion checkpoint

What makes it unique:

- Treats audio generation as an image generation problem
- Can leverage all SD techniques: ControlNet, img2img, etc.
- Fast inference compared to autoregressive music models

## How It Works

Riffusion generates music by treating audio spectrograms as images and using
Stable Diffusion to generate them. The resulting spectrograms are then converted
back to audio using the Griffin-Lim algorithm or neural vocoders.

Technical details:

- Uses mel-spectrograms with specific parameters
- Typically 512x512 spectrogram images
- Griffin-Lim or neural vocoder for audio reconstruction
- Supports seed-based interpolation for smooth transitions
- Compatible with LoRA adapters for style transfer

Reference: Based on Riffusion project (riffusion.com)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RiffusionModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,SpectrogramConfig,Nullable<Int32>)` | Initializes a new instance of RiffusionModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `SpectrogramConfiguration` | Gets the spectrogram configuration. |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateAudioLength(Int32)` | Calculates audio length from spectrogram width. |
| `CalculateSpectrogramWidth(Double)` | Calculates spectrogram width from duration. |
| `Clone` |  |
| `DeepCopy` |  |
| `EnsureParameterShapesResolved` | Materializes lazy submodule weights before state-dict style operations. |
| `GenerateAudio(String,String,Double,Int32,Nullable<Double>,Nullable<Int32>)` | Generates audio directly from a text prompt. |
| `GenerateSpectrogram(String,String,Double,Int32,Nullable<Double>,Nullable<Int32>)` | Generates a spectrogram from a text prompt. |
| `GenerateSpectrogramInternal(String,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` | Generates a spectrogram with specific dimensions. |
| `GetParameters` |  |
| `InitializeLayers(UNetNoisePredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the U-Net, VAE, and audio processing components. |
| `InterpolateStyles(String,String,Double,Double,Int32,Nullable<Double>,Nullable<Int32>)` | Interpolates between two music styles. |
| `InterpolateTensors(Tensor<>,Tensor<>,Double)` | Interpolates between two tensors. |
| `SetParameters(Vector<>)` |  |
| `SpectrogramToAudio(Tensor<>)` | Converts a spectrogram to audio waveform. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DEFAULT_SPEC_SIZE` | Default spectrogram size in pixels. |
| `RIFF_LATENT_CHANNELS` | Number of latent channels for Riffusion (same as SD 1.5). |
| `RIFF_VAE_SCALE_FACTOR` | Spatial downsampling factor of the VAE. |
| `_conditioner` | The text conditioning module. |
| `_griffinLim` | GPU-accelerated Griffin-Lim processor for spectrogram inversion. |
| `_melSpectrogram` | GPU-accelerated mel spectrogram processor. |
| `_parameterShapesResolved` | Tracks whether lazy UNet/VAE parameter shapes have been materialized. |
| `_spectrogramConfig` | Spectrogram configuration. |
| `_unet` | The U-Net noise predictor. |
| `_vae` | The VAE for encoding/decoding spectrograms. |

