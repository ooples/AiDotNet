---
title: "AudioVAE<T>"
description: "Variational Autoencoder for audio mel-spectrogram encoding and decoding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.VAE`

Variational Autoencoder for audio mel-spectrogram encoding and decoding.

## For Beginners

Audio cannot be directly processed by diffusion models
because raw audio waveforms are very long (e.g., 10 seconds at 16kHz = 160,000 samples).
Instead, we use this pipeline:

Audio -> Mel Spectrogram -> VAE Encode -> Latent -> Diffusion -> VAE Decode -> Mel -> Vocoder -> Audio

The AudioVAE handles the "Mel -> Latent" and "Latent -> Mel" steps.

What is a mel spectrogram?

- A visual representation of sound
- X-axis: time, Y-axis: frequency (mel scale), Color: intensity
- Looks like an image, so we can use image-like networks!

Example dimensions:

- Mel spectrogram: [1, 64, 256] = 1 channel, 64 mel bins, 256 time frames
- Latent: [1, 8, 64] = 8 channels, 64 time frames (compressed)

## How It Works

The AudioVAE encodes mel spectrograms into a compressed latent representation
and decodes latents back to mel spectrograms. This is a key component of
audio latent diffusion models like AudioLDM.

Architecture:

- Encoder: 1D convolutions with downsampling along time axis
- Latent: Compressed representation with 8 channels
- Decoder: 1D transposed convolutions to reconstruct spectrogram
- Uses KL divergence for regularization

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioVAE` | Initializes a new AudioVAE with default parameters. |
| `AudioVAE(Int32,Int32,Int32,Int32[],Int32,ILossFunction<>,Nullable<Int32>)` | Initializes a new AudioVAE with custom parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DownsampleFactor` |  |
| `InputChannels` |  |
| `LatentChannels` |  |
| `LatentScaleFactor` |  |
| `MelChannels` | Gets the number of mel channels. |
| `ParameterCount` |  |
| `TimeDownsampleFactor` | Gets the time downsampling factor. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AudioToMelSpectrogram(Tensor<>,Int32,Int32,Int32)` | Converts raw audio waveform to mel spectrogram. |
| `BackpropagateLossGradient(Tensor<>)` |  |
| `Clone` |  |
| `CopyTransposedToResult(Tensor<>,Tensor<>,Int32,Int32,Int32)` | Copies transposed mel spectrogram to result tensor. |
| `Decode(Tensor<>)` |  |
| `DeepCopy` |  |
| `Encode(Tensor<>,Boolean)` |  |
| `EncodeWithDistribution(Tensor<>)` |  |
| `ExtractBatchItem(Tensor<>,Int32,Int32)` | Extracts a single waveform from a batch. |
| `GetParameters` |  |
| `InitializeLayers` | Initializes encoder and decoder layers. |
| `MelSpectrogramToAudio(Tensor<>,Int32,Int32)` | Converts mel spectrogram back to audio waveform. |
| `MelToFrequency(Int32,Int32,Int32)` | Converts mel bin index to frequency in Hz. |
| `SetParameters(Vector<>)` |  |
| `TransposeAndAddBatch(Tensor<>)` | Transposes mel spectrogram from [timeFrames, melChannels] to [1, melChannels, timeFrames]. |
| `TransposeFromBatch(Tensor<>,Int32,Int32,Int32)` | Transposes mel spectrogram from batch format [batch, melChannels, timeFrames] to [timeFrames, melChannels]. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_baseChannels` | Base channel count for conv layers. |
| `_channelMultipliers` | Channel multipliers for each level. |
| `_decoderLayers` | Decoder layers. |
| `_encoderLayers` | Encoder layers. |
| `_griffinLimProcessor` | GPU-accelerated Griffin-Lim processor for audio reconstruction. |
| `_lastInput` | Cached input for backward pass. |
| `_latentChannels` | Number of latent channels. |
| `_latentToDecoder` | Latent to decoder projection. |
| `_logVarProjection` | LogVar projection for latent. |
| `_melChannels` | Number of mel spectrogram channels (frequency bins). |
| `_melSpectrogramProcessor` | GPU-accelerated mel spectrogram processor. |
| `_muProjection` | Mu projection for latent. |
| `_numResBlocks` | Number of residual blocks per level. |
| `_timeDownsampleFactor` | Time downsampling factor (2^numLevels). |

