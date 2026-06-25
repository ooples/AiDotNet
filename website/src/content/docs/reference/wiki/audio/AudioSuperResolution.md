---
title: "AudioSuperResolution<T>"
description: "Audio Super-Resolution model for upsampling low-resolution audio to high-resolution (Kuleshov et al., 2017; Li et al., 2021)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Effects`

Audio Super-Resolution model for upsampling low-resolution audio to high-resolution
(Kuleshov et al., 2017; Li et al., 2021).

## For Beginners

Audio Super-Resolution is like AI-powered upscaling for sound.
Just as image super-resolution makes blurry photos sharper, this model makes low-quality
audio sound clearer and more detailed.

Common uses:

- Upscaling old telephone recordings (8 kHz to 44.1 kHz)
- Recovering quality from heavily compressed audio (MP3 at 64 kbps)
- Enhancing voice recordings from cheap microphones
- Restoring bandwidth-limited historical recordings

How it works:

1. Takes a low-resolution audio waveform as input
2. Passes through residual blocks that learn to predict missing high-frequency content
3. Outputs a high-resolution waveform with restored detail

**Usage:**

## How It Works

Audio Super-Resolution uses deep neural networks to predict missing high-frequency
content in low-resolution audio. Given input at a low sample rate (e.g., 8 kHz telephone
quality), it reconstructs audio at a higher sample rate (e.g., 44.1 kHz studio quality)
by predicting the missing frequency bands. The architecture uses residual blocks with
attention modules to capture both local and global spectral patterns.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioSuperResolution(NeuralNetworkArchitecture<>,AudioSuperResolutionOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an Audio Super-Resolution model in native training mode. |
| `AudioSuperResolution(NeuralNetworkArchitecture<>,String,AudioSuperResolutionOptions)` | Creates an Audio Super-Resolution model in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EnhancementStrength` |  |
| `LatencySamples` |  |
| `NumChannels` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Enhance(Tensor<>)` |  |
| `EnhanceWithReference(Tensor<>,Tensor<>)` |  |
| `EstimateNoiseProfile(Tensor<>)` |  |
| `ProcessChunk(Tensor<>)` |  |

