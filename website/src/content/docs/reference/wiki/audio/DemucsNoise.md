---
title: "DemucsNoise<T>"
description: "Demucs for Noise - real-time noise suppression using the Demucs architecture (Defossez et al., 2020, Meta)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Effects`

Demucs for Noise - real-time noise suppression using the Demucs architecture (Defossez et al., 2020, Meta).

## For Beginners

Demucs for Noise works like a music separator, but instead of
separating instruments, it separates clean speech from background noise. Feed it a noisy
recording and it outputs just the clean speech.

How it works:

1. Encoder: Progressively compresses audio into a compact representation
2. LSTM bottleneck: Captures temporal patterns in the compressed audio
3. Decoder: Reconstructs clean audio with skip connections from the encoder

**Usage:**

## How It Works

Demucs for Noise adapts the Demucs U-Net encoder-decoder architecture for real-time speech
denoising. Operating in the time domain with skip connections and LSTM bottleneck, it
achieves high-quality noise removal at 40ms latency. It was developed at Meta for
real-time communication applications.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DemucsNoise(NeuralNetworkArchitecture<>,DemucsNoiseOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a Demucs Noise model in native training mode. |
| `DemucsNoise(NeuralNetworkArchitecture<>,String,DemucsNoiseOptions)` | Creates a Demucs Noise model in ONNX inference mode. |

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

