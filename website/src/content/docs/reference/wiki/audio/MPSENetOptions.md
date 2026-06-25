---
title: "MPSENetOptions"
description: "Configuration options for the MP-SENet (Multi-Path Speech Enhancement Network) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Enhancement`

Configuration options for the MP-SENet (Multi-Path Speech Enhancement Network) model.

## For Beginners

Sound has two components: loudness (magnitude) and timing (phase).
Most enhancers only fix loudness and leave timing alone, which limits quality.
MP-SENet fixes both simultaneously using two parallel paths that share information,
leading to cleaner and more natural-sounding audio.

## How It Works

MP-SENet (Lu et al., 2023) predicts both magnitude and phase of the complex spectrogram
using parallel magnitude and phase estimation paths with a cross-domain fusion module.
It achieves PESQ 3.60 on VoiceBank+DEMAND, surpassing prior single-channel methods.

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `FFTSize` | Gets or sets the FFT size for STFT computation. |
| `FeedForwardDim` | Gets or sets the feed-forward dimension. |
| `HiddenDim` | Gets or sets the encoder hidden dimension. |
| `HopLength` | Gets or sets the hop length for STFT computation. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumAttentionHeads` | Gets or sets the number of attention heads. |
| `NumFreqBins` | Gets or sets the number of frequency bins (FFTSize / 2 + 1). |
| `NumLayers` | Gets or sets the number of encoder layers. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |
| `Variant` | Gets or sets the model variant ("small" or "large"). |

