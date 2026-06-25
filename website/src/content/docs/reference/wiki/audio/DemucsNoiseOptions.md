---
title: "DemucsNoiseOptions"
description: "Configuration options for the Demucs for Noise model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Effects`

Configuration options for the Demucs for Noise model.

## For Beginners

Demucs for Noise is like using a music separator to pull apart
"speech" and "noise" tracks from a recording. The original Demucs separates vocals,
drums, bass, and other instruments. This version separates clean speech from background
noise - perfect for cleaning up phone calls, podcasts, and video meetings.

## How It Works

Demucs for Noise (Defossez et al., 2020, Meta) adapts the Demucs architecture
(originally for music source separation) for real-time noise suppression. It operates
in the time domain with a U-Net encoder-decoder structure and skip connections,
achieving high-quality noise removal at low latency (40ms).

## Properties

| Property | Summary |
|:-----|:--------|
| `ChannelGrowth` | Gets or sets the channel growth factor per depth level. |
| `Depth` | Gets or sets the encoder/decoder depth (number of layers). |
| `DropoutRate` | Gets or sets the dropout rate. |
| `HiddenChannels` | Gets or sets the initial hidden channels. |
| `KernelSize` | Gets or sets the kernel size for the encoder convolutions. |
| `LSTMHiddenSize` | Gets or sets the LSTM hidden size for the bottleneck. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumChannels` | Gets or sets the number of audio channels. |
| `NumLSTMLayers` | Gets or sets the number of LSTM layers in the bottleneck. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `Stride` | Gets or sets the stride for the encoder convolutions. |
| `Variant` | Gets or sets the model variant ("small", "base", "large"). |

