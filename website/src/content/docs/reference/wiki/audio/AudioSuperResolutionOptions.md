---
title: "AudioSuperResolutionOptions"
description: "Configuration options for the Audio Super-Resolution model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Effects`

Configuration options for the Audio Super-Resolution model.

## For Beginners

Audio Super-Resolution is like AI-powered upscaling for sound.
Just as image super-resolution makes blurry photos sharper, audio super-resolution makes
low-quality audio sound clearer and more detailed.

Common uses:

- Upscaling old telephone recordings (8 kHz -> 44.1 kHz)
- Recovering quality from heavily compressed audio (MP3 at 64 kbps)
- Enhancing voice recordings from cheap microphones
- Restoring bandwidth-limited historical recordings

## How It Works

Audio Super-Resolution (Kuleshov et al., 2017; Li et al., 2021) upsamples low-resolution
audio to high-resolution audio using neural networks. It predicts the missing high-frequency
content that was lost during compression or low-quality recording, effectively converting
8 kHz telephone audio to 44.1 kHz studio quality, or recovering detail lost in MP3 compression.

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `HiddenDim` | Gets or sets the hidden dimension. |
| `InputSampleRate` | Gets or sets the input sample rate in Hz (low resolution). |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumAttentionLayers` | Gets or sets the number of attention layers. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumMels` | Gets or sets the number of mel bins for feature extraction. |
| `NumResBlocks` | Gets or sets the number of residual blocks. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `OutputSampleRate` | Gets or sets the output sample rate in Hz (high resolution). |
| `UpsampleFactor` | Gets or sets the upsampling factor (OutputSampleRate / InputSampleRate). |
| `Variant` | Gets or sets the model variant ("small", "base", "large"). |

