---
title: "MusicTaggingTransformerOptions"
description: "Configuration options for the Music Tagging Transformer model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.MusicAnalysis`

Configuration options for the Music Tagging Transformer model.

## For Beginners

This model listens to music and automatically tags it with descriptive
labels—like "rock", "upbeat", "guitar", "1980s", or "relaxing". It's the technology behind
automatic music categorization in streaming services like Spotify's genre detection.

## How It Works

The Music Tagging Transformer (Won et al., 2021) uses a Transformer encoder on mel spectrogram
features to predict music tags (genre, mood, instrument, era). It achieves state-of-the-art
results on the MagnaTagATune and Million Song Dataset benchmarks.

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `FeedForwardDim` | Gets or sets the feed-forward dimension. |
| `FftSize` | Gets or sets the FFT window size. |
| `HiddenDim` | Gets or sets the Transformer hidden dimension. |
| `HopLength` | Gets or sets the hop length between frames. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumAttentionHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of Transformer layers. |
| `NumMels` | Gets or sets the number of mel filterbank channels. |
| `NumTags` | Gets or sets the number of output tags. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |
| `TagLabels` | Gets or sets the tag label names. |

