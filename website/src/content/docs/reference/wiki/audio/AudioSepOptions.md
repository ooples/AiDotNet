---
title: "AudioSepOptions"
description: "Configuration options for the AudioSep (Audio Separation with Natural Language Queries) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Classification`

Configuration options for the AudioSep (Audio Separation with Natural Language Queries) model.

## For Beginners

AudioSep is like having a smart audio assistant that understands
natural language. Instead of being limited to a fixed set of 527 sound categories:

- You can ask: "Find the sound of a baby crying" and it will detect/separate it
- You can ask: "Extract the bird chirping" from a noisy recording
- You can describe any sound in your own words, and AudioSep understands

This is possible because AudioSep combines two powerful ideas:

1. **CLAP**: A model that understands the relationship between sounds and text descriptions
2. **Separation network**: A U-Net that extracts the described sound from the mixture

AudioSep achieves state-of-the-art results on both sound separation and sound event detection
benchmarks, making it one of the most versatile audio models available.

## How It Works

AudioSep (Liu et al., ICML 2024) is a foundation model for open-vocabulary audio separation
and sound event detection. Unlike traditional SED models that use fixed label sets, AudioSep
can detect and separate any sound described by natural language. It uses CLAP (Contrastive
Language-Audio Pretraining) embeddings to condition a separation network, enabling queries like
"separate the dog barking from the traffic noise" or "detect the sound of glass breaking."

## Properties

| Property | Summary |
|:-----|:--------|
| `CLAPEmbeddingDim` | Gets or sets the CLAP embedding dimension. |
| `CustomLabels` | Gets or sets custom event labels. |
| `DetectionWindowSize` | Gets or sets the window size in seconds for detection. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EncoderChannels` | Gets or sets the U-Net encoder channels. |
| `FMax` | Gets or sets the maximum frequency for mel filterbank. |
| `FMin` | Gets or sets the minimum frequency for mel filterbank. |
| `FftSize` | Gets or sets the FFT window size. |
| `HopLength` | Gets or sets the hop length between FFT frames. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to a pre-trained ONNX model file. |
| `NumHeads` | Gets or sets the number of attention heads in the separator. |
| `NumMels` | Gets or sets the number of mel filterbank channels. |
| `NumSeparationLayers` | Gets or sets the number of separation network layers. |
| `OnnxOptions` | Gets or sets ONNX runtime options. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `SeparationDim` | Gets or sets the separation network hidden dimension. |
| `Threshold` | Gets or sets the confidence threshold for event detection. |
| `Variant` | Gets or sets the model variant ("base", "large"). |
| `WindowOverlap` | Gets or sets the window overlap ratio (0-1). |

