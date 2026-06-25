---
title: "MusicStructureAnalyzerOptions"
description: "Configuration options for the Music Structure Analyzer model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.MusicAnalysis`

Configuration options for the Music Structure Analyzer model.

## For Beginners

This model listens to a song and identifies its sections—where the
verse begins, where the chorus kicks in, and where the bridge or outro happens. It's like
creating an automatic table of contents for a song.

## How It Works

The Music Structure Analyzer segments songs into structural sections (intro, verse, chorus,
bridge, outro) using a neural network trained on annotated music datasets. It combines
self-similarity matrix features with a segmentation network.

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `FftSize` | Gets or sets the FFT window size. |
| `HiddenDim` | Gets or sets the hidden dimension. |
| `HopLength` | Gets or sets the hop length between frames. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumAttentionHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of encoder layers. |
| `NumMels` | Gets or sets the number of mel filterbank channels. |
| `NumSections` | Gets or sets the number of section labels. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |
| `SectionLabels` | Gets or sets the section label names. |

