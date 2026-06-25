---
title: "HTSATOptions"
description: "Configuration options for the HTS-AT (Hierarchical Token-Semantic Audio Transformer) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Classification`

Configuration options for the HTS-AT (Hierarchical Token-Semantic Audio Transformer) model.

## For Beginners

HTS-AT is an efficient audio classifier that processes spectrograms
hierarchically (like zooming from overview to detail). It uses a technique called "window
attention" that looks at local regions first, then gradually combines them. This makes it
faster and more memory-efficient than models like AST that look at everything at once.

## How It Works

HTS-AT (Chen et al., ICASSP 2022) is a hierarchical Transformer architecture for audio
classification that uses Swin Transformer blocks with token-semantic modules to efficiently
process audio spectrograms. It achieves 47.1% mAP on AudioSet-2M with only 30M parameters.

**References:**

- Paper: "HTS-AT: A Hierarchical Token-Semantic Audio Transformer" (Chen et al., ICASSP 2022)
- Repository: https://github.com/RetroCirce/HTS-Audio-Transformer

## Properties

| Property | Summary |
|:-----|:--------|
| `AttentionDropoutRate` | Gets or sets the attention dropout rate. |
| `CustomLabels` | Gets or sets custom event labels. |
| `DetectionWindowSize` | Gets or sets the window size in seconds for event detection. |
| `DropPathRate` | Gets or sets the drop path rate for stochastic depth. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EmbeddingDim` | Gets or sets the embedding dimension. |
| `FMax` | Gets or sets the maximum frequency for mel filterbank. |
| `FMin` | Gets or sets the minimum frequency for mel filterbank. |
| `FeedForwardRatio` | Gets or sets the feed-forward expansion ratio. |
| `FftSize` | Gets or sets the FFT window size. |
| `HopLength` | Gets or sets the hop length between FFT frames. |
| `LabelSmoothing` | Gets or sets the label smoothing factor. |
| `LearningRate` | Gets or sets the initial learning rate. |
| `ModelPath` | Gets or sets the path to a pre-trained ONNX model file. |
| `NumHeadsPerStage` | Gets or sets the number of attention heads per stage. |
| `NumLayersPerStage` | Gets or sets the number of Swin Transformer layers per stage. |
| `NumMels` | Gets or sets the number of mel filterbank channels. |
| `NumSemanticGroups` | Gets or sets the number of semantic groups. |
| `OnnxOptions` | Gets or sets ONNX runtime options. |
| `PatchSize` | Gets or sets the patch size for initial embedding. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `Threshold` | Gets or sets the confidence threshold for event detection. |
| `UseTokenSemanticModule` | Gets or sets whether to use the token-semantic module. |
| `WarmUpSteps` | Gets or sets the warm-up steps. |
| `WindowOverlap` | Gets or sets the window overlap ratio (0-1). |
| `WindowSize` | Gets or sets the window size for local attention. |

