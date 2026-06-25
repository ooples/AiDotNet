---
title: "ASTModelOptions"
description: "Configuration options for AST (Audio Spectrogram Transformer) models (Gong et al."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for AST (Audio Spectrogram Transformer) models
(Gong et al. 2021).

## For Beginners

AST treats a mel spectrogram (a 2-D
time/frequency picture of an audio clip) as a sequence of small image
patches and runs a vision transformer over them. The default knob values
here reproduce AST-Base; you usually only need to change them when fine-
tuning on a different dataset (e.g., setting `NumClasses`) or
hardware budget (e.g., shrinking `EmbeddingDim` for mobile).

## How It Works

Defaults follow the published AST-Base recipe (Gong et al. 2021 §2):
128-mel input × ~1024 frames, 12 transformer layers × 12 heads × 768
embedding dim, 16×16 patches (paper §2.2), trained on AudioSet
(527 classes).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ASTModelOptions` | Initializes a new instance with AST-Base defaults. |
| `ASTModelOptions(ASTModelOptions)` | Initializes a new instance by copying every property from `other`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Dropout rate inside the transformer blocks. |
| `EmbeddingDim` | Transformer hidden / embedding dimension. |
| `FeedForwardDim` | Hidden dimension of the per-block feed-forward MLP. |
| `HopLength` | STFT hop length in samples between successive frames. |
| `NumClasses` | Number of output classes for the classification head. |
| `NumHeads` | Number of attention heads per transformer block. |
| `NumLayers` | Number of stacked transformer encoder layers. |
| `NumMelBands` | Number of mel filterbank bands per spectrogram frame. |
| `PatchSize` | Patch size H × W for the ViT patch embedding. |
| `SampleRate` | Audio sample rate in Hz used by the STFT frontend. |
| `StftWindowSize` | STFT window size in samples (analysis frame length). |
| `TargetLength` | Target spectrogram length in frames (time axis). |

