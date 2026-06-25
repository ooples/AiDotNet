---
title: "TFGridNetOptions"
description: "Configuration options for the TF-GridNet (Time-Frequency GridNet) speech enhancement model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Enhancement`

Configuration options for the TF-GridNet (Time-Frequency GridNet) speech enhancement model.

## For Beginners

TF-GridNet processes audio like reading a grid:

- Audio is represented as a time-frequency grid (spectrogram)
- The network looks across time (left-right) to track how sounds change
- Then it looks across frequency (up-down) to understand the harmonic structure
- By alternating these views, it builds a complete understanding of the audio

Imagine cleaning a dirty photo by first cleaning each row, then each column,
and repeating - that is how TF-GridNet cleans audio.

## How It Works

TF-GridNet (Wang et al., ICASSP 2023) applies alternating attention along the time and frequency
axes in a grid pattern, achieving state-of-the-art performance. On the WSJ0-2mix benchmark it
reaches 23.4 dB SI-SNRi, and on DNS Challenge 2020 it achieves PESQ 3.41.

**References:**

- Paper: "TF-GridNet: Making Time-Frequency Domain Models Great Again" (Wang et al., ICASSP 2023)
- Repository: https://github.com/espnet/espnet

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `EmbeddingDim` | Gets or sets the embedding dimension for each T-F bin. |
| `EnhancementStrength` | Gets or sets the enhancement strength (0.0 = no enhancement, 1.0 = maximum). |
| `FftSize` | Gets or sets the FFT window size. |
| `HiddenDim` | Gets or sets the hidden dimension for the grid network. |
| `HopLength` | Gets or sets the hop length between frames. |
| `InterFrameHiddenSize` | Gets or sets the LSTM hidden size for inter-frame (time) processing. |
| `IntraFrameHiddenSize` | Gets or sets the LSTM hidden size for intra-frame (frequency) processing. |
| `LearningRate` | Gets or sets the initial learning rate. |
| `ModelPath` | Gets or sets the path to a pre-trained ONNX model file. |
| `NumAttentionHeads` | Gets or sets the number of attention heads. |
| `NumBlocks` | Gets or sets the number of grid blocks (each contains time + frequency attention). |
| `NumFreqBins` | Gets or sets the number of frequency bins. |
| `NumSources` | Gets or sets the number of sources (1 for enhancement, 2+ for separation). |
| `OnnxOptions` | Gets or sets ONNX runtime options. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `WeightDecay` | Gets or sets the weight decay for regularization. |

