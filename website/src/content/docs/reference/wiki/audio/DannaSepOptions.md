---
title: "DannaSepOptions"
description: "Configuration options for the Danna-Sep (Dual-path Attention Neural Network Audio Separator) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.SourceSeparation`

Configuration options for the Danna-Sep (Dual-path Attention Neural Network Audio Separator) model.

## For Beginners

Danna-Sep separates mixed music into individual instruments by
looking at audio in two ways: short-range patterns (what notes are being played right now)
and long-range patterns (how the music evolves over time). This dual perspective helps it
accurately pull apart overlapping instruments.

## How It Works

Danna-Sep (2024) uses dual-path attention with interleaved intra-chunk and inter-chunk
processing for music source separation. It achieves competitive results on MUSDB18 by
efficiently modeling both local spectral patterns and long-range temporal dependencies.

## Properties

| Property | Summary |
|:-----|:--------|
| `ChunkSize` | Gets or sets the chunk size for dual-path processing. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EncoderDim` | Gets or sets the encoder dimension. |
| `FftSize` | Gets or sets the FFT size for STFT computation. |
| `HopLength` | Gets or sets the hop length for STFT computation. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumChannels` | Gets or sets the number of audio channels. |
| `NumDualPathBlocks` | Gets or sets the number of dual-path blocks. |
| `NumFreqBins` | Gets or sets the number of frequency bins (FftSize/2 + 1). |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumSources` | Gets or sets the number of sources to separate. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `SourceNames` | Gets or sets the source names. |
| `Variant` | Gets or sets the model variant. |

