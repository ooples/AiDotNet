---
title: "FastConformerOptions"
description: "Configuration options for the Fast Conformer speech recognition model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.SpeechRecognition`

Configuration options for the Fast Conformer speech recognition model.

## For Beginners

Fast Conformer is NVIDIA's speed-optimized version of the Conformer.
It compresses audio early on (8x downsampling) so the expensive transformer layers process
much shorter sequences. Think of it as reading a summary instead of the full book - same
information, much faster processing.

## How It Works

Fast Conformer (Rekesh et al., 2023, NVIDIA NeMo) is an optimized Conformer variant
with 8x depthwise-separable convolution downsampling in the front-end, reducing the
sequence length early and enabling efficient processing of long audio. Combined with
multi-blank CTC or RNN-T, it achieves 2.4x speedup over standard Conformer with no
accuracy loss.

## Properties

| Property | Summary |
|:-----|:--------|
| `ConvKernelSize` | Gets or sets the convolution kernel size. |
| `DownsampleFactor` | Gets or sets the front-end downsampling factor. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EncoderDim` | Gets or sets the encoder hidden dimension. |
| `FeedForwardDim` | Gets or sets the feed-forward dimension. |
| `Language` | Gets or sets the language code. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of encoder layers. |
| `NumMels` | Gets or sets the number of mel spectrogram channels. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `Variant` | Gets or sets the model variant ("small", "medium", "large"). |
| `VocabSize` | Gets or sets the vocabulary size. |
| `Vocabulary` | Gets or sets the CTC vocabulary (characters or BPE tokens). |

