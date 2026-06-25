---
title: "ConformerOptions"
description: "Configuration options for the Conformer speech recognition model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.SpeechRecognition`

Configuration options for the Conformer speech recognition model.

## For Beginners

The Conformer is the go-to architecture for modern speech recognition.
It outperforms both pure Transformer and pure CNN encoders by using the best of both:

- Convolution captures local patterns like phonemes (individual sounds)
- Self-attention captures long-range context (e.g., sentence-level meaning)

The result is an encoder that understands both fine-grained and global speech patterns.

## How It Works

The Conformer (Gulati et al., 2020, Google) combines convolution and self-attention to
capture both local and global audio dependencies. It achieves state-of-the-art on
LibriSpeech (WER 1.9%/3.9% test-clean/other with LM) and is now the dominant encoder
architecture for production ASR systems (used in Google, NVIDIA NeMo, etc.).

## Properties

| Property | Summary |
|:-----|:--------|
| `ConvKernelSize` | Gets or sets the convolution kernel size in the Conformer block. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EncoderDim` | Gets or sets the encoder hidden dimension. |
| `FeedForwardExpansionFactor` | Gets or sets the feed-forward expansion factor. |
| `LabelSmoothing` | Gets or sets the label smoothing factor for CTC loss. |
| `Language` | Gets or sets the default language code. |
| `LearningRate` | Gets or sets the learning rate. |
| `MaxAudioLengthSeconds` | Gets or sets the maximum audio length in seconds. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumAttentionHeads` | Gets or sets the number of attention heads. |
| `NumEncoderLayers` | Gets or sets the number of Conformer encoder layers. |
| `NumMels` | Gets or sets the number of mel-spectrogram frequency bins. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |
| `SubsamplingFactor` | Gets or sets the subsampling factor for the encoder front-end. |
| `Variant` | Gets or sets the model variant ("small", "medium", "large"). |
| `VocabSize` | Gets or sets the vocabulary size for the CTC output head. |
| `Vocabulary` | Gets or sets the CTC vocabulary (characters or BPE tokens). |
| `WarmupSteps` | Gets or sets the warmup steps for the Noam learning-rate schedule. |

