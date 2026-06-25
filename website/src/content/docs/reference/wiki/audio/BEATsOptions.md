---
title: "BEATsOptions"
description: "Configuration options for the BEATs (Audio Pre-Training with Acoustic Tokenizers) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Classification`

Configuration options for the BEATs (Audio Pre-Training with Acoustic Tokenizers) model.

## For Beginners

BEATs is a powerful model for recognizing sounds in audio.
It works by:

- Breaking audio into small patches (like cutting a spectrogram into tiles)
- Using a Transformer (the same architecture behind ChatGPT) to understand relationships between patches
- Learning what sounds are present by comparing against known audio patterns

You can use BEATs in two ways:

- **ONNX mode**: Load a pre-trained model for instant inference
- **Native mode**: Train from scratch on your own audio data

Example usage:

## How It Works

BEATs (Chen et al., ICML 2023) is a state-of-the-art audio classification model that uses
iterative self-distillation between an acoustic tokenizer and an audio SSL model. It achieves
50.6% mAP on AudioSet-2M and 98.1% accuracy on ESC-50, setting new benchmarks for audio
event detection and classification tasks.

**Architecture Overview:**

- Audio waveform is converted to a mel spectrogram
- Spectrogram patches are extracted and linearly projected to embedding vectors
- Positional embeddings are added to preserve spatial information
- A stack of Transformer encoder layers processes the patch embeddings
- A classification head maps the aggregated features to event labels

**References:**

- Paper: "BEATs: Audio Pre-Training with Acoustic Tokenizers" (Chen et al., ICML 2023)
- Repository: https://github.com/microsoft/unilm/tree/master/beats

## Properties

| Property | Summary |
|:-----|:--------|
| `AttentionDropoutRate` | Gets or sets the attention dropout rate. |
| `CodebookSize` | Gets or sets the codebook size for the acoustic tokenizer. |
| `CustomLabels` | Gets or sets custom event labels. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `EmbeddingDim` | Gets or sets the embedding dimension of the Transformer encoder. |
| `FMax` | Gets or sets the maximum frequency for the mel filterbank in Hz. |
| `FMin` | Gets or sets the minimum frequency for the mel filterbank in Hz. |
| `FeedForwardDim` | Gets or sets the feed-forward network dimension in each Transformer layer. |
| `FftSize` | Gets or sets the FFT window size for spectrogram computation. |
| `HopLength` | Gets or sets the hop length between consecutive FFT frames. |
| `LabelSmoothing` | Gets or sets the label smoothing factor for classification. |
| `LearningRate` | Gets or sets the initial learning rate for training. |
| `MaskProbability` | Gets or sets the mask probability for masked patch prediction pre-training. |
| `MinMaskSpanLength` | Gets or sets the minimum span length for contiguous masking. |
| `ModelPath` | Gets or sets the path to a pre-trained ONNX model file. |
| `NumAttentionHeads` | Gets or sets the number of attention heads in each Transformer layer. |
| `NumEncoderLayers` | Gets or sets the number of Transformer encoder layers. |
| `NumMels` | Gets or sets the number of mel filterbank channels. |
| `OnnxOptions` | Gets or sets ONNX runtime options. |
| `PatchSize` | Gets or sets the patch size (height) for spectrogram patching. |
| `PatchStride` | Gets or sets the patch stride for spectrogram patching. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `Threshold` | Gets or sets the confidence threshold for event detection. |
| `TokenizerIterations` | Gets or sets the number of acoustic tokenizer iterations. |
| `WarmUpSteps` | Gets or sets the number of warm-up steps for learning rate scheduling. |
| `WindowOverlap` | Gets or sets the window overlap ratio (0-1) for event detection. |
| `WindowSize` | Gets or sets the window size in seconds for event detection. |

