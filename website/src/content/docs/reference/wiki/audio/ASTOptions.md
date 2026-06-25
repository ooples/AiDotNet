---
title: "ASTOptions"
description: "Configuration options for the AST (Audio Spectrogram Transformer) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Classification`

Configuration options for the AST (Audio Spectrogram Transformer) model.

## For Beginners

AST treats audio spectrograms exactly like images. It cuts the
spectrogram into small patches (like puzzle pieces), then uses a Transformer to understand
the relationships between patches. This approach works surprisingly well because spectrograms
are 2D representations of sound that look similar to images.

You can use AST in two ways:

- **ONNX mode**: Load a pre-trained model for instant inference
- **Native mode**: Train from scratch on your own audio data

Example usage:

## How It Works

AST (Gong et al., Interspeech 2021) is the first purely attention-based model for audio
classification that directly applies a Vision Transformer to audio spectrograms. It achieves
45.9% mAP on AudioSet and 95.6% accuracy on ESC-50.

**Architecture Overview:**

- Audio waveform is converted to a mel spectrogram (128 bins)
- The spectrogram is split into 16x16 patches with overlap
- Patches are linearly projected to embedding vectors with [CLS] token prepended
- A stack of Transformer encoder layers processes the patch embeddings
- The [CLS] token output is used for classification

**References:**

- Paper: "AST: Audio Spectrogram Transformer" (Gong et al., Interspeech 2021)
- Repository: https://github.com/YuanGongND/ast

## Properties

| Property | Summary |
|:-----|:--------|
| `AttentionDropoutRate` | Gets or sets the attention dropout rate. |
| `CustomLabels` | Gets or sets custom event labels. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `EmbeddingDim` | Gets or sets the embedding dimension of the Transformer encoder. |
| `FMax` | Gets or sets the maximum frequency for the mel filterbank in Hz. |
| `FMin` | Gets or sets the minimum frequency for the mel filterbank in Hz. |
| `FeedForwardDim` | Gets or sets the feed-forward network dimension in each Transformer layer. |
| `FftSize` | Gets or sets the FFT window size for spectrogram computation. |
| `HopLength` | Gets or sets the hop length between consecutive FFT frames. |
| `LabelSmoothing` | Gets or sets the label smoothing factor. |
| `LearningRate` | Gets or sets the initial learning rate for training. |
| `ModelPath` | Gets or sets the path to a pre-trained ONNX model file. |
| `NumAttentionHeads` | Gets or sets the number of attention heads in each Transformer layer. |
| `NumEncoderLayers` | Gets or sets the number of Transformer encoder layers. |
| `NumMels` | Gets or sets the number of mel filterbank channels. |
| `OnnxOptions` | Gets or sets ONNX runtime options. |
| `PatchSize` | Gets or sets the patch size for spectrogram patching. |
| `PatchStride` | Gets or sets the patch stride for overlapping patches. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `Threshold` | Gets or sets the confidence threshold for event detection. |
| `UseImageNetPretrain` | Gets or sets whether to use ImageNet-pretrained weights for initialization. |
| `WarmUpEpochs` | Gets or sets the number of warm-up epochs for learning rate scheduling. |
| `WindowOverlap` | Gets or sets the window overlap ratio (0-1) for event detection. |
| `WindowSize` | Gets or sets the window size in seconds for event detection. |

