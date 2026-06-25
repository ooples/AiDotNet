---
title: "AudioMAEOptions"
description: "Configuration options for the Audio-MAE (Masked Autoencoders for Audio) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Classification`

Configuration options for the Audio-MAE (Masked Autoencoders for Audio) model.

## For Beginners

Audio-MAE learns by playing "fill in the blanks" with spectrograms.
It hides 80% of the audio picture and tries to reconstruct what was hidden. This forces
the model to truly understand audio patterns. After this pre-training, it can be fine-tuned
for classification with very little labeled data.

## How It Works

Audio-MAE (Xu et al., NeurIPS 2022) applies the Masked Autoencoder framework to audio
spectrograms for self-supervised pre-training. By masking 80% of spectrogram patches and
reconstructing them, Audio-MAE learns rich audio representations that achieve 47.3% mAP
on AudioSet and 97.0% on ESC-50 after fine-tuning.

**References:**

- Paper: "Masked Autoencoders that Listen" (Xu et al., NeurIPS 2022)
- Repository: https://github.com/facebookresearch/AudioMAE

## Properties

| Property | Summary |
|:-----|:--------|
| `CustomLabels` | Gets or sets custom event labels. |
| `DecoderEmbeddingDim` | Gets or sets the decoder embedding dimension. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EncoderEmbeddingDim` | Gets or sets the encoder embedding dimension. |
| `FMax` | Gets or sets the maximum frequency. |
| `FMin` | Gets or sets the minimum frequency. |
| `FeedForwardRatio` | Gets or sets the feed-forward dimension ratio. |
| `FftSize` | Gets or sets the FFT window size. |
| `HopLength` | Gets or sets the hop length. |
| `LabelSmoothing` | Gets or sets the label smoothing factor. |
| `LearningRate` | Gets or sets the initial learning rate. |
| `LocalAttentionWindow` | Gets or sets the local attention window size. |
| `MaskRatio` | Gets or sets the mask ratio for pre-training. |
| `ModelPath` | Gets or sets the path to a pre-trained ONNX model file. |
| `NumDecoderHeads` | Gets or sets the number of decoder attention heads. |
| `NumDecoderLayers` | Gets or sets the number of decoder layers. |
| `NumEncoderHeads` | Gets or sets the number of encoder attention heads. |
| `NumEncoderLayers` | Gets or sets the number of encoder layers. |
| `NumMels` | Gets or sets the number of mel bands. |
| `OnnxOptions` | Gets or sets ONNX runtime options. |
| `PatchSize` | Gets or sets the patch size. |
| `PatchStride` | Gets or sets the patch stride. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `Threshold` | Gets or sets the confidence threshold. |
| `UseLocalAttention` | Gets or sets whether to use local attention in the encoder. |
| `WarmUpEpochs` | Gets or sets the number of warm-up epochs. |
| `WindowOverlap` | Gets or sets the window overlap ratio. |
| `WindowSize` | Gets or sets the window size in seconds. |

