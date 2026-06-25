---
title: "CLAPOptions"
description: "Configuration options for the CLAP (Contrastive Language-Audio Pre-training) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Classification`

Configuration options for the CLAP (Contrastive Language-Audio Pre-training) model.

## For Beginners

CLAP is special because it understands both audio and text. Instead
of having fixed labels like "dog bark" or "siren", you can describe any sound in plain
English and CLAP will find it in audio. For example, you can search for "the sound of
rain hitting a tin roof" without ever training on that specific label.

How it works:

- CLAP has two encoders: one for audio and one for text
- Both encoders map their inputs to the same shared space
- Matching audio-text pairs are placed close together in this space
- To classify audio, compare it with text descriptions and find the closest match

## How It Works

CLAP (Wu et al., ICASSP 2023) learns joint audio-text representations through contrastive
learning, similar to CLIP for images. It can classify audio using natural language descriptions
without task-specific training (zero-shot classification), achieving 26.7% zero-shot accuracy
on ESC-50 and 46.8% mAP on AudioSet with fine-tuning.

**References:**

- Paper: "Large-Scale Contrastive Language-Audio Pre-Training with Feature Fusion and Keyword-to-Caption Augmentation" (Wu et al., ICASSP 2023)
- Repository: https://github.com/LAION-AI/CLAP

## Properties

| Property | Summary |
|:-----|:--------|
| `AudioEmbeddingDim` | Gets or sets the audio embedding dimension. |
| `AudioEncoderType` | Gets or sets the audio encoder type. |
| `CustomLabels` | Gets or sets custom event labels. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `FMax` | Gets or sets the maximum frequency. |
| `FMin` | Gets or sets the minimum frequency. |
| `FftSize` | Gets or sets the FFT window size. |
| `HopLength` | Gets or sets the hop length. |
| `LabelSmoothing` | Gets or sets the label smoothing factor. |
| `LearningRate` | Gets or sets the initial learning rate. |
| `MaxTextLength` | Gets or sets the maximum text token length. |
| `ModelPath` | Gets or sets the path to a pre-trained ONNX model file. |
| `NumAudioAttentionHeads` | Gets or sets the number of audio attention heads. |
| `NumAudioEncoderLayers` | Gets or sets the number of audio encoder layers. |
| `NumMels` | Gets or sets the number of mel bands. |
| `OnnxOptions` | Gets or sets ONNX runtime options. |
| `ProjectionDim` | Gets or sets the projection dimension for the joint space. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `Temperature` | Gets or sets the temperature parameter for contrastive loss. |
| `TextEmbeddingDim` | Gets or sets the text embedding dimension. |
| `TextEncoderModelPath` | Gets or sets the path to the text encoder ONNX model. |
| `TextEncoderType` | Gets or sets the text encoder type. |
| `TextPrompts` | Gets or sets text prompts for zero-shot classification. |
| `Threshold` | Gets or sets the confidence threshold. |
| `UseFeatureFusion` | Gets or sets whether to enable feature fusion. |
| `WarmUpSteps` | Gets or sets the warm-up steps. |
| `WindowOverlap` | Gets or sets the window overlap ratio. |
| `WindowSize` | Gets or sets the window size in seconds. |

