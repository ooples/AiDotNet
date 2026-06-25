---
title: "CLAPModelOptions"
description: "Configuration options for CLAP (Contrastive Language-Audio Pretraining) models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for CLAP (Contrastive Language-Audio Pretraining) models.

## For Beginners

These knobs control how big and deep the audio and
text encoders are. The defaults match the published CLAP checkpoint and
produce 512-dim contrastive embeddings. Override individual fields to
build a smaller / faster variant.

## How It Works

Defaults follow the published CLAP "HTSAT + RoBERTa" recipe (Wu et al. 2023
"Large-Scale Contrastive Language-Audio Pre-Training with Feature Fusion
and Keyword-to-Caption Augmentation"). The audio encoder is an HTSAT-based
Swin Transformer (Chen et al. 2022; Liu et al. 2021); the text encoder
is a RoBERTa-style transformer stack.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CLAPModelOptions` | Initializes a new instance with CLAP HTSAT+RoBERTa defaults. |
| `CLAPModelOptions(CLAPModelOptions)` | Initializes a new instance by copying every property from `other`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AudioEncoderHeads` | Number of attention heads per Swin block. |
| `AudioEncoderLayers` | Number of Swin Transformer blocks stacked in the audio encoder. |
| `AudioHiddenDim` | Hidden / embedding dimension of the audio Swin blocks. |
| `AudioPatchSize` | HTSAT patch size — height and width of the 2D mel-spectrogram patch that gets embedded into one token. |
| `DropoutRate` | Dropout rate inside the transformer blocks. |
| `HopLength` | STFT hop length in audio samples. |
| `InitialTemperature` | Initial temperature τ for the contrastive softmax. |
| `MaxTextLength` | Maximum text token sequence length (CLAP truncates / pads to this). |
| `NumMelBands` | Number of mel-frequency bands extracted before the encoder. |
| `ProjectionDim` | Final shared-embedding-space dimension that both encoders project into. |
| `SampleRate` | Audio sample rate in Hz the encoder is configured for. |
| `StftWindowSize` | STFT window size in audio samples. |
| `SwinWindowSize` | Swin window size (W-MSA / SW-MSA) — Liu et al. |
| `TextEncoderHeads` | Number of attention heads per text-encoder layer. |
| `TextEncoderLayers` | Number of transformer encoder layers in the text encoder. |
| `TextHiddenDim` | Hidden / embedding dimension of the text encoder. |
| `VocabSize` | Text vocabulary size — RoBERTa BPE vocab. |

