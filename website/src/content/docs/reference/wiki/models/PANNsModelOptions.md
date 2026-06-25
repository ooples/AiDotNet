---
title: "PANNsModelOptions"
description: "Configuration options for PANNs (Pretrained Audio Neural Networks) models (Kong et al."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for PANNs (Pretrained Audio Neural Networks) models
(Kong et al. 2020).

## For Beginners

PANNs is a family of pretrained convolutional
networks for audio tagging. CNN14 is the canonical balanced-accuracy
variant. The defaults here reproduce the AudioSet-pretrained checkpoint;
you usually only override `NumClasses` when fine-tuning on a
different label set.

## How It Works

Defaults follow the published CNN14 recipe (Kong et al. 2020 §3): 64 mel
bands, 32 kHz sample rate, 1024-sample STFT window, 320-sample hop, four
CNN stages (64→128→256→512 channels) + global pool + embedding head +
527-class AudioSet linear classifier.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PANNsModelOptions` | Initializes a new instance with PANNs CNN14 defaults. |
| `PANNsModelOptions(PANNsModelOptions)` | Initializes a new instance by copying every property from `other`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Dropout rate inside the CNN blocks. |
| `EmbeddingDim` | Embedding dimension produced before the classification head. |
| `HopLength` | STFT hop length in samples between successive frames. |
| `NumClasses` | Number of output classes for the classification head. |
| `NumMelBands` | Number of mel filterbank bands per spectrogram frame. |
| `SampleRate` | Audio sample rate in Hz used by the STFT frontend. |
| `StftWindowSize` | STFT window size in samples (analysis frame length). |

