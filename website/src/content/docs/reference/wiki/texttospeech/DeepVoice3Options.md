---
title: "DeepVoice3Options"
description: "Options for Deep Voice 3 (fully convolutional attention-based TTS)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.TextToSpeech.Classic`

Options for Deep Voice 3 (fully convolutional attention-based TTS).

## For Beginners

These options configure the DeepVoice3 model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepVoice3Options(DeepVoice3Options)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ConvKernelSize` | Gets or sets the convolution kernel size for encoder blocks. |
| `ConverterDim` | Gets or sets the converter (post-net) hidden dim. |
| `NumSpeakers` | Gets or sets the number of speaker embeddings for multi-speaker. |
| `SpeakerEmbeddingDim` | Gets or sets the speaker embedding dimension. |

