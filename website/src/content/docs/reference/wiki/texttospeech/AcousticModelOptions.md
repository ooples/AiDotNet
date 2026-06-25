---
title: "AcousticModelOptions"
description: "Base configuration options for classic acoustic TTS models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.TextToSpeech.Classic`

Base configuration options for classic acoustic TTS models.

## For Beginners

These options configure the AcousticModel model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AcousticModelOptions` | Initializes a new instance with default values. |
| `AcousticModelOptions(AcousticModelOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DecoderDim` | Gets or sets the decoder output dimension (mel channels). |
| `EncoderDim` | Gets or sets the encoder embedding dimension. |
| `OutputsPerStep` | Gets or sets the number of mel frames generated per decoder step (reduction factor). |
| `PostnetDim` | Gets or sets the postnet embedding dimension. |
| `PostnetLayers` | Gets or sets the number of postnet convolution layers. |
| `UsePostnet` | Gets or sets whether to use a postnet for mel refinement. |

