---
title: "EndToEndTtsOptions"
description: "Base options for end-to-end TTS models that generate waveforms directly from text."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.TextToSpeech.EndToEnd`

Base options for end-to-end TTS models that generate waveforms directly from text.

## For Beginners

These options configure the EndToEndTts model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EndToEndTtsOptions(EndToEndTtsOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DecoderDim` | Gets or sets the decoder hidden dimension (defaults to HiddenDim). |
| `EncoderDim` | Gets or sets the encoder hidden dimension (defaults to HiddenDim). |
| `FilterChannels` | Gets or sets the filter channels for the text encoder. |
| `InterChannels` | Gets or sets the inter-channel dimension for the normalizing flow. |
| `NumDiffusionSteps` | Gets or sets the number of diffusion/denoising steps (if applicable). |
| `NumFlowSteps` | Gets or sets the number of flow steps in the posterior encoder. |

