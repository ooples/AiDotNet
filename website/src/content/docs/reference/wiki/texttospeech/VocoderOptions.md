---
title: "VocoderOptions"
description: "Base configuration options for neural vocoder models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.TextToSpeech.Vocoders`

Base configuration options for neural vocoder models.

## For Beginners

These options configure the Vocoder model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VocoderOptions` | Initializes a new instance with default values. |
| `VocoderOptions(VocoderOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumDiffusionSteps` | Gets or sets the number of diffusion steps (for diffusion vocoders). |
| `ResblockKernelSizes` | Gets or sets the resblock kernel sizes. |
| `UpsampleInitialChannels` | Gets or sets the initial upsample channel count. |
| `UpsampleKernelSizes` | Gets or sets the kernel sizes for each upsampling block. |
| `UpsampleRates` | Gets or sets the upsample rates for each upsampling block. |

