---
title: "GradTTSOptions"
description: "Options for Grad-TTS (diffusion-based acoustic model with score matching)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.TextToSpeech.Classic`

Options for Grad-TTS (diffusion-based acoustic model with score matching).

## For Beginners

These options configure the GradTTS model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GradTTSOptions(GradTTSOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BetaEnd` | Gets or sets the noise schedule beta end. |
| `BetaStart` | Gets or sets the noise schedule beta start. |
| `NumDiffusionSteps` | Gets or sets the number of diffusion steps at inference. |

