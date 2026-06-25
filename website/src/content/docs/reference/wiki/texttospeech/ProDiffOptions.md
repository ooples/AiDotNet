---
title: "ProDiffOptions"
description: "Options for ProDiff (progressive fast diffusion model for high-quality TTS)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.TextToSpeech.Classic`

Options for ProDiff (progressive fast diffusion model for high-quality TTS).

## For Beginners

These options configure the ProDiff model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ProDiffOptions(ProDiffOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumDiffusionSteps` | Gets or sets the number of diffusion steps at inference (progressive reduces to 2-4). |
| `UseProgressiveDistillation` | Gets or sets whether to use knowledge distillation for step reduction. |

