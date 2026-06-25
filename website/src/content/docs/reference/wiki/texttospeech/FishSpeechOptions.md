---
title: "FishSpeechOptions"
description: "Options for FishSpeech (Fish Audio, 2024) dual-AR architecture with GFSQ."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.TextToSpeech.CodecBased`

Options for FishSpeech (Fish Audio, 2024) dual-AR architecture with GFSQ.

## For Beginners

These options configure the FishSpeech model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FishSpeechOptions(FishSpeechOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MinReferenceSeconds` | Gets or sets the minimum reference audio duration in seconds for voice cloning. |
| `NumGroups` | Gets or sets the number of GFSQ groups for grouped finite scalar quantization. |
| `RepetitionPenalty` | Gets or sets the repetition penalty factor. |
| `Temperature` | Gets or sets the sampling temperature for generation. |
| `TopP` | Gets or sets the top-p (nucleus) sampling parameter. |

