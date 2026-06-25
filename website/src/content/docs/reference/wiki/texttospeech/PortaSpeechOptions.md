---
title: "PortaSpeechOptions"
description: "Options for PortaSpeech (portable TTS with word-level prosody modeling and normalizing flow post-net)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.TextToSpeech.Classic`

Options for PortaSpeech (portable TTS with word-level prosody modeling and normalizing flow post-net).

## For Beginners

These options configure the PortaSpeech model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PortaSpeechOptions(PortaSpeechOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DurationScale` | Gets or sets the duration scale factor for phoneme duration prediction. |
| `MaxDuration` | Gets or sets the maximum frames per phoneme. |
| `NumFlowLayers` | Gets or sets the number of normalizing flow layers for the post-net. |
| `ProsodyDim` | Gets or sets the word-level prosody embedding dimension. |

