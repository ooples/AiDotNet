---
title: "ParakeetTDTOptions"
description: "Options for Parakeet-TDT (NVIDIA NeMo, 2024): 1.1B Conformer with Token-and-Duration Transducer."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.SpeechRecognition.NeMo`

Options for Parakeet-TDT (NVIDIA NeMo, 2024): 1.1B Conformer with Token-and-Duration Transducer.

## For Beginners

These options configure the ParakeetTDT model. Default values follow the original paper's recommended settings for optimal speech recognition accuracy.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ParakeetTDTOptions` | Initializes a new instance with default values. |
| `ParakeetTDTOptions(ParakeetTDTOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxDurationTokens` | Maximum duration tokens (number of frames to skip per emission). |

