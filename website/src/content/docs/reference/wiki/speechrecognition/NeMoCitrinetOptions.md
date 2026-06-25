---
title: "NeMoCitrinetOptions"
description: "Options for NeMo Citrinet (NVIDIA, 2021): 1D time-channel separable convolution CTC model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.SpeechRecognition.NeMo`

Options for NeMo Citrinet (NVIDIA, 2021): 1D time-channel separable convolution CTC model.

## For Beginners

These options configure the NeMoCitrinet model. Default values follow the original paper's recommended settings for optimal speech recognition accuracy.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NeMoCitrinetOptions` | Initializes a new instance with default values. |
| `NeMoCitrinetOptions(NeMoCitrinetOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SqueezeExcitationRatio` | Squeeze-excitation reduction ratio for channel attention. |

