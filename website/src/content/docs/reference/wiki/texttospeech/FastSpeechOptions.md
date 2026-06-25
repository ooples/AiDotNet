---
title: "FastSpeechOptions"
description: "Options for FastSpeech (non-autoregressive TTS with duration predictor)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.TextToSpeech.Classic`

Options for FastSpeech (non-autoregressive TTS with duration predictor).

## For Beginners

These options configure the FastSpeech model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FastSpeechOptions(FastSpeechOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DurationPredictorFilterSize` | Gets or sets the duration predictor filter size. |
| `DurationPredictorKernelSize` | Gets or sets the duration predictor kernel size. |
| `DurationScale` | Gets or sets the duration scale factor for phoneme duration prediction. |
| `MaxDuration` | Gets or sets the maximum frames per phoneme. |

