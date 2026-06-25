---
title: "ForwardTacotronOptions"
description: "Options for Forward Tacotron (non-autoregressive Tacotron with duration predictor)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.TextToSpeech.Classic`

Options for Forward Tacotron (non-autoregressive Tacotron with duration predictor).

## For Beginners

These options configure the ForwardTacotron model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ForwardTacotronOptions(ForwardTacotronOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DurationScale` | Gets or sets the duration scale factor for phoneme duration prediction. |
| `HighwayDim` | Gets or sets the highway network dimension. |
| `PrenetDim` | Gets or sets the prenet dimension for the LSTM encoder. |

