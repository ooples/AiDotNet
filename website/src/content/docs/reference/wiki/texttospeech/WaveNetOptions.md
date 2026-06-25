---
title: "WaveNetOptions"
description: "Options for WaveNet (autoregressive dilated causal convolution vocoder)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.TextToSpeech.Vocoders`

Options for WaveNet (autoregressive dilated causal convolution vocoder).

## For Beginners

These options configure the WaveNet model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WaveNetOptions(WaveNetOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MuLawLevels` | Gets or sets the number of mu-law quantization levels. |
| `NumDilatedLayers` | Gets or sets the number of dilated causal convolution layers. |
| `ResidualChannels` | Gets or sets the residual channel count. |
| `SkipChannels` | Gets or sets the skip channel count. |

