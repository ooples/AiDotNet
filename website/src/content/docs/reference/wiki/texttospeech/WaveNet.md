---
title: "WaveNet<T>"
description: "WaveNet: autoregressive generative model using dilated causal convolutions for raw audio generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TextToSpeech.Vocoders`

WaveNet: autoregressive generative model using dilated causal convolutions for raw audio generation.

## For Beginners

/// WaveNet: autoregressive generative model using dilated causal convolutions for raw audio generation.
. This model converts text input into speech audio output.

## How It Works

**References:**

- Paper: "WaveNet: A Generative Model for Raw Audio" (van den Oord et al., 2016)

## Methods

| Method | Summary |
|:-----|:--------|
| `MelToWaveform(Tensor<>)` | Converts mel-spectrogram to waveform using WaveNet's autoregressive sample-by-sample generation. |

