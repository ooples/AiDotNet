---
title: "FastSpeech2<T>"
description: "FastSpeech 2: non-autoregressive TTS with variance adaptor for pitch, energy, and duration."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TextToSpeech.Classic`

FastSpeech 2: non-autoregressive TTS with variance adaptor for pitch, energy, and duration.

## For Beginners

/// FastSpeech 2: non-autoregressive TTS with variance adaptor for pitch, energy, and duration.
. This model converts text input into speech audio output.

## How It Works

**References:**

- Paper: "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech" (Ren et al., 2020)

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyVarianceAdaptor(Tensor<>,Int32,Int32)` | Applies the variance adaptor: duration prediction + length regulation + pitch/energy conditioning. |
| `Synthesize(String)` | Synthesizes audio waveform from text using FastSpeech 2's non-autoregressive pipeline. |
| `TextToMel(String)` | Generates a mel-spectrogram from text using FastSpeech 2. |

