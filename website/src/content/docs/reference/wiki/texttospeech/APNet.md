---
title: "APNet<T>"
description: "APNet: amplitude-phase network that predicts amplitude and phase spectra separately then reconstructs waveform via iSTFT."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TextToSpeech.Vocoders`

APNet: amplitude-phase network that predicts amplitude and phase spectra separately then reconstructs waveform via iSTFT.

## For Beginners

APNet: amplitude-phase network that predicts amplitude and phase spectra separately then reconstructs waveform via iSTFT.. This model converts text input into speech audio output.

## How It Works

**References:**

- Paper: "APNet: Neural Vocoder that Generates Complex Spectrogram with Amplitude and Phase" (Ai et al., 2023)

## Methods

| Method | Summary |
|:-----|:--------|
| `MelToWaveform(Tensor<>)` | Converts mel to waveform using APNet's dual-stream amplitude and phase prediction. |

