---
title: "Vocos<T>"
description: "Vocos: ConvNeXt-based vocoder that reconstructs waveform from Fourier coefficients (STFT magnitude + phase via ISTFT) instead of time-domain upsampling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TextToSpeech.Vocoders`

Vocos: ConvNeXt-based vocoder that reconstructs waveform from Fourier coefficients (STFT magnitude + phase via ISTFT) instead of time-domain upsampling.

## For Beginners

Vocos: ConvNeXt-based vocoder that reconstructs waveform from Fourier coefficients (STFT magnitude + phase via ISTFT) instead of time-domain upsampling.. This model converts text input into speech audio output.

## How It Works

**References:**

- Paper: "Vocos: Closing the Gap between Time-Domain and Fourier-Based Neural Vocoders for High-Quality Audio Synthesis" (Siuzdak, 2023)

## Methods

| Method | Summary |
|:-----|:--------|
| `MelToWaveform(Tensor<>)` | Converts mel to waveform using Vocos' ConvNeXt backbone predicting STFT coefficients. |

