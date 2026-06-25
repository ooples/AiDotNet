---
title: "ISTFTNet<T>"
description: "iSTFTNet: vocoder that predicts STFT magnitude and phase, then uses inverse STFT for waveform reconstruction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TextToSpeech.Vocoders`

iSTFTNet: vocoder that predicts STFT magnitude and phase, then uses inverse STFT for waveform reconstruction.

## For Beginners

iSTFTNet: vocoder that predicts STFT magnitude and phase, then uses inverse STFT for waveform reconstruction.. This model converts text input into speech audio output.

## How It Works

**References:**

- Paper: "iSTFTNet: Fast and Lightweight Mel-Spectrogram Vocoder Incorporating Inverse Short-Time Fourier Transform" (Kaneko et al., 2022)

## Methods

| Method | Summary |
|:-----|:--------|
| `MelToWaveform(Tensor<>)` | Converts mel to waveform by predicting STFT coefficients then applying inverse STFT. |

