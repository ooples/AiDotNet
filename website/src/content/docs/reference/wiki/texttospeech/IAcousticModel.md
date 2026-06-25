---
title: "IAcousticModel<T>"
description: "Interface for acoustic models that generate mel-spectrograms from text."
section: "API Reference"
---

`Interfaces` · `AiDotNet.TextToSpeech.Interfaces`

Interface for acoustic models that generate mel-spectrograms from text.

## How It Works

Acoustic models form the first stage of a two-stage TTS pipeline:
Text -> [Acoustic Model] -> Mel-Spectrogram -> [Vocoder] -> Waveform.
Architectures include:

- Autoregressive: Tacotron, Tacotron 2 (attention-based seq2seq)
- Non-autoregressive: FastSpeech, FastSpeech 2 (parallel with duration predictor)
- Flow-based: Glow-TTS (monotonic alignment search + flow)
- Diffusion-based: Grad-TTS (score-based diffusion decoder)

## Properties

| Property | Summary |
|:-----|:--------|
| `FftSize` | Gets the FFT window size used for spectrogram computation. |
| `HopSize` | Gets the mel-spectrogram hop size in audio samples. |
| `MelChannels` | Gets the number of mel-spectrogram frequency channels (typically 80). |

## Methods

| Method | Summary |
|:-----|:--------|
| `TextToMel(String)` | Generates a mel-spectrogram from text input. |

