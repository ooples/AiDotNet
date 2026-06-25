---
title: "ITtsModel<T>"
description: "Base interface for all text-to-speech models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.TextToSpeech.Interfaces`

Base interface for all text-to-speech models.

## How It Works

TTS models convert text input into audio waveform output. This base interface defines
the core synthesis method shared by all TTS architectures:

- Acoustic models (Tacotron, FastSpeech) that generate mel-spectrograms
- Vocoders (HiFi-GAN, WaveNet) that convert mel-spectrograms to waveforms
- End-to-end models (VITS) that go directly from text to waveform
- Codec-based models (VALL-E, CosyVoice) that use neural audio codecs

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxTextLength` | Gets the maximum input text length in characters. |
| `SampleRate` | Gets the audio sample rate in Hz (e.g., 22050, 24000, 44100). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Synthesize(String)` | Synthesizes audio waveform from text input. |

