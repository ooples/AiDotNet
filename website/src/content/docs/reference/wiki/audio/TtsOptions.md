---
title: "TtsOptions"
description: "Configuration options for text-to-speech models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.TextToSpeech`

Configuration options for text-to-speech models.

## For Beginners

Think of TTS as the opposite of speech recognition.
The acoustic model decides "what should this text sound like" (intonation, timing),
and the vocoder makes it actually sound like speech.

## How It Works

TTS (Text-to-Speech) converts written text into natural-sounding speech.
Modern TTS uses a two-stage pipeline:

1. Acoustic Model (e.g., FastSpeech2): Converts text to mel spectrogram
2. Vocoder (e.g., HiFi-GAN): Converts mel spectrogram to audio waveform

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TtsOptions` | Initializes a new instance with default values. |
| `TtsOptions(TtsOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AcousticModelPath` | Gets or sets the path to the acoustic model (FastSpeech2) ONNX file. |
| `Energy` | Gets or sets the energy/volume level. |
| `FftSize` | Gets or sets the FFT size. |
| `GriffinLimIterations` | Gets or sets the number of Griffin-Lim iterations if used. |
| `HopLength` | Gets or sets the hop length. |
| `Language` | Gets or sets the language code for multi-lingual models. |
| `NumMels` | Gets or sets the number of mel channels. |
| `OnnxOptions` | Gets or sets the ONNX execution options. |
| `PitchShift` | Gets or sets the pitch shift in semitones. |
| `SampleRate` | Gets or sets the output sample rate. |
| `SpeakerId` | Gets or sets the speaker ID for multi-speaker models. |
| `SpeakingRate` | Gets or sets the speaking rate multiplier. |
| `UseGriffinLimFallback` | Gets or sets whether to use Griffin-Lim as a fallback vocoder. |
| `VocoderModelPath` | Gets or sets the path to the vocoder (HiFi-GAN) ONNX file. |

