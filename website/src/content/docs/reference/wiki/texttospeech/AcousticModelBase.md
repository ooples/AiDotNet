---
title: "AcousticModelBase<T>"
description: "Base class for acoustic TTS models that generate mel-spectrograms from text."
section: "API Reference"
---

`Base Classes` · `AiDotNet.TextToSpeech`

Base class for acoustic TTS models that generate mel-spectrograms from text.

## How It Works

Acoustic models form the first stage of a two-stage TTS pipeline:
Text -> [Acoustic Model] -> Mel-Spectrogram -> [Vocoder] -> Waveform.

Subclasses include Tacotron, Tacotron2, FastSpeech, FastSpeech2, GlowTTS, GradTTS,
and other models that produce mel-spectrograms requiring a separate vocoder for audio output.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AcousticModelBase(NeuralNetworkArchitecture<>,ILossFunction<>)` | Initializes a new instance of the AcousticModelBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AiDotNet#TextToSpeech#Interfaces#IAcousticModel{T}#HopSize` | Gets the mel hop size. |
| `AiDotNet#TextToSpeech#Interfaces#IAcousticModel{T}#MelChannels` | Gets the number of mel frequency channels. |
| `AiDotNet#TextToSpeech#Interfaces#ITtsModel{T}#SampleRate` | Gets the sample rate. |
| `FftSize` | Gets the FFT window size. |
| `MaxTextLength` | Gets the maximum text length. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Synthesize(String)` |  |
| `TextToMel(String)` |  |

