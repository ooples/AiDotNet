---
title: "VocoderBase<T>"
description: "Base class for neural vocoder models that convert mel-spectrograms to audio waveforms."
section: "API Reference"
---

`Base Classes` · `AiDotNet.TextToSpeech`

Base class for neural vocoder models that convert mel-spectrograms to audio waveforms.

## How It Works

Vocoders form the second stage of a two-stage TTS pipeline:
Text -> [Acoustic Model] -> Mel-Spectrogram -> [Vocoder] -> Waveform.

Subclasses include HiFi-GAN, WaveNet, WaveRNN, BigVGAN, DiffWave, Vocos,
and other models that synthesize raw audio from mel-spectrogram representations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VocoderBase(NeuralNetworkArchitecture<>,ILossFunction<>)` | Initializes a new instance of the VocoderBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AiDotNet#TextToSpeech#Interfaces#IVocoder{T}#MelChannels` | Gets the mel channels. |
| `AiDotNet#TextToSpeech#Interfaces#IVocoder{T}#SampleRate` | Gets the sample rate. |
| `UpsampleFactor` | Gets the upsampling factor (hop size). |

## Methods

| Method | Summary |
|:-----|:--------|
| `MelToWaveform(Tensor<>)` |  |

