---
title: "IVocoder<T>"
description: "Interface for neural vocoders that convert mel-spectrograms to audio waveforms."
section: "API Reference"
---

`Interfaces` · `AiDotNet.TextToSpeech.Interfaces`

Interface for neural vocoders that convert mel-spectrograms to audio waveforms.

## How It Works

Vocoders form the second stage of a two-stage TTS pipeline:
Text -> [Acoustic Model] -> Mel-Spectrogram -> [Vocoder] -> Waveform.
Architectures include:

- Autoregressive: WaveNet, WaveRNN (sample-by-sample generation)
- GAN-based: HiFi-GAN, MelGAN, BigVGAN (adversarial training, parallel)
- Flow-based: WaveGlow (invertible flow, parallel)
- Diffusion-based: DiffWave, WaveGrad (denoising diffusion)
- Fourier-based: Vocos, iSTFTNet (inverse STFT output)

## Properties

| Property | Summary |
|:-----|:--------|
| `MelChannels` | Gets the expected number of mel channels in the input spectrogram. |
| `SampleRate` | Gets the audio sample rate in Hz (e.g., 22050, 24000). |
| `UpsampleFactor` | Gets the upsampling factor (hop_size) from mel frames to audio samples. |

## Methods

| Method | Summary |
|:-----|:--------|
| `MelToWaveform(Tensor<>)` | Converts a mel-spectrogram to an audio waveform. |

