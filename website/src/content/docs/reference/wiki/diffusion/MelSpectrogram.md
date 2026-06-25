---
title: "MelSpectrogram<T>"
description: "Computes Mel spectrograms from audio signals."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Audio`

Computes Mel spectrograms from audio signals.

## For Beginners

Human hearing doesn't perceive pitch linearly - we can
tell the difference between 100Hz and 200Hz easily, but 10,000Hz and 10,100Hz
sound almost the same to us. The Mel scale accounts for this.

A Mel spectrogram:

1. Computes the power spectrogram using STFT
2. Applies a bank of triangular filters on the Mel scale
3. Takes the log (optional) to compress dynamic range

This representation is commonly used for:

- Speech recognition (like Whisper)
- Music generation (like Riffusion)
- Audio classification
- Speaker verification

Usage:
```cs
var melSpec = new MelSpectrogram<float>(
sampleRate: 44100,
nMels: 128,
nFft: 2048
);
var mel = melSpec.Forward(audioSignal);
// mel.Shape = [numFrames, nMels]
```

## How It Works

The Mel spectrogram is a representation of audio that mimics human hearing.
It applies the Mel scale, which spaces frequencies according to how humans
perceive pitch rather than the physical frequency.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MelSpectrogram(Int32,Int32,Int32,Nullable<Int32>,Double,Nullable<Double>,IWindowFunction<>,Boolean,Double,Double)` | Initializes a new Mel spectrogram processor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | IEngine for GPU-accelerated operations. |
| `NumMels` | Gets the number of Mel bins. |
| `STFT` | Gets the STFT parameters. |
| `SampleRate` | Gets the sample rate. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyMelFilterbank(Tensor<>)` | Applies the Mel filterbank to a power spectrogram. |
| `CreateMelFilterbank(Int32,Int32,Int32,Double,Double)` | Creates a Mel filterbank matrix. |
| `DbToPower(Tensor<>)` | Converts dB spectrogram back to power. |
| `Forward(Tensor<>)` | Computes the Mel spectrogram of an audio signal. |
| `FromPowerSpectrogram(Tensor<>)` | Computes Mel spectrogram from a pre-computed power spectrogram. |
| `GetFilterbank` | Gets the Mel filterbank matrix. |
| `GetMelCenterFrequencies` | Computes the frequency (in Hz) for each Mel bin center. |
| `HzToMel(Double)` | Converts frequency in Hz to Mel scale. |
| `InvertMelToMagnitude(Tensor<>,Nullable<Boolean>)` | Inverts a Mel spectrogram to approximate magnitude spectrogram. |
| `MelToHz(Double)` | Converts frequency in Mels to Hz. |
| `PowerToDb(Tensor<>)` | Converts power spectrogram to decibels. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides numeric operations for the specific type T. |
| `_fMax` | Maximum frequency in Hz. |
| `_fMin` | Minimum frequency in Hz. |
| `_hopLength` | Hop length for direct GPU operations. |
| `_logMel` | Whether to apply log compression. |
| `_melFilterbank` | Mel filterbank matrix [nMels, nFreqs]. |
| `_minDb` | Minimum dB value (floor). |
| `_nFft` | FFT size for direct GPU operations. |
| `_nMels` | Number of Mel frequency bins. |
| `_refDb` | Reference value for dB conversion. |
| `_sampleRate` | Sample rate of the audio in Hz. |
| `_stft` | The STFT processor. |
| `_windowTensor` | Window tensor for IEngine operations. |

