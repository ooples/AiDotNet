---
title: "AudioProcessor<T>"
description: "Complete audio processing pipeline for diffusion-based audio generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Audio`

Complete audio processing pipeline for diffusion-based audio generation.

## For Beginners

This is your one-stop shop for working with audio in
diffusion models. It handles:

- Converting audio waveforms to spectrograms (for training/conditioning)
- Converting spectrograms back to audio (for generation)
- Normalizing and denormalizing spectrograms

Typical workflow for Riffusion-style generation:
```cs
var processor = new AudioProcessor<float>(sampleRate: 44100);

// Encode reference audio to latent space (via spectrogram)
var spectrogram = processor.AudioToSpectrogram(referenceAudio);
var normalized = processor.NormalizeSpectrogram(spectrogram);

// ... diffusion model generates new spectrogram ...

// Decode generated spectrogram back to audio
var denormalized = processor.DenormalizeSpectrogram(generatedSpec);
var audio = processor.SpectrogramToAudio(denormalized);
```

## How It Works

This class combines STFT, Mel spectrogram, and Griffin-Lim into a unified
pipeline for audio analysis and synthesis. It's designed for use with
diffusion models like Riffusion that generate spectrograms.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioProcessor(Int32,Int32,Int32,Int32,Double,Nullable<Double>,Double,Double,Int32)` | Initializes a new audio processor with Riffusion-compatible defaults. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GriffinLim` | Gets the Griffin-Lim processor. |
| `HopLength` | Gets the hop length. |
| `MelSpectrogram` | Gets the Mel spectrogram processor. |
| `NFft` | Gets the FFT size. |
| `NumMels` | Gets the number of Mel bins. |
| `STFT` | Gets the STFT processor. |
| `SampleRate` | Gets the sample rate. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AudioToImageSpectrogram(Tensor<>,Int32,Int32)` | Creates a spectrogram suitable for image-based diffusion models. |
| `AudioToSpectrogram(Tensor<>)` | Converts audio waveform to a normalized Mel spectrogram. |
| `DbToPower(Tensor<>)` | Converts dB spectrogram to power. |
| `DenormalizeSpectrogram(Tensor<>)` | Denormalizes a [0, 1] spectrogram back to dB. |
| `DurationToFrames(Double)` | Computes the number of frames for a given duration. |
| `DurationToSamples(Double)` | Computes the number of samples for a given duration. |
| `FramesToDuration(Int32)` | Computes the duration of audio from spectrogram dimensions. |
| `GetMelFrequencyAxis` | Gets the frequency axis values for a Mel spectrogram. |
| `GetTimeAxis(Int32)` | Gets the time axis values for a spectrogram. |
| `NormalizeAudio(Tensor<>,Double)` | Normalizes audio to a peak amplitude. |
| `NormalizeSpectrogram(Tensor<>)` | Normalizes a dB spectrogram to [0, 1] range. |
| `PadOrTruncate(Tensor<>,Int32)` | Pads or truncates audio to a specific length. |
| `PadOrTruncate(Tensor<>,Int32,)` | Pads or truncates audio to a specific length. |
| `PowerToDb(Tensor<>)` | Converts power spectrogram to dB. |
| `ResizeSpectrogram(Tensor<>,Int32,Int32)` | Resizes a spectrogram using bilinear interpolation. |
| `SpectrogramToAudio(Tensor<>,Nullable<Int32>)` | Converts a normalized spectrogram back to audio. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides numeric operations for the specific type T. |
| `_griffinLim` | Griffin-Lim algorithm. |
| `_hopLength` | Hop length between frames. |
| `_maxDb` | Maximum dB for normalization. |
| `_melSpec` | Mel spectrogram processor. |
| `_minDb` | Minimum dB for normalization. |
| `_nFft` | FFT size. |
| `_nMels` | Number of Mel bins. |
| `_sampleRate` | Sample rate in Hz. |
| `_stft` | STFT processor. |

