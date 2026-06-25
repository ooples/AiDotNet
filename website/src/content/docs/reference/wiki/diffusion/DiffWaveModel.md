---
title: "DiffWaveModel<T>"
description: "DiffWave model for high-quality audio waveform synthesis using diffusion."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Audio`

DiffWave model for high-quality audio waveform synthesis using diffusion.

## For Beginners

DiffWave generates audio (like speech or music)
directly as a waveform - the actual audio signal that speakers play.

Unlike spectrograms (visual representations of sound), DiffWave creates:

- Raw audio samples that can be played directly
- High-quality, natural-sounding audio
- Various audio types: speech, music, sound effects

How it works:

1. Start with random noise (static)
2. Gradually refine it into clear audio
3. Use dilated convolutions to understand audio context
4. Optionally condition on mel-spectrograms or text

Applications:

- Text-to-speech synthesis
- Music generation
- Audio super-resolution
- Neural vocoders

## How It Works

DiffWave is a versatile diffusion model for raw audio waveform synthesis.
It uses a non-autoregressive architecture with dilated convolutions to
achieve high-quality audio generation with fast inference.

Technical details:

- Non-autoregressive: generates all samples in parallel
- Dilated convolutions: capture long-range audio dependencies
- Mel-spectrogram conditioning: for speech synthesis
- Fast inference compared to autoregressive models
- Supports variable-length audio generation

Reference: Kong et al., "DiffWave: A Versatile Diffusion Model for Audio Synthesis", 2020

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiffWaveModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,Int32,Int32,Int32,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of DiffWaveModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |
| `SampleRate` | Gets the sample rate in Hz. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GenerateAudio(Int32,Int32,Nullable<Int32>)` | Generates unconditional audio. |
| `GenerateBatch(Int32,Int32,Int32,Nullable<Int32>)` | Generates a batch of audio samples. |
| `GenerateFromMelSpectrogram(Tensor<>,Nullable<Int32>,Int32,Nullable<Int32>)` | Generates audio from a mel-spectrogram (vocoder mode). |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `InitializeLayers(Int32,Int32,Int32,Int32,Nullable<Int32>)` | Initializes the model layers. |
| `PredictNoise(Tensor<>,Int32)` |  |
| `SampleNoise(Int32[],Random)` | Samples noise for audio generation. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DEFAULT_SAMPLE_RATE` | Default sample rate in Hz. |
| `_dilationCycle` | Dilation cycle length. |
| `_lastInputShape` | Last audio input shape seen by `Int32)`. |
| `_melChannels` | Number of mel-spectrogram channels for conditioning. |
| `_network` | The diffusion network. |
| `_residualChannels` | Number of residual channels. |
| `_residualLayers` | Number of residual layers. |

