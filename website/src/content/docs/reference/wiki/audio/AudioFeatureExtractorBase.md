---
title: "AudioFeatureExtractorBase<T>"
description: "Base class for audio feature extractors providing common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Audio.Features`

Base class for audio feature extractors providing common functionality.

## For Beginners

for provides AI safety functionality. Default values follow the original paper settings.

## How It Works

This base class provides:

- Common audio processing utilities (windowing, framing)
- Numeric operations through INumericOperations<T>
- Sample rate and FFT configuration

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioFeatureExtractorBase(AudioFeatureOptions)` | Initializes a new instance of the AudioFeatureExtractorBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | Gets the hardware-accelerated computation engine for vectorized operations. |
| `FeatureDimension` |  |
| `FftSize` | Gets the FFT size. |
| `HopLength` | Gets the hop length between frames. |
| `Name` |  |
| `SampleRate` |  |
| `WindowLength` | Gets the window length. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeNumFrames(Int32)` | Computes the number of frames that will be produced for a given audio length. |
| `CreateHammingWindow(Int32)` | Creates a Hamming window of the specified length. |
| `CreateHannWindow(Int32)` | Creates a Hann window of the specified length. |
| `CreateMelFilterbank(Int32,Int32,Int32,Double,Nullable<Double>)` | Creates mel filterbank. |
| `Extract(Tensor<>)` |  |
| `Extract(Vector<>)` |  |
| `ExtractAsync(Tensor<>,CancellationToken)` |  |
| `ExtractFrame([],Int32,[])` | Extracts a single frame from the audio signal. |
| `HzToMel(Double)` | Converts frequency in Hz to mel scale. |
| `MelToHz(Double)` | Converts mel scale frequency to Hz. |
| `PadAudioCenter([])` | Pads audio for center-aligned frames. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Numeric operations for the current type. |
| `Options` | The audio feature extraction options. |

