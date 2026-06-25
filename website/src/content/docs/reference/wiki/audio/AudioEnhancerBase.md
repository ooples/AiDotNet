---
title: "AudioEnhancerBase<T>"
description: "Base class for algorithmic audio enhancement (non-neural network based)."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Audio.Enhancement`

Base class for algorithmic audio enhancement (non-neural network based).

## For Beginners

for provides AI safety functionality. Default values follow the original paper settings.

## How It Works

Provides common functionality for all audio enhancers including:

- Frame-based processing with overlap-add
- Streaming mode with state management
- STFT-based analysis/synthesis

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioEnhancerBase(Int32,Int32,Int32,Int32,Double)` | Initializes a new instance of the AudioEnhancerBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | Gets the hardware-accelerated computation engine for vectorized operations. |
| `EnhancementStrength` |  |
| `LatencySamples` |  |
| `NumChannels` |  |
| `SampleRate` | Audio sample rate. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeFFT([])` | Computes FFT of audio frame using FftSharp library (O(N log N) algorithm). |
| `ComputeIFFT([],[])` | Computes inverse FFT using FftSharp library. |
| `CreateHannWindow(Int32)` | Creates a Hann window of the specified size. |
| `Enhance(Tensor<>)` |  |
| `EnhanceWithReference(Tensor<>,Tensor<>)` |  |
| `EstimateNoiseProfile(Tensor<>)` |  |
| `EstimateNoiseSpectrum([])` | Estimates noise spectrum from noise-only audio. |
| `ProcessChunk(Tensor<>)` |  |
| `ProcessOverlapAdd([])` | Processes audio using overlap-add method. |
| `ProcessSpectralFrame([],[])` | Processes a single spectral frame. |
| `ProcessStreamingChunk([])` | Processes a streaming chunk of audio. |
| `ResetState` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Numeric operations for type T. |
| `_bufferPosition` | Current position in input buffer. |
| `_fftSize` | FFT size for spectral analysis. |
| `_hopSize` | Hop size between frames. |
| `_inputBuffer` | Input buffer for streaming mode. |
| `_noiseProfile` | Estimated noise profile for spectral subtraction. |
| `_outputBuffer` | Output buffer for overlap-add. |
| `_window` | Window function coefficients. |

