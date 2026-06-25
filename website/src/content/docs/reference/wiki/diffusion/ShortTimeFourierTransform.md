---
title: "ShortTimeFourierTransform<T>"
description: "Short-Time Fourier Transform (STFT) for analyzing audio signals over time."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Audio`

Short-Time Fourier Transform (STFT) for analyzing audio signals over time.

## For Beginners

Audio signals like music or speech change over time.
While a regular FFT tells you which frequencies are in the entire signal,
it doesn't tell you WHEN those frequencies occur.

The STFT solves this by:

1. Cutting the audio into small overlapping pieces (frames)
2. Applying a window function to each frame (reduces edge artifacts)
3. Computing FFT on each windowed frame
4. Stacking the results to form a spectrogram (time vs. frequency)

Usage:
```cs
var stft = new ShortTimeFourierTransform<float>(nFft: 2048, hopLength: 512);
var spectrogram = stft.Forward(audioSignal);
// spectrogram.Shape = [numFrames, nFft/2 + 1] (complex values)

// To reconstruct audio from spectrogram:
var reconstructed = stft.Inverse(spectrogram);
```

## How It Works

The STFT breaks a signal into short overlapping segments and computes the
Fourier transform of each segment. This reveals how the frequency content
of a signal changes over time.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ShortTimeFourierTransform(Int32,Nullable<Int32>,Nullable<Int32>,IWindowFunction<>,Boolean,PaddingMode)` | Initializes a new STFT processor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | IEngine for GPU-accelerated FFT operations. |
| `HopLength` | Gets the hop length. |
| `NFft` | Gets the FFT size. |
| `NumFrequencyBins` | Gets the number of frequency bins (nFft / 2 + 1). |
| `WindowTensor` | Gets the window tensor for GPU operations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateNumFrames(Int32)` | Calculates the number of frames for a given signal length. |
| `CalculateSignalLength(Int32)` | Calculates signal length from number of frames. |
| `ComputeMagnitude(Tensor<Complex<>>)` | Computes magnitude from complex spectrogram. |
| `ExtractPhase(Tensor<Complex<>>)` | Extracts phase from complex spectrogram. |
| `Forward(Tensor<>)` | Computes the Short-Time Fourier Transform of a signal. |
| `ForwardBatched(Tensor<>)` | Computes STFT for a batch of signals. |
| `ForwardSingle([])` | Computes STFT for a single signal. |
| `Inverse(Tensor<Complex<>>,Nullable<Int32>)` | Computes the Inverse Short-Time Fourier Transform (overlap-add reconstruction). |
| `InverseBatched(Tensor<Complex<>>,Nullable<Int32>)` | Computes ISTFT for a batch of spectrograms. |
| `InverseFromMagnitudeAndPhase(Tensor<>,Tensor<>,Nullable<Int32>)` | Reconstructs audio signal from magnitude and phase spectrograms. |
| `InverseSingle(Tensor<Complex<>>,Nullable<Int32>)` | Computes ISTFT for a single spectrogram. |
| `Magnitude(Tensor<>)` | Computes the magnitude spectrogram. |
| `MagnitudeAndPhase(Tensor<>,Tensor<>,Tensor<>)` | Computes magnitude and phase spectrograms simultaneously. |
| `PadSignal([],Int32,Int32,PaddingMode)` | Pads a signal according to the specified mode. |
| `PolarToComplex(Tensor<>,Tensor<>)` | Creates complex spectrogram from magnitude and phase. |
| `Power(Tensor<>)` | Computes the power spectrogram (magnitude squared). |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides numeric operations for the specific type T. |
| `_center` | Whether to center the signal by padding. |
| `_fft` | FFT implementation (fallback for non-GPU operations). |
| `_hopLength` | Number of samples between successive frames. |
| `_nFft` | FFT size (number of frequency bins). |
| `_padMode` | Padding mode when centering. |
| `_window` | Window function coefficients. |
| `_windowLength` | Length of the window (defaults to nFft). |
| `_windowTensor` | Window function as a tensor (for IEngine operations). |

