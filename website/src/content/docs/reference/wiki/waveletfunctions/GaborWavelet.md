---
title: "GaborWavelet<T>"
description: "Represents a Gabor wavelet function implementation for time-frequency analysis and signal processing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.WaveletFunctions`

Represents a Gabor wavelet function implementation for time-frequency analysis and signal processing.

## For Beginners

A Gabor wavelet is a special mathematical tool that helps analyze patterns in data.

Think of it like a musical note with a specific pitch and duration:

- It can detect specific patterns (like frequencies) in your data
- It's especially good at finding where in your data a certain pattern occurs
- It's widely used in image processing to detect edges and textures

For example, in image recognition, Gabor wavelets can help detect specific features like edges
oriented in particular directions, making them useful for tasks like fingerprint recognition,
face detection, and texture classification.

## How It Works

The Gabor wavelet is a complex wavelet defined as a sinusoidal function multiplied by a Gaussian window.
It provides excellent time-frequency localization and is widely used in image processing, computer vision, 
texture analysis, and various signal processing applications. This implementation supports
customization of frequency, bandwidth, and phase parameters.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GaborWavelet(Double,Double,Double,Double)` | Initializes a new instance of the `GaborWavelet` class with the specified parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate()` | Calculates the wavelet function value at the specified point. |
| `Decompose(Vector<>)` | Decomposes an input signal into approximation and detail coefficients using the Gabor transform. |
| `GaborFunction(Int32,Boolean)` | Calculates the Gabor function value at the specified point, either real (cosine) or imaginary (sine) part. |
| `GetScalingCoefficients` | Gets the scaling coefficients (real part of the Gabor function) used in the wavelet transform. |
| `GetWaveletCoefficients` | Gets the wavelet coefficients (imaginary part of the Gabor function) used in the wavelet transform. |
| `Reconstruct(Vector<>,Vector<>)` | Reconstructs the original signal from approximation and detail coefficients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lambda` | The wavelength parameter that determines oscillation frequency. |
| `_omega` | The central frequency of the Gabor wavelet. |
| `_psi` | The phase offset of the sinusoidal component. |
| `_sigma` | The standard deviation of the Gaussian envelope. |

