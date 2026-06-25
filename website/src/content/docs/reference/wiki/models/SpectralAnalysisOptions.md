---
title: "SpectralAnalysisOptions<T>"
description: "Configuration options for spectral analysis of time series data, which transforms time-domain signals into the frequency domain to identify periodic components and patterns."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for spectral analysis of time series data, which transforms time-domain signals
into the frequency domain to identify periodic components and patterns.

## For Beginners

Spectral analysis helps identify repeating patterns in your time series data.

Time series data shows how values change over time, but sometimes it's hard to see patterns:

- A stock price might fluctuate in ways that look random
- Sensor readings might contain multiple overlapping cycles
- Audio signals contain many frequencies mixed together

Spectral analysis transforms this data from the time domain to the frequency domain:

- Instead of seeing "what happened when"
- You see "which cycles/frequencies are present and how strong they are"

This is like:

- Breaking down a musical chord into individual notes
- Identifying that a stock has both weekly and quarterly patterns
- Finding that a machine vibrates at specific frequencies when it's about to fail

This class lets you configure how this transformation happens, controlling the
trade-offs between frequency precision, time precision, and other factors.

## How It Works

Spectral analysis is a technique used to decompose a time series signal into its constituent frequency 
components, revealing periodic patterns that might not be apparent in the original time domain representation. 
This is typically accomplished using the Fast Fourier Transform (FFT) algorithm, which efficiently computes 
the Discrete Fourier Transform of a signal. The resulting frequency spectrum shows the amplitude and phase 
of different frequency components present in the signal. This class provides configuration options for 
spectral analysis, including the FFT size, window function selection, and overlap settings for spectrograms 
or short-time Fourier transforms. These options allow fine-tuning of the spectral analysis to balance 
frequency resolution, time resolution, and spectral leakage based on the specific characteristics of the 
time series being analyzed.

## Properties

| Property | Summary |
|:-----|:--------|
| `NFFT` | Gets or sets the number of points used in the Fast Fourier Transform (FFT). |
| `OverlapPercentage` | Gets or sets the percentage of overlap between adjacent segments in spectrograms or short-time Fourier transforms. |
| `SamplingRate` | Gets or sets the sampling rate of the time series data. |
| `UseWindowFunction` | Gets or sets whether to apply a window function to the signal before spectral analysis. |
| `WindowFunction` | Gets or sets the window function to apply to the signal before spectral analysis. |

