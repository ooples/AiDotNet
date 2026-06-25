---
title: "MusicAnalysisBase<T>"
description: "Base class for music analysis algorithms (beat tracking, chord recognition, key detection)."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Audio.MusicAnalysis`

Base class for music analysis algorithms (beat tracking, chord recognition, key detection).

## For Beginners

Music analysis is about understanding music computationally:

- Beat tracking: Finding the rhythm/pulse of music
- Chord recognition: Identifying the harmony
- Key detection: Finding the musical key (C major, A minor, etc.)

This base class provides:

- Common spectral feature extractors
- Chromagram computation for harmonic analysis
- Onset detection utilities

## How It Works

Music analysis algorithms extract musical information from audio signals.
Unlike neural network models, many traditional music analysis methods use
signal processing techniques that don't require training.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MusicAnalysisBase` | Initializes a new instance of the MusicAnalysisBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ChromaExtractor` | Gets the chromagram extractor for harmonic analysis. |
| `Engine` | Gets the hardware-accelerated computation engine for vectorized operations. |
| `FftSize` | Gets or sets the FFT size for spectral analysis. |
| `HopLength` | Gets or sets the hop length for frame-based analysis. |
| `SampleRate` | Gets or sets the expected sample rate for input audio. |
| `SpectralExtractor` | Gets the spectral feature extractor. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeOnsetStrength(Tensor<>)` | Computes onset strength envelope for beat/onset detection. |
| `ComputeTempogram([],Double,Double)` | Computes tempogram for tempo estimation. |
| `ExtractChromagram(Tensor<>)` | Extracts chromagram (pitch class profile) from audio. |
| `FindPeaks([],Double,Int32)` | Finds peaks in a signal (for beat detection, etc.). |
| `FrameToTime(Int32,Nullable<Int32>)` | Converts frame index to time in seconds. |
| `TimeToFrame(Double,Nullable<Int32>)` | Converts time in seconds to frame index. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Operations for the numeric type T. |

