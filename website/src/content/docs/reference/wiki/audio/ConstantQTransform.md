---
title: "ConstantQTransform<T>"
description: "Constant-Q Transform (CQT) for music analysis with logarithmic frequency resolution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Features`

Constant-Q Transform (CQT) for music analysis with logarithmic frequency resolution.

## For Beginners

The CQT is perfect for music analysis because:

- Musical notes are logarithmically spaced (each octave doubles frequency)
- Low notes get wide bins, high notes get narrow bins (matches perception)
- Makes chord/key detection much easier than FFT

Usage:

## How It Works

Unlike the FFT which has linear frequency spacing, the CQT uses logarithmic spacing
where each frequency bin is a constant ratio (Q factor) above the previous one.
This matches how humans perceive pitch: octaves are equally spaced.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConstantQTransform(Int32,Double,Int32,Int32,Int32,WindowType)` | Creates a new Constant-Q Transform instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` |  |
| `Frequencies` | Gets the center frequencies for each bin. |
| `NumBins` | Gets the number of frequency bins in the CQT output. |
| `QFactor` | Gets the Q factor (quality factor) for this CQT. |
| `SampleRate` | Gets the sample rate. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DeepCopy` |  |
| `GetMidiNote(Int32)` | Gets the MIDI note number for a given bin index. |
| `GetNoteName(Int32)` | Gets the note name for a given bin index. |
| `GetParameters` |  |
| `Predict(Tensor<>)` |  |
| `SetParameters(Vector<>)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `Transform(Tensor<>)` | Computes the Constant-Q Transform of an audio signal. |
| `TransformComplex(Tensor<>)` | Computes the complex CQT (with phase information). |
| `TransformDb(Tensor<>,Double,Double)` | Computes CQT in decibels (log scale). |
| `TransformPower(Tensor<>,Double)` | Computes CQT with power spectrum (magnitude squared). |
| `WithParameters(Vector<>)` |  |

