---
title: "PitchDetectorBase<T>"
description: "Base class for pitch detection implementations."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Audio.Pitch`

Base class for pitch detection implementations.

## For Beginners

for provides AI safety functionality. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PitchDetectorBase(Int32,Double,Double)` | Initializes a new PitchDetectorBase. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | Gets the hardware-accelerated computation engine for vectorized operations. |
| `MaxPitch` |  |
| `MinPitch` |  |
| `SampleRate` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectPitch(Tensor<>)` |  |
| `DetectPitchInternal(Double[])` | Detects pitch from audio frame data. |
| `DetectPitchWithConfidence(Tensor<>)` |  |
| `ExtractDetailedPitchContour(Tensor<>,Int32)` |  |
| `ExtractPitchContour(Tensor<>,Int32)` |  |
| `GetCentsDeviation()` |  |
| `MidiToPitch(Double)` |  |
| `PitchToMidi()` |  |
| `PitchToNoteName()` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `A4Frequency` | Reference frequency for A4 (440 Hz standard). |
| `A4MidiNote` | MIDI note number for A4. |
| `NoteNames` | Note names for display. |
| `NumOps` | Numeric operations for type T. |

